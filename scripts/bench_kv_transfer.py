"""MEASURED: cost of shipping a Qwen3-0.6B-shaped KV cache through each transport.

No model is loaded: KV tensors are random bf16 tensors with the exact Qwen3-0.6B layout
(28 layers x 8 KV heads x 128 head_dim), created on the GPU.  For each prompt length we time
the four stages of a handoff:

  gather  -- torch.stack of 56 tensors into one contiguous GPU buffer
  d2h     -- one GPU -> pinned host copy (PCIe)
  wire    -- transport send() start -> receiver has the bytes (loopback TCP / shared memory)
  h2d     -- host -> GPU copy + DynamicCache rebuild on the receiver

Sender and receiver live in one process here (receiver = background thread) so the numbers
isolate the transport; ``run_mini_cluster.py`` measures the same stages across processes.
Run through the shared GPU lock (see README).
"""

from __future__ import annotations

import json
import queue
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from disagg.kvcodec import kv_bytes_per_token, layers_to_cache, pack_kv, unpack_kv  # noqa: E402
from disagg.transport import ShmSpec, TcpSpec  # noqa: E402

LAYERS, KV_HEADS, HEAD_DIM = 28, 8, 128
LENGTHS = [128, 256, 512, 1024, 2048]
REPEATS = 5


def one_handoff(layers, tx, rx) -> dict:
    meta, payload, pt = pack_kv(layers)
    tx.send(meta, memoryview(payload.numpy()))
    meta_r, buf = rx.recv(timeout=30)
    t_recv = meta_r["t_recv_done"]
    cache = layers_to_cache(unpack_kv(meta_r, buf, "cuda"))
    torch.cuda.synchronize()
    t_ready = time.perf_counter()
    del buf
    rx.release(meta_r)
    del cache
    return {"gather": pt.gather_s, "d2h": pt.d2h_s, "wire": t_recv - meta_r["t_send_start"], "h2d": t_ready - t_recv}


def main() -> None:
    max_bytes = max(LENGTHS) * kv_bytes_per_token(LAYERS, KV_HEADS, HEAD_DIM)
    specs = {"tcp": TcpSpec(), "shm": ShmSpec(1, max_bytes, queue.Queue(), queue.Queue())}
    results = []
    for name, spec in specs.items():
        rx = spec.make_receiver()  # receiver first: it owns the listening socket / the arena
        tx = spec.make_sender()
        for T in LENGTHS:
            layers = [(torch.randn(1, KV_HEADS, T, HEAD_DIM, device="cuda", dtype=torch.bfloat16),
                       torch.randn(1, KV_HEADS, T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)) for _ in range(LAYERS)]
            one_handoff(layers, tx, rx)  # warm-up (allocator, page faults on the arena)
            runs = [one_handoff(layers, tx, rx) for _ in range(REPEATS)]
            row = {"transport": name, "tokens": T, "MB": T * kv_bytes_per_token(LAYERS, KV_HEADS, HEAD_DIM) / 1e6}
            for k in ("gather", "d2h", "wire", "h2d"):
                row[f"{k}_ms"] = round(statistics.median(r[k] for r in runs) * 1e3, 3)
            row["total_ms"] = round(sum(row[f"{k}_ms"] for k in ("gather", "d2h", "wire", "h2d")), 3)
            row["wire_GBps"] = round(row["MB"] / 1e3 / (row["wire_ms"] / 1e3), 2)
            results.append(row)
            print(row, flush=True)
            del layers
            torch.cuda.empty_cache()
        tx.close()
        rx.close()
    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "repeats": REPEATS,
           "kv_bytes_per_token": kv_bytes_per_token(LAYERS, KV_HEADS, HEAD_DIM), "rows": results}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "kv_transfer.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
