"""MEASURED: check the simulator's roofline model against this machine's GPU.

1. Measure the two ceilings of the roofline on this GPU: dense bf16 GEMM throughput and
   device-memory copy bandwidth.
2. Measure Qwen3-0.6B prefill latency vs prompt length and decode-step latency vs context.
3. Predict both with ``disagg.sim.hardware.iteration_time`` using the measured ceilings at
   100% efficiency, and report measured / roofline.  The ratio for large prefills is the
   achieved MFU; for decode it shows how far an un-optimised HF eager loop is from the
   memory roof (it is launch/CPU-overhead bound -- which is why production engines use
   CUDA graphs and batching).
4. Separate *GPU kernel time* (sum of CUDA kernel durations from torch.profiler) from wall
   time.  When wall >> kernel time the GPU is idle waiting for the Python host to launch
   the next of ~2000 kernels per step: the step is host-bound, not memory- or compute-bound.

Writes results/calibration.json.  Loads a model -> run through the shared GPU lock.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from disagg.engine import Engine  # noqa: E402
from disagg.sim.hardware import GPU, QWEN3_06B, decode_time, prefill_time  # noqa: E402


def timed(fn, repeats: int = 5) -> float:
    fn()  # warm-up
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def kernel_ms(fn) -> float:
    """Total CUDA kernel time of one call of ``fn`` (ms), from torch.profiler."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(e.self_device_time_total for e in prof.key_averages()) / 1e3


def peak_gemm_flops(n: int = 8192) -> float:
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    t = timed(lambda: a @ b, repeats=10)
    del a, b
    return 2 * n**3 / t


def peak_copy_bw(nbytes: int = 1 << 30) -> float:
    src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    t = timed(lambda: dst.copy_(src), repeats=10)
    del src, dst
    return 2 * nbytes / t  # a copy reads and writes every byte


def main() -> None:
    flops, bw = peak_gemm_flops(), peak_copy_bw()
    torch.cuda.empty_cache()
    print(f"measured ceilings: {flops / 1e12:.1f} TFLOP/s bf16 GEMM, {bw / 1e9:.0f} GB/s copy", flush=True)
    props = torch.cuda.get_device_properties(0)
    local = GPU(props.name, flops, bw, props.total_memory, mfu=1.0, mbu=1.0, overhead_s=0.0)

    eng = Engine()
    prompt = list(range(1000, 1000 + 2048))
    rows = []
    for T in (128, 512, 2048):
        t = timed(lambda: eng.prefill(prompt[:T]), repeats=3)
        roof = prefill_time(local, QWEN3_06B, [T])
        rows.append({"phase": "prefill", "tokens": T, "measured_ms": t * 1e3, "roofline_ms": roof * 1e3,
                     "gpu_kernel_ms": kernel_ms(lambda: eng.prefill(prompt[:T]))})
        print(rows[-1], flush=True)
    for ctx in (128, 2048):
        tok, cache = eng.prefill(prompt[:ctx])
        state = {"tok": tok}

        def step() -> None:
            state["tok"] = eng.decode_step(cache, state["tok"])

        t = timed(step, repeats=10)  # cache grows by ~12 tokens: negligible vs ctx
        roof = decode_time(local, QWEN3_06B, 1, ctx)
        rows.append({"phase": "decode", "tokens": ctx, "measured_ms": t * 1e3, "roofline_ms": roof * 1e3,
                     "gpu_kernel_ms": kernel_ms(step)})
        print(rows[-1], flush=True)
        del cache
    for r in rows:
        r["roofline_over_measured"] = round(r["roofline_ms"] / r["measured_ms"], 3)
        r["roofline_over_kernel"] = round(r["roofline_ms"] / r["gpu_kernel_ms"], 3)
    out = {"device": props.name, "torch_threads": torch.get_num_threads(), "cpu_count": os.cpu_count(),
           "note": "shared machine: other jobs may run concurrently; wall time includes host launch overhead", "torch": torch.__version__, "peak_gemm_tflops": flops / 1e12,
           "peak_copy_GBps": bw / 1e9, "model": "Qwen/Qwen3-0.6B (bf16, HF eager, batch 1)",
           "weight_bytes_model": QWEN3_06B.weight_bytes, "peak_cuda_GiB": torch.cuda.max_memory_allocated() / 2**30,
           "rows": rows}
    eng.close()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "calibration.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
