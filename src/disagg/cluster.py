"""The real mini-cluster: prefill worker(s), decode worker(s), a colocated baseline, a router.

Process layout (disaggregated, 1P:1D on one GPU):

    router (this process, no torch/CUDA)
      |-- req_q -->  prefill worker  -- KV via transport -->  decode worker
      '<------------------ event_q (first token, KV stats, tokens) ---------'

Every worker is a separate OS process with its own CUDA context and its own copy of the
weights, exactly like separate prefill/decode instances in DistServe / Dynamo / llm-d.
Timestamps are ``time.perf_counter()``; on Windows (QueryPerformanceCounter) and Linux
(CLOCK_MONOTONIC) that clock is system-wide, so values from different processes compare.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import random
import time
from dataclasses import dataclass, field

import numpy as np

from .transport import ShmSpec, TcpSpec

KV_BYTES_PER_TOKEN_QWEN3_06B = 2 * 28 * 8 * 128 * 2  # 2(K,V) * layers * kv_heads * head_dim * bf16


@dataclass
class Request:
    rid: int
    prompt: list[int]
    max_new: int  # total generated tokens, including the one produced by prefill
    decode_idx: int = 0
    t_arrive: float = 0.0


# ------------------------------------------------------------------------------ workers
def _load(model_id: str):
    from .engine import Engine

    eng = Engine(model_id)
    tok, cache = eng.prefill(list(range(1000, 1064)))  # warm up kernels / allocator
    for _ in range(3):
        tok = eng.decode_step(cache, tok)
    return eng


def _report_memory(event_q, name: str) -> None:
    """Peak host RSS and peak CUDA allocation of this worker (resource discipline is measured)."""
    import psutil
    import torch

    mi = psutil.Process().memory_info()
    event_q.put(("mem", name, getattr(mi, "peak_wset", mi.rss) / 2**30, torch.cuda.max_memory_allocated() / 2**30))


def _decode_iteration(eng, active: list[dict], event_q) -> None:
    """One decode iteration: every active request advances by one token.

    Without batching, iteration time grows linearly with the number of active requests; a real
    engine batches them into one forward pass (and that batch is what makes decode cheaper per
    token -- see the roofline section of the README)."""
    for r in list(active):
        r["last"] = eng.decode_step(r["cache"], r["last"])
        r["tokens"].append(r["last"])
        event_q.put(("tok", r["rid"], time.perf_counter()))
        r["remaining"] -= 1
        if r["remaining"] == 0:
            event_q.put(("done", r["rid"], r["tokens"]))
            active.remove(r)  # drops the last reference to its KV cache


def prefill_worker(model_id: str, req_q, specs: list, event_q, name: str = "prefill") -> None:
    from .kvcodec import cache_to_layers, pack_kv

    eng = _load(model_id)
    senders = [s.make_sender() for s in specs]
    event_q.put(("ready", name))
    while (req := req_q.get()) is not None:
        t0 = time.perf_counter()
        tok, cache = eng.prefill(req.prompt)
        t1 = time.perf_counter()
        event_q.put(("first", req.rid, t1, tok, t1 - t0))
        meta, payload, pt = pack_kv(cache_to_layers(cache))
        del cache
        meta.update(rid=req.rid, first=tok, remaining=req.max_new - 1, gather_s=pt.gather_s, d2h_s=pt.d2h_s)
        senders[req.decode_idx].send(meta, memoryview(payload.numpy()))
        del payload
    for s in senders:
        s.send({"stop": True}, b"")
        s.close()
    _report_memory(event_q, name)
    eng.close()


def decode_worker(model_id: str, spec, event_q, name: str = "decode") -> None:
    from .kvcodec import layers_to_cache, unpack_kv

    eng = _load(model_id)
    rx = spec.make_receiver()
    event_q.put(("ready", name))
    active: list[dict] = []
    stopped = False
    while not (stopped and not active):
        # Admit every KV that has landed; only block (briefly) when there is nothing to decode.
        while (item := rx.recv(timeout=0 if active else 0.5)) is not None:
            meta, payload = item
            if meta.get("stop"):
                stopped = True
                break
            cache = layers_to_cache(unpack_kv(meta, payload, eng.device), eng.config)
            del payload
            rx.release(meta)  # slot credit back to the prefill side
            t_ready = time.perf_counter()
            event_q.put(("kv", meta["rid"], meta["nbytes"], meta["gather_s"], meta["d2h_s"],
                         meta["t_recv_done"] - meta["t_send_start"], t_ready - meta["t_recv_done"]))
            active.append({"rid": meta["rid"], "cache": cache, "last": meta["first"],
                           "remaining": meta["remaining"], "tokens": [meta["first"]]})
        if active:
            _decode_iteration(eng, active, event_q)
    _report_memory(event_q, name)
    rx.close()
    eng.close()


def colocated_worker(model_id: str, req_q, event_q, name: str = "colocated") -> None:
    """Baseline: one instance does both phases with *prefill-first* scheduling (what vLLM did
    before chunked prefill): an arriving prompt runs immediately and stalls every decode."""
    eng = _load(model_id)
    event_q.put(("ready", name))
    active: list[dict] = []
    stopped = False
    while not (stopped and not active):
        while True:
            try:
                req = req_q.get(timeout=0 if (active or stopped) else 0.5)
            except queue.Empty:
                break
            if req is None:
                stopped = True
                break
            t0 = time.perf_counter()
            tok, cache = eng.prefill(req.prompt)
            t1 = time.perf_counter()
            event_q.put(("first", req.rid, t1, tok, t1 - t0))
            active.append({"rid": req.rid, "cache": cache, "last": tok,
                           "remaining": req.max_new - 1, "tokens": [tok]})
        if active:
            _decode_iteration(eng, active, event_q)
    _report_memory(event_q, name)
    eng.close()


# ------------------------------------------------------------------------------- router
@dataclass
class Workload:
    n_requests: int = 24
    rate_rps: float = 2.0  # Poisson arrival rate
    prompt_lens: tuple[int, ...] = (256, 512, 1024)
    max_new: int = 32
    seed: int = 0
    requests: list[Request] = field(init=False)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        # Random token ids are fine for latency work: cost depends on length, not content.
        self.requests = [Request(i, [rng.randrange(1000, 100_000) for _ in range(rng.choice(self.prompt_lens))],
                                 self.max_new) for i in range(self.n_requests)]
        self.gaps = [rng.expovariate(self.rate_rps) for _ in range(self.n_requests)]


def run(mode: str, wl: Workload, transport: str = "tcp", model_id: str = "Qwen/Qwen3-0.6B",
        n_decode: int = 1, timeout_s: float = 600.0) -> dict:
    """Launch workers, replay the workload, collect per-request metrics, tear everything down."""
    ctx = mp.get_context("spawn")
    event_q, req_q = ctx.Queue(), ctx.Queue()
    procs: list = []
    if mode == "disagg":
        max_tokens = max(len(r.prompt) for r in wl.requests)
        if transport == "tcp":
            specs = [TcpSpec() for _ in range(n_decode)]
        else:  # 2 slots per decode worker: the 3rd in-flight KV waits for a credit
            specs = [ShmSpec(2, max_tokens * KV_BYTES_PER_TOKEN_QWEN3_06B, ctx.Queue(), ctx.Queue())
                     for _ in range(n_decode)]
        procs.append(ctx.Process(target=prefill_worker, args=(model_id, req_q, specs, event_q)))
        procs += [ctx.Process(target=decode_worker, args=(model_id, s, event_q, f"decode{i}"))
                  for i, s in enumerate(specs)]
    elif mode == "colocated":
        procs.append(ctx.Process(target=colocated_worker, args=(model_id, req_q, event_q)))
    else:
        raise ValueError(mode)

    rows = {r.rid: {"rid": r.rid, "prompt_len": len(r.prompt), "times": []} for r in wl.requests}
    outstanding = [0] * n_decode  # router's view of queued+running tokens per decode worker
    cost = {r.rid: len(r.prompt) + r.max_new for r in wl.requests}
    done = 0
    mem: dict[str, tuple[float, float]] = {}

    def handle(ev) -> None:
        nonlocal done
        kind, rid = ev[0], ev[1]
        if kind == "first":
            rows[rid].update(t_first=ev[2], prefill_s=ev[4])
            rows[rid]["times"].append(ev[2])
        elif kind == "tok":
            rows[rid]["times"].append(ev[2])
        elif kind == "kv":
            rows[rid].update(kv_bytes=ev[2], gather_s=ev[3], d2h_s=ev[4], wire_s=ev[5], h2d_s=ev[6])
        elif kind == "mem":
            mem[rid] = (round(ev[2], 2), round(ev[3], 2))  # (peak host GiB, peak CUDA GiB)
        elif kind == "done":
            rows[rid]["tokens"] = ev[2]
            outstanding[rows[rid]["decode_idx"]] -= cost[rid]
            done += 1

    try:
        for p in procs:
            p.start()
        ready = 0
        while ready < len(procs):  # model loading happens here, outside the measured window
            ev = event_q.get(timeout=300)
            ready += ev[0] == "ready"
        t_next = time.perf_counter()
        for r, gap in zip(wl.requests, wl.gaps):
            t_next += gap
            while (now := time.perf_counter()) < t_next:  # drain events while we wait
                try:
                    handle(event_q.get(timeout=t_next - now))
                except queue.Empty:
                    pass
            # Router policy: send the KV to the decode worker with the least outstanding work.
            r.decode_idx = int(np.argmin(outstanding))
            outstanding[r.decode_idx] += cost[r.rid]
            r.t_arrive = time.perf_counter()
            rows[r.rid].update(t_arrive=r.t_arrive, decode_idx=r.decode_idx)
            req_q.put(r)
        req_q.put(None)
        deadline = time.perf_counter() + timeout_s
        while done < len(wl.requests) and time.perf_counter() < deadline:
            handle(event_q.get(timeout=timeout_s))
        for p in procs:
            p.join(timeout=60)
        while True:  # late events (memory reports) arrive after the last token
            try:
                handle(event_q.get(timeout=1.0))
            except queue.Empty:
                break
    finally:
        for p in procs:  # never leave a worker (and its GPU memory) behind
            if p.is_alive():
                p.terminate()
                p.join()
    out = summarize(list(rows.values()))
    out["worker_peak_mem_GiB"] = mem
    return out


def _pct(xs, q) -> float:
    return float(np.percentile(xs, q)) * 1e3 if len(xs) else float("nan")


def summarize(rows: list[dict]) -> dict:
    """Per-request TTFT/ITL/TPOT and KV-transfer stats (all times reported in ms)."""
    ttft, itl, tpot, gap2 = [], [], [], []
    for r in rows:
        t = r["times"]
        ttft.append(r["t_first"] - r["t_arrive"])
        d = np.diff(t)
        itl.extend(d.tolist())
        gap2.append(d[0])  # first->second token: includes the whole KV handoff in disagg mode
        tpot.append((t[-1] - t[0]) / (len(t) - 1))
    out = {
        "n": len(rows),
        "ttft_ms_p50": _pct(ttft, 50), "ttft_ms_p99": _pct(ttft, 99),
        "itl_ms_p50": _pct(itl, 50), "itl_ms_p99": _pct(itl, 99), "itl_ms_max": _pct(itl, 100),
        "tpot_ms_mean": float(np.mean(tpot)) * 1e3,
        "gap_tok1_tok2_ms_p50": _pct(gap2, 50),
        "prefill_ms_mean": float(np.mean([r["prefill_s"] for r in rows])) * 1e3,
    }
    if all("kv_bytes" in r for r in rows):
        for k in ("gather_s", "d2h_s", "wire_s", "h2d_s"):
            out[f"kv_{k[:-2]}_ms_mean"] = float(np.mean([r[k] for r in rows])) * 1e3
        out["kv_MB_mean"] = float(np.mean([r["kv_bytes"] for r in rows])) / 1e6
    out["rows"] = [{k: v for k, v in r.items() if k != "times"} for r in rows]
    return out
