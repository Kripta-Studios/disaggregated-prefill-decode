"""Discrete-event simulator of colocated vs disaggregated serving.

Every instance is an iteration-level engine (like vLLM/SGLang): it repeatedly picks a batch,
"runs" it for ``hardware.iteration_time(...)`` seconds of simulated time, then applies the
result.  Three instance kinds:

* ``ColocatedInstance`` -- prefill and decode on the same GPU, with either
    - ``prefill_first`` scheduling: waiting prompts run as soon as they fit and stall every
      running decode (TPOT spikes = prefill/decode interference), or
    - ``chunked`` scheduling (Sarathi-Serve): each iteration carries all decodes plus a prompt
      chunk that fills a token budget, bounding the stall at the cost of slower prefill.
* ``PrefillInstance`` -- prompts only, FCFS batches up to a token budget; then the KV cache is
  shipped over a per-instance NIC link (serialised FIFO, latency + bytes / bandwidth).
* ``DecodeInstance``  -- continuous batching of decodes only.

Admission reserves KV for prompt + output (no preemption), which keeps the model simple and
never over-commits memory.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .hardware import GPU, Model, decode_time, iteration_time, prefill_time


@dataclass
class Req:
    rid: int
    t_arr: float
    n_in: int
    n_out: int
    t_first: float = math.nan
    t_done: float = math.nan
    prefilled: int = 0  # tokens of the prompt already processed (chunked prefill)
    reserved: bool = False  # KV for this request already reserved (chunked prefill)

    @property
    def ttft(self) -> float:
        return self.t_first - self.t_arr

    @property
    def tpot(self) -> float:
        return (self.t_done - self.t_first) / max(self.n_out - 1, 1)


class EventLoop:
    def __init__(self) -> None:
        self.now = 0.0
        self._q: list = []
        self._seq = itertools.count()

    def at(self, t: float, fn, *args) -> None:
        heapq.heappush(self._q, (t, next(self._seq), fn, args))

    def run(self) -> None:
        while self._q:
            self.now, _, fn, args = heapq.heappop(self._q)
            fn(*args)


class _Engine:
    """Shared machinery: KV accounting and the busy/idle iteration loop."""

    def __init__(self, loop: EventLoop, gpu: GPU, m: Model, max_batch: int = 256):
        self.loop, self.gpu, self.m, self.max_batch = loop, gpu, m, max_batch
        self.kv_free = m.kv_capacity_tokens(gpu)
        if self.kv_free <= 0:
            raise ValueError(f"{m.name} does not fit on {gpu.name}")
        self.busy = False
        # running decodes: heap of (finish_step, rid, req); ctx_sum = KV tokens attended per step
        self.running: list = []
        self.step = 0
        self.ctx_sum = 0
        self.busy_time = 0.0

    def load(self) -> int:
        return len(self.running)

    def kick(self) -> None:
        if not self.busy and (plan := self.plan()) is not None:
            dur, apply = plan
            self.busy = True
            self.busy_time += dur
            self.loop.at(self.loop.now + dur, self._finish, apply)

    def _finish(self, apply) -> None:
        apply()
        self.busy = False
        self.kick()

    def plan(self):  # -> (duration, apply_fn) | None
        raise NotImplementedError

    def _start_decode(self, r: Req) -> None:
        """Request has its first token and its KV here; it needs n_out - 1 more steps."""
        heapq.heappush(self.running, (self.step + r.n_out - 1, r.rid, r))
        self.ctx_sum += r.n_in + 1

    def _after_decode_step(self) -> None:
        self.step += 1
        self.ctx_sum += len(self.running)  # every running sequence grew by one token
        while self.running and self.running[0][0] <= self.step:
            _, _, r = heapq.heappop(self.running)
            r.t_done = self.loop.now
            self.ctx_sum -= r.n_in + r.n_out
            self.kv_free += r.n_in + r.n_out

    def _decode_plan(self):
        return decode_time(self.gpu, self.m, len(self.running), self.ctx_sum), self._after_decode_step


class DecodeInstance(_Engine):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.waiting: deque[Req] = deque()

    def submit(self, r: Req) -> None:
        self.waiting.append(r)
        self.kick()

    def load(self) -> int:
        return len(self.running) + len(self.waiting)

    def plan(self):
        while self.waiting and len(self.running) < self.max_batch and self.kv_free >= self.waiting[0].n_in + self.waiting[0].n_out:
            r = self.waiting.popleft()
            self.kv_free -= r.n_in + r.n_out
            self._start_decode(r)
        return self._decode_plan() if self.running else None


class PrefillInstance(_Engine):
    def __init__(self, loop, gpu, m, decoders: list[DecodeInstance], link_bw: float, link_lat: float = 20e-6,
                 token_budget: int = 8192):
        super().__init__(loop, gpu, m)
        self.decoders, self.link_bw, self.link_lat, self.budget = decoders, link_bw, link_lat, token_budget
        self.waiting: deque[Req] = deque()
        self.link_free_at = 0.0
        self.transfer_time = 0.0

    def submit(self, r: Req) -> None:
        self.waiting.append(r)
        self.kick()

    def load(self) -> int:
        return sum(r.n_in for r in self.waiting)

    def plan(self):
        batch, toks = [], 0
        while self.waiting and (not batch or toks + self.waiting[0].n_in <= self.budget):
            r = self.waiting.popleft()
            batch.append(r)
            toks += r.n_in
        if not batch:
            return None
        return prefill_time(self.gpu, self.m, [r.n_in for r in batch]), lambda: self._done(batch)

    def _done(self, batch: list[Req]) -> None:
        for r in batch:
            r.t_first = self.loop.now  # the prefill instance emits token #1 to the client
            # KV leaves over this instance's NIC; transfers queue behind each other.
            dur = self.link_lat + r.n_in * self.m.kv_bytes_per_token / self.link_bw
            start = max(self.loop.now, self.link_free_at)
            self.link_free_at = start + dur
            self.transfer_time += dur
            dst = min(self.decoders, key=lambda d: d.load())  # router: least-loaded decode
            self.loop.at(start + dur, dst.submit, r)


class ColocatedInstance(_Engine):
    def __init__(self, loop, gpu, m, policy: str = "prefill_first", token_budget: int = 8192, chunk_budget: int = 2048):
        super().__init__(loop, gpu, m)
        self.policy, self.budget, self.chunk_budget = policy, token_budget, chunk_budget
        self.waiting: deque[Req] = deque()

    def submit(self, r: Req) -> None:
        self.waiting.append(r)
        self.kick()

    def load(self) -> int:
        return len(self.running) + len(self.waiting)

    def _fits(self, r: Req) -> bool:
        return self.kv_free >= r.n_in + r.n_out and len(self.running) < self.max_batch

    def plan(self):
        if self.policy == "prefill_first":
            batch, toks = [], 0
            while self.waiting and self._fits(self.waiting[0]) and (not batch or toks + self.waiting[0].n_in <= self.budget):
                r = self.waiting.popleft()
                self.kv_free -= r.n_in + r.n_out
                batch.append(r)
                toks += r.n_in
            if batch:  # every running decode waits for this whole prefill
                return prefill_time(self.gpu, self.m, [r.n_in for r in batch]), lambda: self._prefilled(batch)
            return self._decode_plan() if self.running else None
        return self._chunked_plan()

    def _prefilled(self, batch: list[Req]) -> None:
        for r in batch:
            r.t_first = self.loop.now
            self._start_decode(r)

    def _chunked_plan(self):
        """Sarathi-style hybrid batch: all decodes + one prompt chunk filling the budget."""
        b = len(self.running)
        room = self.chunk_budget - b
        head = self.waiting[0] if self.waiting else None
        if head is not None and not head.reserved:
            if self._fits(head):
                self.kv_free -= head.n_in + head.n_out  # reserve the whole request on its first chunk
                head.reserved = True
            else:
                head = None
        c = min(room, head.n_in - head.prefilled) if head is not None and room > 0 else 0
        if b == 0 and c == 0:
            return None
        done0 = head.prefilled if c else 0
        attn_ctx = self.ctx_sum + c * done0 + c * c / 2  # decodes + the chunk attending to its prefix
        dur = iteration_time(self.gpu, self.m, b + c, b + (1 if c else 0), attn_ctx, self.ctx_sum + done0)

        def apply() -> None:
            if b:
                self._after_decode_step()
            if c:
                head.prefilled += c
                if head.prefilled == head.n_in:
                    self.waiting.popleft()
                    head.t_first = self.loop.now
                    self._start_decode(head)

        return dur, apply


# ------------------------------------------------------------------------------ driver
@dataclass
class Workload:
    """Synthetic request stream: Poisson arrivals, log-normal prompt/output lengths.

    The trace length is set in *seconds*, not requests: a fixed request count at a high rate
    gives a trace shorter than the SLO, and a backlog that never builds up looks "fine"."""

    rate: float
    duration_s: float = 60.0
    in_mean: float = 1024
    out_mean: float = 256
    sigma: float = 0.6
    seed: int = 0
    reqs: list[Req] = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.n = max(200, int(self.rate * self.duration_s))
        t = np.cumsum(rng.exponential(1 / self.rate, self.n))
        mu = lambda mean: np.log(mean) - self.sigma**2 / 2  # noqa: E731 - lognormal with this mean
        n_in = np.clip(rng.lognormal(mu(self.in_mean), self.sigma, self.n), 16, 8 * self.in_mean).astype(int)
        n_out = np.clip(rng.lognormal(mu(self.out_mean), self.sigma, self.n), 2, 8 * self.out_mean).astype(int)
        self.reqs = [Req(i, float(t[i]), int(n_in[i]), int(n_out[i])) for i in range(self.n)]


@dataclass
class Cluster:
    """``kind`` is 'colocated' (n_p instances of gpu_p, policy) or 'disagg' (n_p prefill on gpu_p,
    n_d decode on gpu_d, connected by links of ``link_bw`` bytes/s per prefill instance)."""

    kind: str
    gpu_p: GPU
    n_p: int
    gpu_d: GPU | None = None
    n_d: int = 0
    policy: str = "prefill_first"
    link_bw: float = 50e9

    @property
    def label(self) -> str:
        if self.kind == "colocated":
            return f"colocated-{self.policy} {self.n_p}x{self.gpu_p.name}"
        return f"disagg {self.n_p}x{self.gpu_p.name} : {self.n_d}x{self.gpu_d.name}"

    def cost(self, price: dict[str, float]) -> float:
        return self.n_p * price[self.gpu_p.name] + (self.n_d * price[self.gpu_d.name] if self.n_d else 0)


def simulate(cl: Cluster, m: Model, wl: Workload) -> list[Req]:
    loop = EventLoop()
    reqs = [Req(r.rid, r.t_arr, r.n_in, r.n_out) for r in wl.reqs]  # fresh copies
    if cl.kind == "colocated":
        front = [ColocatedInstance(loop, cl.gpu_p, m, cl.policy) for _ in range(cl.n_p)]
    else:
        decs = [DecodeInstance(loop, cl.gpu_d, m) for _ in range(cl.n_d)]
        front = [PrefillInstance(loop, cl.gpu_p, m, decs, cl.link_bw) for _ in range(cl.n_p)]
    for r in reqs:
        loop.at(r.t_arr, lambda r=r: min(front, key=lambda i: i.load()).submit(r))
    loop.run()
    return reqs


def attainment(reqs: list[Req], ttft_slo: float, tpot_slo: float) -> float:
    ok = sum(1 for r in reqs if r.ttft <= ttft_slo and r.tpot <= tpot_slo)
    return ok / len(reqs)


def goodput(cl: Cluster, m: Model, ttft_slo: float, tpot_slo: float, target: float = 0.9,
            lo: float = 0.05, hi: float = 400.0, iters: int = 14, **wl_kw) -> float:
    """Largest arrival rate (req/s) at which >= ``target`` of requests meet BOTH SLOs
    (DistServe's definition).  Bisection in log space; attainment is ~monotone in rate."""
    if attainment(simulate(cl, m, Workload(lo, **wl_kw)), ttft_slo, tpot_slo) < target:
        return 0.0
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if attainment(simulate(cl, m, Workload(mid, **wl_kw)), ttft_slo, tpot_slo) >= target:
            lo = mid
        else:
            hi = mid
    return lo
