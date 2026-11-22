import math

from disagg.sim import H100, L40S, LLAMA31_8B, Cluster, Workload, attainment, decode_time, goodput, prefill_time, simulate
from disagg.sim.hardware import GPU, iteration_time

m = LLAMA31_8B
IDEAL = GPU("ideal", 1e15, 1e12, 80e9, mfu=1.0, mbu=1.0, overhead_s=0.0)


def test_llama8b_sizes():
    assert abs(m.weight_bytes / 1e9 - 16.06) < 0.05  # 8.03 B params in bf16
    assert m.kv_bytes_per_token == 128 * 1024  # 32 layers * 8 heads * 128 * 2 (K,V) * 2 B


def test_roofline_regimes():
    # One decode token: weight streaming dominates -> time ~= weight_bytes / BW.
    t = iteration_time(IDEAL, m, 1, 1, 0, 0)
    assert math.isclose(t, (m.weight_bytes + m.kv_bytes_per_token) / 1e12, rel_tol=1e-9)
    # A 4k-token prefill on the ideal GPU is compute bound: time grows ~linearly with tokens.
    assert prefill_time(IDEAL, m, [4096]) > 3.5 * prefill_time(IDEAL, m, [1024])
    # Decode batch 1 -> 32 is almost free (same weights, 32x the tokens): memory-bound regime.
    assert decode_time(IDEAL, m, 32, 0) < 1.1 * decode_time(IDEAL, m, 1, 0)


def test_l40s_is_slower_per_decode_step():
    assert decode_time(L40S, m, 64, 64 * 1000) > 3 * decode_time(H100, m, 64, 64 * 1000)


def _check_complete(reqs):
    for r in reqs:
        assert r.t_arr <= r.t_first <= r.t_done and not math.isnan(r.t_done)


def test_every_request_finishes_in_every_mode():
    wl = Workload(rate=5, duration_s=20)
    for cl in (Cluster("colocated", H100, 1), Cluster("colocated", H100, 1, policy="chunked"),
               Cluster("disagg", H100, 1, L40S, 2)):
        _check_complete(simulate(cl, m, wl))


def test_kv_transfer_delay_appears_with_slow_link():
    wl = Workload(rate=2, duration_s=20)
    fast = simulate(Cluster("disagg", H100, 1, H100, 1, link_bw=50e9), m, wl)
    slow = simulate(Cluster("disagg", H100, 1, H100, 1, link_bw=0.5e9), m, wl)
    # 1k-token prompt = 134 MB of KV: +0.27 s at 0.5 GB/s, folded into TPOT (TTFT is unchanged).
    assert sum(r.tpot for r in slow) > sum(r.tpot for r in fast)
    assert math.isclose(sum(r.ttft for r in slow), sum(r.ttft for r in fast), rel_tol=1e-9)


def test_chunked_prefill_cuts_tpot_interference():
    wl = Workload(rate=30, duration_s=20, in_mean=4096, out_mean=128)
    pf = simulate(Cluster("colocated", H100, 1), m, wl)
    ch = simulate(Cluster("colocated", H100, 1, policy="chunked"), m, wl)
    assert max(r.tpot for r in ch) < max(r.tpot for r in pf)


def test_goodput_is_a_rate_meeting_the_slo():
    cl = Cluster("disagg", H100, 1, H100, 1)
    g = goodput(cl, m, 0.5, 0.05, iters=6, duration_s=20)
    assert g > 0
    assert attainment(simulate(cl, m, Workload(g, duration_s=20)), 0.5, 0.05) >= 0.9
