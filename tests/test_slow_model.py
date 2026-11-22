"""Real-model end-to-end check. Skipped by default; run with RUN_SLOW=1 through the GPU lock."""

import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="loads Qwen3-0.6B; set RUN_SLOW=1")


def test_disagg_tokens_match_colocated():
    from disagg.cluster import Workload, run

    wl = dict(n_requests=4, rate_rps=4.0, prompt_lens=(64, 300), max_new=8)
    a = run("colocated", Workload(**wl))
    b = run("disagg", Workload(**wl), "shm")
    assert [r["tokens"] for r in a["rows"]] == [r["tokens"] for r in b["rows"]]
    assert b["kv_MB_mean"] > 0
