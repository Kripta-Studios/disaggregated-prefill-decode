"""MEASURED: the real mini cluster (Qwen3-0.6B, one GPU) -- colocated vs disaggregated.

Runs the same Poisson workload three times:
  colocated     -- one process, prefill-first scheduling
  disagg-tcp    -- prefill process -> KV over loopback TCP -> decode process
  disagg-shm    -- same, KV through a receiver-owned shared-memory arena
and checks that all three produce *identical* greedy tokens (the KV handoff is lossless).

Every worker process is started and joined inside this script.  Run it through the shared
GPU lock (see README); it writes results/mini_cluster.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disagg.cluster import Workload, run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+", default=[0.5, 2.0])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()

    results = []
    for rate in args.rates:
        tokens = {}
        for mode, transport in (("colocated", "-"), ("disagg", "tcp"), ("disagg", "shm")):
            wl = Workload(n_requests=args.n, rate_rps=rate, prompt_lens=(256, 512, 1024, 1536), max_new=args.max_new)
            res = run(mode, wl, transport)
            label = mode if mode == "colocated" else f"{mode}-{transport}"
            tokens[label] = {r["rid"]: r["tokens"] for r in res["rows"]}
            summary = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "rows"}
            summary.update(mode=label, rate_rps=rate)
            print(json.dumps(summary), flush=True)
            results.append(summary)
        ref = tokens["colocated"]
        for label, toks in tokens.items():
            same = sum(toks[i] == ref[i] for i in ref)
            print(f"rate {rate}: {label}: {same}/{len(ref)} requests token-identical to colocated", flush=True)
            for s in results:
                if s["mode"] == label and s["rate_rps"] == rate:
                    s["token_identical_to_colocated"] = f"{same}/{len(ref)}"
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "mini_cluster.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
