"""SIMULATED: goodput of colocated vs disaggregated clusters (Llama-3.1-8B, roofline costs).

Three sweeps (pick with --sweep; each fits well inside the GPU-lock 15-min budget):

  pd      8 x H100: colocated (prefill-first, chunked) vs disaggregated P:D = 1:7 ... 7:1,
          for several TPOT SLOs  -> results/sim_pd_ratio.csv
  hetero  H100 prefill pool + L40S decode pool (P in 1..4, D in 1..8), goodput per
          relative-cost unit vs all-H100 baselines            -> results/sim_hetero.csv
  link    KV link bandwidth sensitivity for a fixed disaggregated cluster and long prompts
                                                              -> results/sim_link.csv

No GPU is touched; it is pure-Python discrete-event simulation.  Every goodput search is
independent, so the configurations of a sweep run in parallel on ``--workers`` processes
(default 4, the swarm's per-job process cap).
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disagg.sim import H100, L40S, LLAMA31_8B, Cluster, goodput  # noqa: E402

M = LLAMA31_8B
TTFT_SLO = 0.5  # seconds -- our choice for an interactive chat workload
# Relative hourly price (ASSUMPTION; pass --l40s-price to change). Only the ratio matters.
PRICE = {"H100-SXM": 1.0, "L40S": 0.4}
CHAT = dict(in_mean=1024, out_mean=256)
LONG = dict(in_mean=4096, out_mean=256)


def write(name: str, rows: list[dict]) -> None:
    path = ROOT / "results" / name
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


WORKERS = 4


def gp(cl: Cluster, tpot: float, wl: dict, iters: int = 9) -> float:
    return goodput(cl, M, TTFT_SLO, tpot, iters=iters, duration_s=40, **wl)


def _gp_job(job: tuple) -> float:
    return gp(*job)


def gp_many(jobs: list[tuple]) -> list[float]:
    """Run independent goodput searches (cluster, tpot, workload) in parallel processes."""
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        return list(ex.map(_gp_job, jobs))


def sweep_pd(n_gpus: int = 8) -> None:
    configs = [Cluster("colocated", H100, n_gpus), Cluster("colocated", H100, n_gpus, policy="chunked")]
    configs += [Cluster("disagg", H100, p, H100, n_gpus - p) for p in range(1, n_gpus)]
    cases = [(wl_name, wl, tpot, cl) for wl_name, wl in (("chat", CHAT), ("long", LONG))
             for tpot in (0.025, 0.05) for cl in configs]
    rows = []
    for (wl_name, _, tpot, cl), g in zip(cases, gp_many([(cl, tpot, wl) for _, wl, tpot, cl in cases])):
        rows.append({"workload": wl_name, "tpot_slo_ms": tpot * 1e3, "config": cl.label, "kind": cl.kind,
                     "policy": cl.policy if cl.kind == "colocated" else "", "n_prefill": cl.n_p if cl.kind == "disagg" else "",
                     "n_decode": cl.n_d, "goodput_rps": round(g, 3), "goodput_per_gpu": round(g / n_gpus, 3)})
        print(rows[-1], flush=True)
    write("sim_pd_ratio.csv", rows)


def sweep_hetero() -> None:
    rows = []
    tpot = 0.05  # an 8B model cannot beat ~25 ms/token on an L40S (16 GB of weights / 864 GB/s)
    configs = [Cluster("colocated", H100, 4, policy="chunked"), Cluster("colocated", H100, 4)]
    configs += [Cluster("disagg", H100, p, H100, d) for p, d in ((3, 1), (2, 2))]
    configs += [Cluster("disagg", H100, p, L40S, d) for p in (1, 2, 3) for d in (1, 2, 4, 6, 8) if d >= p]
    for cl, g in zip(configs, gp_many([(cl, tpot, CHAT) for cl in configs])):
        cost = cl.cost(PRICE)
        rows.append({"config": cl.label, "kind": cl.kind, "n_h100": cl.n_p + (cl.n_d if cl.gpu_d is H100 else 0),
                     "n_l40s": cl.n_d if cl.gpu_d is L40S else 0, "rel_cost": cost, "goodput_rps": round(g, 3),
                     "goodput_per_cost": round(g / cost, 3)})
        print(rows[-1], flush=True)
    write("sim_hetero.csv", rows)


def sweep_link() -> None:
    links = ((1.25, "10 GbE"), (12.5, "100 Gb RoCE"), (50, "400 Gb IB (NDR)"), (450, "NVLink-class"))
    cases = [(bw, name, Cluster("disagg", H100, p, H100, d, link_bw=bw * 1e9)) for bw, name in links
             for p, d in ((4, 4), (5, 3))]
    rows = []
    for (bw_gbs, name, cl), g in zip(cases, gp_many([(cl, 0.05, LONG) for _, _, cl in cases])):
        rows.append({"link": name, "link_GBps": bw_gbs, "config": cl.label, "goodput_rps": round(g, 3),
                     "kv_transfer_ms_4k_prompt": round(4096 * M.kv_bytes_per_token / (bw_gbs * 1e9) * 1e3, 2)})
        print(rows[-1], flush=True)
    write("sim_link.csv", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["pd", "hetero", "link"], required=True)
    ap.add_argument("--l40s-price", type=float, default=PRICE["L40S"])
    ap.add_argument("--workers", type=int, default=WORKERS, help="parallel processes (max 4 on the shared box)")
    a = ap.parse_args()
    PRICE["L40S"] = a.l40s_price
    WORKERS = min(a.workers, 4)
    {"pd": sweep_pd, "hetero": sweep_hetero, "link": sweep_link}[a.sweep]()
