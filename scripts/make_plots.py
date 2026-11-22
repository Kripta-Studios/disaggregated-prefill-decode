"""Render results/*.json|csv into small PNGs for the README (CPU only, no model)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"
BLUE, ORANGE, GREY, GREEN = "#2b6cb0", "#dd6b20", "#718096", "#2f855a"


def kv_transfer() -> None:
    rows = pd.DataFrame(json.loads((R / "kv_transfer.json").read_text())["rows"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (name, df) in zip(axes, rows.groupby("transport")):
        bottom = None
        for col, c in (("gather_ms", GREY), ("d2h_ms", BLUE), ("wire_ms", ORANGE), ("h2d_ms", GREEN)):
            ax.bar(df["tokens"].astype(str), df[col], bottom=bottom, color=c, label=col.replace("_ms", ""))
            bottom = df[col] if bottom is None else bottom + df[col]
        ax.set_title(f"KV handoff, {name} (measured)")
        ax.set_xlabel("prompt tokens (KV = 0.115 MB/token)")
    axes[0].set_ylabel("ms (median of 5)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(R / "kv_transfer.png", dpi=110)


def pd_ratio() -> None:
    df = pd.read_csv(R / "sim_pd_ratio.csv")
    panels = list(df.groupby(["workload", "tpot_slo_ms"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4), sharey=False)
    for ax, ((wl, slo), g) in zip(axes, panels):
        d = g[g.kind == "disagg"]
        ax.plot(d["n_prefill"].astype(int), d["goodput_per_gpu"], "o-", color=BLUE, label="disagg P:(8-P)")
        for pol, ls in (("prefill_first", "--"), ("chunked", ":")):
            v = g[(g.kind == "colocated") & (g.policy == pol)]["goodput_per_gpu"].iloc[0]
            ax.axhline(v, color=ORANGE, ls=ls, label=f"colocated {pol}")
        ax.set_title(f"{wl}, TPOT<={slo:.0f}ms", fontsize=9)
        ax.set_xlabel("# prefill GPUs (of 8 H100)")
    axes[0].set_ylabel("goodput / GPU (req/s)  [SIMULATED]")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(R / "sim_pd_ratio.png", dpi=110)


def hetero() -> None:
    df = pd.read_csv(R / "sim_hetero.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [ORANGE if k == "colocated" else (GREEN if n == 0 else BLUE) for k, n in zip(df.kind, df.n_l40s)]
    ax.barh(df["config"], df["goodput_per_cost"], color=colors)
    ax.set_xlabel("goodput per relative-cost unit (req/s)  [SIMULATED, L40S price = 0.4 x H100 assumed]")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(R / "sim_hetero.png", dpi=110)


if __name__ == "__main__":
    for fn in (kv_transfer, pd_ratio, hetero):
        try:
            fn()
            print("ok", fn.__name__)
        except FileNotFoundError as e:
            print("skip", fn.__name__, e)
