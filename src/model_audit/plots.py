"""Figures for docs/: PSI/CSI bars, ROC and KS, calibration."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .pipeline import AuditOutput  # noqa: E402

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#6b7280"
STATUS_COL = {"Pass": "#4b8f6b", "Watch": "#e0a82e", "Fail": "#c0392b"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def psi_bars(out: AuditOutput, path: Path, cfg: dict):
    r = {x.check_id: x for x in out.results}
    names = ["Score (PSI)", *out.tables["csi"]["feature"]]
    vals = [r["ST-01"].value] + list(out.tables["csi"]["csi"])
    stat = [r["ST-01"].status.label] + list(out.tables["csi"]["status"])
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([names[i] for i in order], [vals[i] for i in order], color=[STATUS_COL[stat[i]] for i in order])
    t = cfg["thresholds"]["psi"]
    for x in (t["watch"], t["fail"]):
        ax.axvline(x, color=GREY, ls="--", lw=1)
        ax.text(x, len(names) - 0.4, f"{x:.2f}", ha="center", color=GREY, fontsize=8)
    for i, j in enumerate(order):
        ax.text(vals[j] + 0.003, i, f"{vals[j]:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Population / characteristic stability index (development vs monitoring)")
    ax.set_title("Stability: score PSI and per-characteristic CSI", pad=16)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def roc_ks(out: AuditOutput, path: Path):
    a = out.arrays
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    tbl = out.tables["discrimination"].set_index(["metric", "sample"])
    for tag, col, lab in (("dev", BLUE, "Development"), ("mon", ORANGE, "Monitoring")):
        y, s = a[f"y_{tag}"], -a[f"s_{tag}"]
        fpr, tpr, _ = roc_curve(y, s)
        g = tbl.loc[("Gini", tag), "value"]
        ax1.plot(fpr, tpr, color=col, label=f"{lab} (Gini {g:.3f})")
    ax1.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    ax1.legend(frameon=False, loc="lower right")
    _style(ax1)

    y, s = a["y_mon"], a["s_mon"]
    grid = np.linspace(s.min(), s.max(), 400)
    cb = np.array([(s[y == 1] <= x).mean() for x in grid])
    cg = np.array([(s[y == 0] <= x).mean() for x in grid])
    k = int(np.argmax(np.abs(cb - cg)))
    ax2.plot(grid, cb, color=ORANGE, label="Defaulters")
    ax2.plot(grid, cg, color=BLUE, label="Non-defaulters")
    ax2.vlines(grid[k], cg[k], cb[k], color=GREY, ls="--")
    ax2.text(grid[k] + 3, 0.04, f"KS = {tbl.loc[('KS', 'mon'), 'value']:.3f}", fontsize=9)
    ax2.set(xlabel="Score", ylabel="Cumulative share", title="KS, monitoring sample")
    ax2.legend(frameon=False, loc="upper left")
    _style(ax2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def calibration_plot(out: AuditOutput, path: Path):
    t = out.tables["calibration_bands"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = t["band"]
    ax1.bar(x - 0.2, t["mean_pd"], 0.4, color=BLUE, label="Predicted PD")
    ax1.bar(x + 0.2, t["observed_dr"], 0.4, color=ORANGE, label="Observed default rate")
    ax1.errorbar(
        x + 0.2,
        t["observed_dr"],
        yerr=[t["observed_dr"] - t["ci_low"], t["ci_high"] - t["observed_dr"]],
        fmt="none",
        ecolor="black",
        lw=1,
    )
    ax1.set(
        xlabel="Score band (1 = riskiest)", ylabel="Default rate", title="Predicted vs observed by score band"
    )
    ax1.set_xticks(list(x))
    ax1.legend(frameon=False)
    _style(ax1)
    m = max(t["mean_pd"].max(), t["ci_high"].max()) * 1.05
    ax2.plot([0, m], [0, m], color=GREY, ls=":", lw=1)
    ax2.errorbar(
        t["mean_pd"],
        t["observed_dr"],
        yerr=[t["observed_dr"] - t["ci_low"], t["ci_high"] - t["observed_dr"]],
        fmt="o",
        color=ORANGE,
        ecolor="black",
        lw=1,
    )
    ax2.set(
        xlabel="Mean predicted PD",
        ylabel="Observed default rate (95% Wilson CI)",
        title="Calibration curve (monitoring)",
    )
    ax2.grid(alpha=0.25)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def traffic_light_plot(out: AuditOutput, path: Path):
    from .summary import RAG, traffic_light

    tl, overall = traffic_light(out.results)
    col = {"Green": STATUS_COL["Pass"], "Amber": STATUS_COL["Watch"], "Red": STATUS_COL["Fail"]}
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(tl) + 1.1))
    for i, r in enumerate(tl.to_dict("records")):
        y = len(tl) - 1 - i
        ax.add_patch(plt.Rectangle((0, y - 0.38), 0.9, 0.76, color=col[r["rating"]]))
        ax.text(0.45, y, r["rating"], va="center", ha="center", fontsize=9, color="white", weight="bold")
        ax.text(1.1, y, r["area"], va="center", fontsize=10)
        ax.text(
            3.2,
            y,
            f"{r['fail']} Fail   {r['watch']} Watch   {r['pass']} Pass",
            va="center",
            fontsize=9,
            color=GREY,
        )
    ax.set(xlim=(0, 6), ylim=(-0.6, len(tl) - 0.4))
    ax.axis("off")
    ax.set_title(f"Traffic-light summary (overall: {RAG[overall]})", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_all(out: AuditOutput, fig_dir: Path, cfg: dict) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        fig_dir / "psi_bars.png",
        fig_dir / "roc_ks.png",
        fig_dir / "calibration.png",
        fig_dir / "traffic_light.png",
    ]
    psi_bars(out, paths[0], cfg)
    roc_ks(out, paths[1])
    calibration_plot(out, paths[2])
    traffic_light_plot(out, paths[3])
    return paths
