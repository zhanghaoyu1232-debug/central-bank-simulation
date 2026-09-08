"""Rebuild Figures 7.1, 7.10 and 7.11 from recorded numerical results.

Run without arguments to use the compact, archived plotting inputs.
Use --artifacts-dir PATH to reconstruct those inputs from figures.zip.
This script does not run the economic simulation or alter recorded paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reproduction" / "plot_inputs.json"
FIGURES = ROOT / "figures"
SOURCES = {
    "DEN": "decentralized/matcher_load_den_gnn_v6_local/compare_artifacts.json",
    "CEN": "centralized/matcher_off/compare_artifacts.json",
}


def collect_inputs(artifacts_dir: Path) -> dict:
    output = {"code_fingerprint": "11dda58061ded609", "mechanisms": {}}
    for mechanism, relative in SOURCES.items():
        path = artifacts_dir / relative
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["code_fingerprint"] != output["code_fingerprint"]:
            raise ValueError(f"Unexpected source fingerprint: {relative}")
        runs = data["baseline_runs"]
        if len(runs) != 20:
            raise ValueError("The formal baseline requires 20 replications")
        seeds = [run["seed"] for run in runs]
        metrics = {}
        for key in ("sr", "fr", "cbs", "cgr"):
            values = np.asarray([run[key] for run in runs], dtype=float)
            if values.shape != (20, 1000) or not np.isfinite(values).all():
                raise ValueError(f"Invalid baseline {mechanism}/{key}")
            mean = values.mean(axis=0)
            np.testing.assert_allclose(mean, data["baseline"][key], atol=1e-12)
            metrics[key] = {
                "mean": mean.tolist(),
                "ci_half_width": (1.96 * values.std(axis=0, ddof=1) / np.sqrt(20)).tolist(),
                "run0": values[0].tolist(),
            }
        theta = data["theta_sweep"]
        weights = data["weight_sweep"]
        np.testing.assert_allclose(theta["sr_curves"][0], metrics["sr"]["mean"], atol=1e-12)
        for w, curve in zip(weights["weights"], weights["sr_curves"]):
            expected = sum(wi * np.asarray(metrics[key]["mean"])
                           for wi, key in zip(w, ("fr", "cbs", "cgr")))
            np.testing.assert_allclose(curve, expected, atol=1e-12)
        shocks = np.asarray(data["extra_metrics_run0"]["common_liquidity_shock"])
        output["mechanisms"][mechanism] = {
            "source": relative,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seeds": seeds,
            "metrics": metrics,
            "theta_sweep": theta,
            "weight_sweep": weights,
            "run0_shock_days": (np.flatnonzero(shocks > 0) + 1).tolist(),
        }
    den, cen = (output["mechanisms"][key] for key in ("DEN", "CEN"))
    if den["seeds"] != cen["seeds"]:
        raise ValueError("Formal replications are not in the same paired order")
    return output


def upper_limit(value: float) -> float:
    padded = value * 1.08
    increment = 0.01 if padded <= 0.1 else 0.025 if padded <= 0.25 else 0.05 if padded <= 0.5 else 0.1
    return float(max(increment, np.ceil(padded / increment) * increment))


def decorate(ax):
    ax.set_xlim(1, 1000)
    ax.set_xticks([1, 200, 400, 600, 800, 1000])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    ax.grid(True, color="#DDE2E8", linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Business day")


def save_figure(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf", facecolor="white", metadata={"Creator": "Matplotlib; archived recorded results"})
    plt.close(fig)


def baseline(data):
    colors = {"DEN": "#2166AC", "CEN": "#C7473B"}
    labels = {"DEN": "RFQ (DEN)", "CEN": "Centralized (CEN)"}
    titles = ["(a) Systemic risk (SR)", "(b) Failure rate (FR)",
              "(c) Capital-breach share (CBS)", "(d) Capital-gap ratio (CGR)"]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.085, top=0.84, hspace=0.42, wspace=0.30)
    x = np.arange(1, 1001)
    for ax, key, title in zip(axes.flat, ("sr", "fr", "cbs", "cgr"), titles):
        maximum = 0.0
        minimum = 0.0
        for mechanism in ("DEN", "CEN"):
            cell = data["mechanisms"][mechanism]["metrics"][key]
            mean, half, run0 = (np.asarray(cell[k]) for k in ("mean", "ci_half_width", "run0"))
            ax.fill_between(x, mean - half, mean + half, color=colors[mechanism], alpha=0.12, linewidth=0)
            ax.plot(x, run0, color=colors[mechanism], linestyle=":", linewidth=1.0, alpha=0.72)
            ax.plot(x, mean, color=colors[mechanism], linewidth=1.65)
            maximum = max(maximum, float((mean + half).max()), float(run0.max()))
            minimum = min(minimum, float((mean - half).min()))
        for day in data["mechanisms"]["DEN"]["run0_shock_days"]:
            ax.axvline(day, color="#56616F", linewidth=0.5, alpha=0.15, zorder=0)
        ax.set_ylim(minimum * 1.08, upper_limit(maximum))
        ax.set_title(title, fontsize=10.8, loc="left", pad=9)
        decorate(ax)
    handles = [Line2D([], [], color=colors[k], linewidth=1.8, label=labels[k]) for k in ("DEN", "CEN")]
    handles += [Line2D([], [], color="#4D5663", linewidth=1.6, label="20-run mean"),
                Line2D([], [], color="#4D5663", linestyle=":", linewidth=1.2, label="Paired run 0"),
                Patch(facecolor="#829AB6", alpha=0.2, label="Pointwise 95% CI"),
                Line2D([], [], color="#56616F", alpha=0.4, linewidth=0.7, label="Run-0 common shock")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.993), ncol=3,
               frameon=False, fontsize=9.0, columnspacing=1.7, handlelength=2.4)
    save_figure(fig, "compare_baseline_rollover_on_support_on")


def sweep(data, kind):
    key, grid_key, symbol, selected, filename = (
        ("theta_sweep", "theta_grid", r"$\theta$", 0.08, "compare_theta_sweep_sr_rollover_on_support_on")
        if kind == "theta" else
        ("weight_sweep", "w1_grid", r"$w_1$", 0.5, "compare_weight_sweep_sr_rollover_on_support_on")
    )
    grid = np.asarray(data["mechanisms"]["DEN"][key][grid_key])
    curves = {m: np.asarray(data["mechanisms"][m][key]["sr_curves"]) for m in ("DEN", "CEN")}
    maximum = max(float(v.max()) for v in curves.values())
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.825, bottom=0.09, top=0.94, hspace=0.31)
    cmap = matplotlib.colormaps["viridis"].resampled(len(grid))
    norm = matplotlib.colors.Normalize(vmin=grid.min(), vmax=grid.max())
    for ax, mechanism, title in zip(axes, ("DEN", "CEN"),
                                    ("(a) Decentralized RFQ (DEN)", "(b) Centralized matching (CEN)")):
        for value, curve in zip(grid, curves[mechanism]):
            base = bool(np.isclose(value, selected))
            ax.plot(np.arange(1, len(curve) + 1), curve,
                    color="#222222" if base else cmap(norm(value)),
                    linewidth=1.9 if base else 1.15, alpha=1 if base else 0.92,
                    zorder=5 if base else 2,
                    label=f"Baseline {symbol} = {selected:g}" if base else None)
        ax.set_ylim(0, upper_limit(maximum))
        ax.set_title(title, fontsize=11.3, loc="left", pad=8)
        ax.set_ylabel("Systemic risk (SR)")
        decorate(ax)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    axes[0].set_xlabel("")
    cax = fig.add_axes([0.86, 0.17, 0.025, 0.66])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=grid)
    cb.set_label(f"Measurement threshold {symbol}" if kind == "theta" else f"Failure weight {symbol}", labelpad=9)
    cb.outline.set_linewidth(0.5)
    save_figure(fig, filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path)
    args = parser.parse_args()
    if args.artifacts_dir:
        data = collect_inputs(args.artifacts_dir)
        INPUT.parent.mkdir(exist_ok=True)
        INPUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    else:
        data = json.loads(INPUT.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    baseline(data)
    sweep(data, "theta")
    sweep(data, "weight")
    print("Rebuilt three figures from archived numerical paths; no simulation rerun.")


if __name__ == "__main__":
    main()
