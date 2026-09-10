"""Rebuild Figures 7.8 and 7.9 from archived replication paths.

Early-stopped runs are extended with their last observed state for display.
This preserves the absorbing failure state and keeps all 20 replications in
the cross-seed mean at every plotted business day. The economic simulation is
not rerun.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCENARIOS = (
    ("ON/ON", None),
    ("ON/OFF", "rollover_on_support_off"),
    ("OFF/ON", "rollover_off_support_on"),
    ("OFF/OFF", "rollover_off_support_off"),
)
MECHANISMS = {
    "decentralized": "decentralized/matcher_load_den_gnn_v6_local",
    "centralized": "centralized/matcher_off",
}
METRICS = (
    ("sr", "Systemic risk"),
    ("fr", "Failure rate"),
    ("cbs", "Capital-breach share"),
    ("cgr", "Capital-gap ratio"),
)
EXPECTED_FINGERPRINT = "11dda58061ded609"
EXPECTED_RUNS = 20
HORIZON = 1000


def load_artifact(root: Path, relative: str | None) -> dict:
    path = root / relative / "compare_artifacts.json" if relative else root / "compare_artifacts.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("code_fingerprint") != EXPECTED_FINGERPRINT:
        raise ValueError(f"Unexpected source fingerprint: {path}")
    if len(data.get("baseline_runs") or []) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} baseline runs: {path}")
    return data


def absorbing_mean(artifact: dict, key: str) -> np.ndarray:
    source_key = "raw_sr" if key == "sr" else key
    rows = []
    for run in artifact["baseline_runs"]:
        values = np.asarray(run.get(source_key) or run.get(key) or [], dtype=float)
        if values.size == 0 or values.size > HORIZON or not np.isfinite(values).all():
            raise ValueError(f"Invalid {key} trajectory for seed {run.get('seed')}")
        if key == "fr" and np.any(np.diff(values) < -1e-12):
            raise ValueError(f"Non-absorbing failure path for seed {run.get('seed')}")
        rows.append(np.pad(values, (0, HORIZON - values.size), constant_values=values[-1]))
    mean = np.mean(np.vstack(rows), axis=0)
    if key == "fr" and np.any(np.diff(mean) < -1e-12):
        raise ValueError("Cross-seed failure-rate mean must be non-decreasing")
    return mean


def upper_limit(curves: list[np.ndarray]) -> float:
    maximum = max(float(np.max(curve)) for curve in curves)
    return min(1.0, max(0.1, np.ceil(maximum * 1.08 * 10.0) / 10.0))


def plot_mechanism(artifacts: list[tuple[str, dict]], mechanism: str, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.35, 7.73), sharex=True)
    x = np.arange(1, HORIZON + 1)
    for ax, (key, title) in zip(axes.flat, METRICS):
        curves = []
        for label, artifact in artifacts:
            curve = absorbing_mean(artifact, key)
            curves.append(curve)
            ax.plot(x, curve, linewidth=1.55, label=label)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(1, HORIZON)
        ax.set_ylim(0.0, upper_limit(curves))
        ax.grid(True, color="#DDE2E8", linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
        ax.set_xlabel("Business day")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.025, 0.075, 0.995, 0.995), h_pad=2.0, w_pad=2.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"compare_four_baselines_{mechanism}.png"
    fig.savefig(target, dpi=300, facecolor="white")
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })
    for mechanism, relative_root in MECHANISMS.items():
        root = args.artifacts_dir / relative_root
        artifacts = [(label, load_artifact(root, relative)) for label, relative in SCENARIOS]
        target = plot_mechanism(artifacts, mechanism, args.output_dir)
        print(f"Saved: {target}")


if __name__ == "__main__":
    main()
