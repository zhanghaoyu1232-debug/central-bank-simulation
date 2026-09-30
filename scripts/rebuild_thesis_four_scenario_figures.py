"""Rebuild thesis Figures 7.8--7.9 from validated comparison artifacts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCENARIOS = (
    ("ON/ON", True, True, "#0072B2"),
    ("ON/OFF", True, False, "#E69F00"),
    ("OFF/ON", False, True, "#009E73"),
    ("OFF/OFF", False, False, "#D55E00"),
)
PANELS = (
    ("Systemic risk", "sr"),
    ("Failure rate", "fr"),
    ("Capital-breach share", "cbs"),
    ("Capital-gap ratio", "cgr"),
)


def artifact_path(root: Path, mechanism: str, rollover: bool, support: bool) -> Path:
    if mechanism == "decentralized":
        base = root / "decentralized" / "matcher_load_den_gnn_v6_local"
    else:
        base = root / "centralized" / "matcher_off"
    if rollover and support:
        return base / "compare_artifacts.json"
    suffix = f"rollover_{'on' if rollover else 'off'}_support_{'on' if support else 'off'}"
    return base / suffix / "compare_artifacts.json"


def load_series(root: Path, mechanism: str) -> list[tuple[str, dict, str]]:
    series = []
    for label, rollover, support, color in SCENARIOS:
        path = artifact_path(root, mechanism, rollover, support)
        with path.open(encoding="utf-8") as stream:
            artifact = json.load(stream)
        series.append((label, artifact["baseline"], color))
    return series


def upper_limit(curves: list[np.ndarray]) -> float:
    maximum = max((float(np.nanmax(y)) for y in curves if y.size), default=0.0)
    if maximum <= 0.285:
        return 0.30
    return min(1.0, max(0.40, math.ceil(maximum * 11.0) / 10.0))


def draw(root: Path, mechanism: str, output: Path) -> None:
    data = load_series(root, mechanism)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.7), sharex=True)
    for ax, (title, key) in zip(axes.ravel(), PANELS):
        curves = []
        for label, baseline, color in data:
            y = np.asarray(baseline[key], dtype=float)
            curves.append(y)
            ax.plot(np.arange(1, len(y) + 1), y, lw=1.6, color=color, label=label)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0.0, upper_limit(curves))
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("Business day", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.12, hspace=0.25, wspace=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    draw(
        args.artifact_root,
        "decentralized",
        args.output_dir / "compare_four_baselines_decentralized.png",
    )
    draw(
        args.artifact_root,
        "centralized",
        args.output_dir / "compare_four_baselines_centralized.png",
    )


if __name__ == "__main__":
    main()
