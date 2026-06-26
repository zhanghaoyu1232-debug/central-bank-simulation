#!/usr/bin/env python3
"""
同时运行 decentralized 与 centralized 两个银行仿真模型，并生成结果对比图。

本脚本不 import 仿真模块（避免 torch 依赖冲突），改为子进程调用各模型脚本。
对比阶段仅依赖 numpy / matplotlib / PIL，读取各模型导出的 compare_artifacts.json。

各模型单独输出：
  输出/figures/decentralized/
  输出/figures/centralized/

对比图输出：
  输出/figures/comparison/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
FIG_ROOT = ROOT / "输出" / "figures"
COMPARE_DIR = FIG_ROOT / "comparison"

DECENTRALIZED_SCRIPT = ROOT / "bank_simulation_model_decentralized_central_policy.py"
CENTRALIZED_SCRIPT = ROOT / "bank_simulation_model_centralized_central_policy.py"

MODEL_SPECS = {
    "decentralized": ("Decentralized (RFQ)", DECENTRALIZED_SCRIPT),
    "centralized": ("Centralized", CENTRALIZED_SCRIPT),
}

FEATURE_SCENARIOS = (
    (True, True),
    (True, False),
    (False, True),
    (False, False),
)

REQUIRED_PACKAGES = (
    "torch",
    "torch_geometric",
    "pandas",
    "openpyxl",
    "matplotlib",
    "scipy",
    "networkx",
    "PIL",
    "numpy",
)

THETA = 0.08


def _check_dependencies(python_exe: str) -> bool:
    check_code = """
import importlib
missing = []
for name in {names!r}:
    mod = "PIL" if name == "PIL" else name
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(name)
if missing:
    print("MISSING:" + ",".join(missing))
else:
    import torch
    print("OK:" + torch.__version__)
""".format(names=list(REQUIRED_PACKAGES))
    try:
        proc = subprocess.run(
            [python_exe, "-c", check_code],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        print(f"无法执行 Python: {python_exe}\n  {exc}")
        return False

    out = (proc.stdout or "").strip()
    if proc.returncode == 0 and out.startswith("OK:"):
        print(f"[env] torch {out[3:]} ({python_exe})")
        return True

    missing = []
    if out.startswith("MISSING:"):
        missing = [x for x in out.split(":", 1)[1].split(",") if x]

    print("当前 Python 缺少仿真依赖，无法运行。")
    print(f"  Python: {python_exe}")
    print("  请安装：")
    print(f"    {python_exe} -m pip install -r \"{ROOT / 'requirements.txt'}\"")
    if missing:
        print(f"  缺少: {', '.join(missing)}")
    if proc.stderr.strip():
        print(f"  详情: {proc.stderr.strip()}")
    return False


def _load_artifacts(fig_dir: Path) -> dict:
    path = fig_dir / "compare_artifacts.json"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 {path}。请先运行对应模型脚本，或使用本脚本不带 --skip-run。"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_model_script(
    script: Path,
    fig_dir: Path,
    *,
    python_exe: str,
    T: int,
    nsim: int,
    no_train: bool,
    rollover_enabled: bool = True,
    policy_support_enabled: bool = True,
) -> None:
    cmd = [
        python_exe,
        str(script),
        "--fig-dir",
        str(fig_dir),
        "--T",
        str(T),
        "--nsim",
        str(nsim),
    ]
    if no_train:
        cmd.append("--no-train")
    if not rollover_enabled:
        cmd.append("--no-rollover")
    if not policy_support_enabled:
        cmd.append("--no-policy-support")
    print(f"\n>>> {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"模型脚本失败 (exit={proc.returncode}): {script.name}"
        )
    print(f"[done] {script.name} in {time.perf_counter() - t0:.1f}s")


def _feature_suffix(rollover_enabled: bool, policy_support_enabled: bool) -> str:
    r = "rollover_on" if rollover_enabled else "rollover_off"
    s = "support_on" if policy_support_enabled else "support_off"
    return f"{r}_{s}"


def _scenario_dir(model_key: str, rollover_enabled: bool, policy_support_enabled: bool) -> Path:
    return FIG_ROOT / "scenarios" / f"{model_key}_{_feature_suffix(rollover_enabled, policy_support_enabled)}"


def _scenario_label(model_key: str, rollover_enabled: bool, policy_support_enabled: bool) -> str:
    model_label = MODEL_SPECS[model_key][0]
    r = "R:on" if rollover_enabled else "R:off"
    s = "Support:on" if policy_support_enabled else "Support:off"
    return f"{model_label} ({r}, {s})"


def plot_baseline_comparison(
    dec_art: dict,
    cen_art: dict,
    out_path: Path,
) -> Path:
    dec_series = dec_art["baseline"]
    cen_series = cen_art["baseline"]
    weights = dec_art.get("weights", (0.5, 0.3, 0.2))
    theta = dec_art.get("theta", THETA)

    labels = [
        ("SR (Systemic Risk)", "sr"),
        ("FR (Failure Rate)", "fr"),
        ("CBS (Low-CAR Share)", "cbs"),
        ("CGR (Capital Gap Ratio)", "cgr"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.ravel()

    for ax, (title, key) in zip(axes, labels):
        y_dec = np.asarray(dec_series[key], dtype=float)
        y_cen = np.asarray(cen_series[key], dtype=float)
        n = min(len(y_dec), len(y_cen))
        xs = np.arange(1, n + 1)
        ax.plot(xs, y_dec[:n], lw=2.0, marker="o", ms=3, label="Decentralized (RFQ)")
        ax.plot(xs, y_cen[:n], lw=2.0, marker="s", ms=3, label="Centralized")
        ax.set_title(title)
        ax.set_ylabel("Value (0–1)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Baseline Trajectory Comparison — W={tuple(weights)}, θ={theta}",
        fontsize=13,
    )
    axes[-1].set_xlabel("Time Step")
    axes[-2].set_xlabel("Time Step")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_theta_sweep_comparison(
    dec_art: dict,
    cen_art: dict,
    out_path: Path,
) -> Path:
    dec_theta = np.asarray(dec_art["theta_sweep"]["theta_grid"], dtype=float)
    cen_theta = np.asarray(cen_art["theta_sweep"]["theta_grid"], dtype=float)
    dec_curves = [np.asarray(y, dtype=float) for y in dec_art["theta_sweep"]["sr_curves"]]
    cen_curves = [np.asarray(y, dtype=float) for y in cen_art["theta_sweep"]["sr_curves"]]
    theta = float(dec_art.get("theta", THETA))

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap_dec = plt.get_cmap("Blues")
    cmap_cen = plt.get_cmap("Oranges")
    norm_dec = plt.Normalize(vmin=float(dec_theta.min()), vmax=float(dec_theta.max()))
    norm_cen = plt.Normalize(vmin=float(cen_theta.min()), vmax=float(cen_theta.max()))

    for th, y in zip(dec_theta, dec_curves):
        if len(y) == 0:
            continue
        xs = np.arange(1, len(y) + 1)
        ax.plot(xs, y, lw=1.6, alpha=0.85, color=cmap_dec(norm_dec(th)), linestyle="-")

    for th, y in zip(cen_theta, cen_curves):
        if len(y) == 0:
            continue
        xs = np.arange(1, len(y) + 1)
        ax.plot(xs, y, lw=1.6, alpha=0.85, color=cmap_cen(norm_cen(th)), linestyle="--")

    dec_base_idx = int(np.argmin(np.abs(dec_theta - theta)))
    cen_base_idx = int(np.argmin(np.abs(cen_theta - theta)))
    if len(dec_curves[dec_base_idx]) > 0:
        xs = np.arange(1, len(dec_curves[dec_base_idx]) + 1)
        ax.plot(xs, dec_curves[dec_base_idx], lw=3.0, color="navy", label=f"Decentralized θ={theta:.2f}")
    if len(cen_curves[cen_base_idx]) > 0:
        xs = np.arange(1, len(cen_curves[cen_base_idx]) + 1)
        ax.plot(
            xs, cen_curves[cen_base_idx], lw=3.0, color="darkorange",
            linestyle="--", label=f"Centralized θ={theta:.2f}",
        )

    ax.set_title("θ Measure Sweep Comparison (SR)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Systemic Risk (SR)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def _load_panel_image(path: Path | None) -> Image.Image | None:
    if path is None or not Path(path).exists():
        return None
    with Image.open(path) as im:
        return im.convert("RGB")


def plot_network_panel_comparison(
    dec_panel: Path | None,
    cen_panel: Path | None,
    out_path: Path,
) -> Path | None:
    dec_img = _load_panel_image(dec_panel)
    cen_img = _load_panel_image(cen_panel)
    if dec_img is None and cen_img is None:
        print("[warn] 无 network panel 可对比，跳过。")
        return None

    panels: list[tuple[str, Image.Image]] = []
    if dec_img is not None:
        panels.append(("Decentralized (RFQ)", dec_img))
    if cen_img is not None:
        panels.append(("Centralized", cen_img))

    fig, axes = plt.subplots(len(panels), 1, figsize=(18, 5 * len(panels)))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, im) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.suptitle("Network Panel Comparison", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_all_scenario_baselines(
    scenario_artifacts: dict[str, tuple[str, dict]],
    out_path: Path,
) -> Path:
    labels = [
        ("SR (Systemic Risk)", "sr"),
        ("FR (Failure Rate)", "fr"),
        ("CBS (Low-CAR Share)", "cbs"),
        ("CGR (Capital Gap Ratio)", "cgr"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()
    for ax, (title, key) in zip(axes, labels):
        for _, (label, art) in scenario_artifacts.items():
            y = np.asarray(art["baseline"][key], dtype=float)
            xs = np.arange(1, len(y) + 1)
            ax.plot(xs, y, lw=1.7, alpha=0.88, label=label)
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.35)
    axes[-1].set_xlabel("Time Step")
    axes[-2].set_xlabel("Time Step")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle("All Scenario Baseline Trajectories", fontsize=14)
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_policy_support_comparison(
    scenario_artifacts: dict[str, tuple[str, dict]],
    out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(14, 7))
    any_series = False
    for _, (label, art) in scenario_artifacts.items():
        support = art.get("policy_support", {})
        y = np.asarray(support.get("total", []), dtype=float)
        if len(y) == 0:
            continue
        any_series = True
        xs = np.arange(1, len(y) + 1)
        ax.plot(xs, y, lw=1.8, alpha=0.9, label=label)
    if not any_series:
        ax.text(0.5, 0.5, "No policy support data", transform=ax.transAxes,
                ha="center", va="center")
    ax.set_title("Central Bank Policy Support Total")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Liquidity + Capital Support")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="对比 decentralized 与 centralized 仿真结果图")
    parser.add_argument("--python", dest="python_exe", default=sys.executable,
                        help="用于运行仿真脚本的 Python（需已安装 torch）")
    parser.add_argument("--T", type=int, default=800, help="仿真步数上限")
    parser.add_argument("--nsim", type=int, default=20, help="plot batch 轨迹条数")
    parser.add_argument("--skip-run", action="store_true",
                        help="跳过仿真，仅从已有 compare_artifacts.json 生成对比图")
    parser.add_argument("--no-train", action="store_true",
                        help="不重新训练 GNN/matcher")
    parser.add_argument("--single-scenario", action="store_true",
                        help="仅运行旧版默认场景：rollover on + policy support on")
    args = parser.parse_args()

    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = ((True, True),) if args.single_scenario else FEATURE_SCENARIOS

    if not args.skip_run:
        if not _check_dependencies(args.python_exe):
            raise SystemExit(1)
        for rollover_enabled, policy_support_enabled in scenarios:
            for model_key, (_, script) in MODEL_SPECS.items():
                _run_model_script(
                    script,
                    _scenario_dir(model_key, rollover_enabled, policy_support_enabled),
                    python_exe=args.python_exe,
                    T=args.T,
                    nsim=args.nsim,
                    no_train=args.no_train,
                    rollover_enabled=rollover_enabled,
                    policy_support_enabled=policy_support_enabled,
                )

    scenario_artifacts: dict[str, tuple[str, dict]] = {}
    for rollover_enabled, policy_support_enabled in scenarios:
        suffix = _feature_suffix(rollover_enabled, policy_support_enabled)
        dec_fig = _scenario_dir("decentralized", rollover_enabled, policy_support_enabled)
        cen_fig = _scenario_dir("centralized", rollover_enabled, policy_support_enabled)
        dec_art = _load_artifacts(dec_fig)
        cen_art = _load_artifacts(cen_fig)
        scenario_artifacts[f"decentralized_{suffix}"] = (
            _scenario_label("decentralized", rollover_enabled, policy_support_enabled),
            dec_art,
        )
        scenario_artifacts[f"centralized_{suffix}"] = (
            _scenario_label("centralized", rollover_enabled, policy_support_enabled),
            cen_art,
        )

        plot_baseline_comparison(
            dec_art, cen_art,
            COMPARE_DIR / f"compare_baseline_{suffix}.png",
        )
        plot_theta_sweep_comparison(
            dec_art, cen_art,
            COMPARE_DIR / f"compare_theta_sweep_sr_{suffix}.png",
        )
        plot_network_panel_comparison(
            dec_fig / "network_decentralized_panel.png",
            cen_fig / "network_centralized_panel.png",
            COMPARE_DIR / f"compare_network_panel_{suffix}.png",
        )

    plot_all_scenario_baselines(
        scenario_artifacts,
        COMPARE_DIR / "compare_all_scenario_baselines.png",
    )
    plot_policy_support_comparison(
        scenario_artifacts,
        COMPARE_DIR / "compare_policy_support_total.png",
    )

    print(f"\n[done] 对比图目录: {COMPARE_DIR}")


if __name__ == "__main__":
    main()
