#!/usr/bin/env python3
""" 
同时运行 decentralized 与 centralized 两个银行仿真模型，并生成结果对比图。

本脚本不 import 仿真模块（避免 torch 依赖冲突），改为子进程调用各模型脚本。
对比阶段仅依赖 numpy / matplotlib / PIL，读取各模型导出的 compare_artifacts.json。

各模型图目录（由 output_paths.py 固定）：
  .../git/输出/figures/decentralized/
  .../git/输出/figures/centralized/
  默认只跑 rollover+support 全开，图直接写入上述目录。

对比图：.../git/输出/figures/comparison/
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

from output_paths import (
    BASE_ON_OFF_COMPARE_DIR,
    CENTRALIZED_FIG_DIR,
    COMPARE_FIG_DIR,
    DECENTRALIZED_FIG_DIR,
    OUTPUT_ROOT,
    ARTIFACT_SCHEMA_VERSION,
    model_figure_dir,
    scenario_figure_dir,
    simulation_code_fingerprint,
)
from interbank_matcher_shared import (
    GNN_PAIR_MATCHER_V6_LOCAL_PATH,
    checkpoint_sha256,
    make_seed_stream,
)

ROOT = Path(__file__).resolve().parent
OUT_ROOT = OUTPUT_ROOT
FIG_ROOT = OUT_ROOT
COMPARE_DIR = COMPARE_FIG_DIR
DEFAULT_MATCH_B = 1200.0
# 最多运行 4000 步；商业银行只剩 STOP_ALIVE_THRESHOLD 家时早停，以其 collapse_step 为存活时间。
# 不要把早停后的 FR 补到 4000 再计算存活时间。
DEFAULT_SIM_T_CAP = 4000
MAX_PLOT_STEPS = 4000
STOP_ALIVE_THRESHOLD = 1

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

# Loan-size cap robustness: baseline B=1200 with low/high checks.
B_VALUES = (600.0, 1200.0, 1800.0)

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


def _read_json_with_retry(
    path: Path,
    *,
    attempts: int = 10,
    delay: float = 0.25,
) -> dict:
    """Read JSON; retry if OneDrive / AV still holds the file after a child write."""
    path = Path(path)
    last_err: Exception | None = None
    n = max(1, int(attempts))
    for i in range(n):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise
        except (PermissionError, OSError, json.JSONDecodeError) as exc:
            last_err = exc
            if i + 1 >= n:
                break
            wait = float(delay) * (2 ** i)
            print(
                f"[warn] retry {i + 1}/{n - 1} reading {path.name}: {exc} "
                f"(wait {wait:.1f}s)"
            )
            time.sleep(wait)
    raise last_err if last_err is not None else RuntimeError(f"failed to read {path}")


def _load_artifacts(fig_dir: Path) -> dict:
    path = fig_dir / "compare_artifacts.json"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 {path}。请先运行对应模型脚本，或使用本脚本不带 --skip-run。"
        )
    return _read_json_with_retry(path)


def _artifacts_ready(
    fig_dir: Path,
    *,
    model_key: str,
    rollover_enabled: bool,
    policy_support_enabled: bool,
    nsim: int,
    T: int | None = None,
    B: float | None = None,
    stop_alive_threshold: int | None = None,
    matcher_mode: str = "off",
    matcher_checkpoint_sha256: str | None = None,
) -> bool:
    """True if compare_artifacts.json looks finished for this scenario (resume skip)."""
    art_path = fig_dir / "compare_artifacts.json"
    if not art_path.exists():
        return False
    try:
        data = _read_json_with_retry(art_path, attempts=4, delay=0.2)
    except (OSError, json.JSONDecodeError):
        return False

    feats = data.get("features") or {}
    if bool(feats.get("rollover_enabled", True)) != bool(rollover_enabled):
        return False
    if bool(feats.get("policy_support_enabled", True)) != bool(policy_support_enabled):
        return False

    matching = data.get("matching") or {}
    want_mode = str(matcher_mode or "off").strip().lower()
    got_mode = str(matching.get("matcher_mode") or "off").strip().lower()
    if got_mode != want_mode:
        return False
    if want_mode in ("load", "train"):
        got_hash = matching.get("matcher_checkpoint_sha256")
        if matcher_checkpoint_sha256:
            if not got_hash or str(got_hash) != str(matcher_checkpoint_sha256):
                return False
        elif not got_hash:
            # Old artifacts without hash cannot be safely resumed for GNN runs.
            return False

    # Reject pre-native-length artifacts and any run from a different code snapshot.
    schema = int(data.get("artifact_schema_version") or 0)
    if schema < int(ARTIFACT_SCHEMA_VERSION):
        return False
    fp = data.get("code_fingerprint")
    if not fp or str(fp) != str(simulation_code_fingerprint()):
        return False

    baseline_runs = data.get("baseline_runs") or []
    network_runs = data.get("network_summary_runs") or []
    if len(baseline_runs) < int(nsim) or len(network_runs) < int(nsim):
        return False

    # Padded-to-cap series are invalid for early-stop metrics.
    for run in baseline_runs:
        series_len = len(run.get("sr") or [])
        observed = run.get("observed_until")
        if observed is not None and series_len and int(series_len) != int(observed):
            return False
        if T is not None and series_len and int(series_len) > int(T):
            return False

    if stop_alive_threshold is not None:
        survival = data.get("survival") or {}
        recorded_threshold = survival.get("stop_alive_threshold")
        if recorded_threshold is None and baseline_runs:
            recorded_threshold = baseline_runs[0].get("stop_alive_threshold")
        if recorded_threshold is None or int(recorded_threshold) != int(stop_alive_threshold):
            return False

    meta = data.get("batch_meta") or {}
    # Prefer T_cap: after early-stop, meta["T"] is realized horizon, not the ceiling.
    if T is not None:
        recorded_cap = meta.get("T_cap", meta.get("T"))
        if recorded_cap is not None and int(recorded_cap) != int(T):
            return False
    if B is not None and "B" in meta and abs(float(meta["B"]) - float(B)) > 1e-9:
        return False

    # Mid-run death: network panel rewritten after artifacts → treat as incomplete.
    panel = fig_dir / f"network_{model_key}_panel.png"
    if panel.exists():
        try:
            if panel.stat().st_mtime > art_path.stat().st_mtime + 30.0:
                return False
        except OSError:
            return False
    return True


def _run_model_script(
    script: Path,
    fig_dir: Path,
    *,
    python_exe: str,
    T: int,
    nsim: int,
    B: float,
    matcher_mode: str = "off",
    no_train: bool | None = None,
    rollover_enabled: bool = True,
    policy_support_enabled: bool = True,
    stop_alive_threshold: int = STOP_ALIVE_THRESHOLD,
    allow_den_rate_only: bool = False,
    seed0: int = 42,
) -> None:
    mode = str(matcher_mode).strip().lower()
    if no_train is True and "centralized" in script.name and "decentralized" not in script.name:
        mode = "off"
    cmd = [
        python_exe,
        str(script),
        "--fig-dir",
        str(fig_dir),
        "--T",
        str(T),
        "--nsim",
        str(nsim),
        "--B",
        str(B),
        "--stop-alive-threshold",
        str(int(stop_alive_threshold)),
        "--matcher-mode",
        mode,
        "--seed0",
        str(int(seed0)),
    ]
    if not rollover_enabled:
        cmd.append("--no-rollover")
    if not policy_support_enabled:
        cmd.append("--no-policy-support")
    if allow_den_rate_only and "decentralized" in script.name:
        cmd.append("--allow-den-rate-only")
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


def _scenario_dir(
    model_key: str,
    rollover_enabled: bool,
    policy_support_enabled: bool,
    matcher_mode: str = "off",
    run_scope: str = "formal",
) -> Path:
    path = scenario_figure_dir(
        model_key,
        rollover_enabled,
        policy_support_enabled,
        matcher_mode=matcher_mode,
    )
    if str(run_scope).lower() == "calibration":
        return OUTPUT_ROOT / "calibration" / path.relative_to(OUTPUT_ROOT)
    return path


def _loan_cap_fig_dir(model_key: str, B: float, matcher_mode: str = "off") -> Path:
    """Dedicated figure dir for loan-cap robustness runs (does not overwrite baseline)."""
    from output_paths import matcher_mode_dir_token

    return (
        OUTPUT_ROOT
        / "loan_cap"
        / f"B{int(round(B))}"
        / matcher_mode_dir_token(matcher_mode, model_key=model_key)
        / model_key
    )


def _list_primary_model_figures(fig_dir: Path) -> list[Path]:
    """rollover+support 标准流程应产出的单模型图（按存在性过滤）。"""
    names = (
        "baseline_trajectory.png",
        "weight_sweep_lines.png",
        "extra_market_metrics.png",
    )
    found: list[Path] = []
    for name in names:
        p = fig_dir / name
        if p.exists():
            found.append(p)
    found.extend(sorted(fig_dir.glob("theta_policy_scenario_lines_*.png")))
    found.extend(sorted(fig_dir.glob("network_*_panel.png")))
    return found


def _scenario_label(model_key: str, rollover_enabled: bool, policy_support_enabled: bool) -> str:
    model_label = MODEL_SPECS[model_key][0]
    r = "R:on" if rollover_enabled else "R:off"
    s = "Support:on" if policy_support_enabled else "Support:off"
    return f"{model_label} ({r}, {s})"


def _edge_set(edges) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for e in edges or []:
        if len(e) < 2:
            continue
        i, j = int(e[0]), int(e[1])
        out.add((i, j) if i <= j else (j, i))
    return out


def _jaccard(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return float(len(a & b) / union) if union else 0.0


def _mark_auction_days(ax, auction_flags, color="#888888", alpha=0.18) -> None:
    flags = np.asarray(auction_flags, dtype=float)
    if flags.size == 0:
        return
    # cycle_length==1 marks every day; skip to avoid painting the whole axis grey.
    if float(np.mean(flags >= 0.5)) >= 0.99:
        return
    for t, flag in enumerate(flags, start=1):
        if flag >= 0.5:
            ax.axvline(t, color=color, alpha=alpha, lw=1.0, zorder=0)


def _mark_shock_days(ax, shock_flags, color="#c0392b", alpha=0.22) -> None:
    flags = np.asarray(shock_flags, dtype=float)
    if flags.size == 0:
        return
    for t, flag in enumerate(flags, start=1):
        if flag >= 0.5:
            ax.axvline(t, color=color, alpha=alpha, lw=0.8, zorder=0)


def _artifact_shock_flags(art: dict) -> list:
    run0 = art.get("extra_metrics_run0") or {}
    flags = run0.get("common_liquidity_shock") or []
    if flags:
        return list(flags)
    runs = art.get("shock_path_runs") or []
    if runs:
        return list(runs[0].get("common_liquidity_shock") or [])
    return list((art.get("extra_metrics") or {}).get("common_liquidity_shock") or [])


def _extra_plot_bundle(art: dict) -> tuple[dict, bool]:
    """Prefer paired-seed run0 so shock days are not averaged away."""
    run0 = art.get("extra_metrics_run0") or {}
    mean = art.get("extra_metrics") or {}
    use_run0 = bool(
        run0.get("unmet_demand_rate")
        or run0.get("common_liquidity_shock")
        or run0.get("mean_lcr")
        or run0.get("funding_gap")
    )
    return (run0 if use_run0 else mean), use_run0


def _stack_runs(runs: list[dict], key: str) -> tuple[np.ndarray, list[int]]:
    """
    Stack replication-level trajectories into a matrix.

    Rows correspond to Monte Carlo replications and columns to time steps.
    For key ``sr`` / ``raw_sr``, prefer paper SR (``raw_sr`` then ``sr``).
    For ``collapse_index``, use the cumulative max series only.

    Early-stopped rows keep their last observed value (absorbing fill).
    NaN-padding plus nanmean would drop early-stopped paths and change the
    contributor set after collapse — a display artifact. Last-state padding is
    only for trajectory display; CBS may still move before stopping because it
    is defined over active banks.
    Formal numbers still use collapse_step, RMST, bank-days, and paired
    common-window metrics, not last-state-filled terminal SR.
    """
    valid = []

    for run in runs:
        if key in ("sr", "raw_sr"):
            series = run.get("raw_sr")
            if series is None:
                series = run.get("sr")
        elif key == "collapse_index":
            series = run.get("collapse_index")
        else:
            series = run.get(key)
        arr = np.asarray(series or [], dtype=float)
        if len(arr) > 0:
            valid.append(arr)

    if not valid:
        return np.empty((0, 0), dtype=float), []

    max_len = max(len(x) for x in valid)

    mat = np.full(
        (len(valid), max_len),
        np.nan,
        dtype=float,
    )

    for i, arr in enumerate(valid):
        mat[i, :len(arr)] = arr
        if len(arr) < max_len:
            mat[i, len(arr):] = arr[-1]

    n = np.sum(~np.isnan(mat), axis=0)

    return mat, n.tolist()


def _mean_ci95_from_runs(
    runs: list[dict],
    key: str,
) -> dict[str, np.ndarray]:
    mat, _ = _stack_runs(runs, key)

    if mat.size == 0:
        empty = np.asarray([], dtype=float)
        return {
            "mean": empty,
            "std": empty,
            "lower": empty,
            "upper": empty,
        }

    mean = np.nanmean(mat, axis=0)
    n = np.sum(~np.isnan(mat), axis=0)

    std = np.nanstd(
        mat,
        axis=0,
        ddof=1,
    )

    se = np.divide(
        std,
        np.sqrt(n),
        out=np.zeros_like(std),
        where=n > 1,
    )

    half = 1.96 * se
    lower = np.clip(mean - half, 0.0, 1.0)
    upper = np.clip(mean + half, 0.0, 1.0)

    return {
        "mean": mean,
        "std": std,
        "lower": lower,
        "upper": upper,
    }


def _runs_by_seed(artifact: dict) -> dict[int, dict]:
    result = {}

    for run in artifact.get("baseline_runs", []):
        if "seed" not in run:
            continue

        result[int(run["seed"])] = run

    return result


def _diff_summary(values) -> dict:
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
        }
    mean = float(np.mean(x))
    median = float(np.median(x))
    if n > 1:
        std = float(np.std(x, ddof=1))
        se = std / np.sqrt(n)
        half = 1.96 * se
    else:
        std = 0.0
        half = 0.0
    return {
        "n": int(n),
        "mean": mean,
        "median": median,
        "std": std,
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
    }


def _run_metric_series(run: dict, key: str) -> list:
    """``sr``/``raw_sr`` → paper SR; ``collapse_index`` → cumulative max (display only)."""
    if key in ("sr", "raw_sr"):
        series = run.get("raw_sr")
        if series is None:
            series = run.get("sr")
        return series or []
    if key == "collapse_index":
        series = run.get("collapse_index")
        if series is None:
            series = run.get("sr")
        return series or []
    return run.get(key) or []


def calculate_paired_path_statistics(
    art_a: dict,
    art_b: dict,
    *,
    keys: tuple[str, ...] = ("sr", "cgr"),
    label_a: str = "a",
    label_b: str = "b",
) -> dict:
    """Paired by seed on common window H_i = min(len_a, len_b). Never uses padded tails.

    For key ``sr``, uses ``raw_sr`` when present (policy / hypothesis tests).
    """
    runs_a = _runs_by_seed(art_a)
    runs_b = _runs_by_seed(art_b)
    common_seeds = sorted(set(runs_a) & set(runs_b))
    if not common_seeds:
        raise ValueError("No paired Monte Carlo runs with common seeds were found.")

    trapz = getattr(np, "trapezoid", None) or np.trapz
    out_metrics: dict[str, dict] = {}
    delta_paths: dict[str, list] = {k: [] for k in keys}
    common_H: list[int] = []
    collapse_a: list[float] = []
    collapse_b: list[float] = []

    for seed in common_seeds:
        ra, rb = runs_a[seed], runs_b[seed]
        Hi = None
        for key in keys:
            ya = np.asarray(_run_metric_series(ra, key), dtype=float)
            yb = np.asarray(_run_metric_series(rb, key), dtype=float)
            n = min(len(ya), len(yb))
            if n == 0:
                Hi = 0
                break
            Hi = n if Hi is None else min(Hi, n)
        if not Hi:
            continue
        common_H.append(int(Hi))

        for key in keys:
            ya = np.asarray(_run_metric_series(ra, key), dtype=float)[:Hi]
            yb = np.asarray(_run_metric_series(rb, key), dtype=float)[:Hi]
            delta = ya - yb
            delta_paths[key].append(delta)
            bucket = out_metrics.setdefault(
                key,
                {
                    "mean_path_diffs": [],
                    "auc_diffs": [],
                    "end_of_window_diffs": [],
                    "a_mean_path": [],
                    "b_mean_path": [],
                    "a_auc": [],
                    "b_auc": [],
                },
            )
            bucket["mean_path_diffs"].append(float(np.mean(delta)))
            bucket["auc_diffs"].append(float(trapz(ya) - trapz(yb)))
            bucket["end_of_window_diffs"].append(float(delta[-1]))
            bucket["a_mean_path"].append(float(np.mean(ya)))
            bucket["b_mean_path"].append(float(np.mean(yb)))
            bucket["a_auc"].append(float(trapz(ya)))
            bucket["b_auc"].append(float(trapz(yb)))

        ca = ra.get("collapse_step", ra.get("survival_time"))
        cb = rb.get("collapse_step", rb.get("survival_time"))
        if ca is not None and cb is not None:
            collapse_a.append(float(ca))
            collapse_b.append(float(cb))

    metrics_summary = {}
    for key, bucket in out_metrics.items():
        metrics_summary[key] = {
            "mean_path_difference": _diff_summary(bucket["mean_path_diffs"]),
            "auc_difference": _diff_summary(bucket["auc_diffs"]),
            # End of *common window*, not padded final-horizon SR_T.
            "end_of_common_window_difference": _diff_summary(bucket["end_of_window_diffs"]),
            f"{label_a}_mean_path": _diff_summary(bucket["a_mean_path"]),
            f"{label_b}_mean_path": _diff_summary(bucket["b_mean_path"]),
            f"{label_a}_auc": _diff_summary(bucket["a_auc"]),
            f"{label_b}_auc": _diff_summary(bucket["b_auc"]),
        }

    collapse_delta = [a - b for a, b in zip(collapse_a, collapse_b)]
    return {
        "n_pairs": len(common_seeds),
        "common_seeds": common_seeds,
        "common_window_H": _diff_summary(common_H),
        "window_rule": "H_i = min(len_a, len_b) on native (unpadded) series",
        "label_a": label_a,
        "label_b": label_b,
        "metrics": metrics_summary,
        "collapse_step": {
            f"{label_a}_mean": float(np.mean(collapse_a)) if collapse_a else float("nan"),
            f"{label_b}_mean": float(np.mean(collapse_b)) if collapse_b else float("nan"),
            "difference": _diff_summary(collapse_delta),
            "n_observed_pairs": len(collapse_delta),
        },
        # Backward-compatible SR view used by older plot helpers.
        "mean_path_difference": metrics_summary.get("sr", {}).get("mean_path_difference", {}),
        "auc_difference": metrics_summary.get("sr", {}).get("auc_difference", {}),
        "final_difference": metrics_summary.get("sr", {}).get(
            "end_of_common_window_difference", {}
        ),
        "delta_paths": delta_paths.get("sr", []),
        "delta_paths_by_key": delta_paths,
    }


def calculate_paired_sr_statistics(
    dec_art: dict,
    cen_art: dict,
) -> dict:
    """DEN−CEN paired path stats on common windows (SR + CGR)."""
    stats = calculate_paired_path_statistics(
        dec_art,
        cen_art,
        keys=("sr", "cgr"),
        label_a="den",
        label_b="cen",
    )
    # Keep legacy top-level keys + RFQ-higher diagnostics on SR.
    paths = stats.get("delta_paths") or []
    rfq_higher = [float(np.mean(np.asarray(d, dtype=float) > 0.0)) for d in paths]
    rfq_higher_final = [
        float(np.asarray(d, dtype=float)[-1] > 0.0) for d in paths if len(d)
    ]
    stats["rfq_higher_fraction"] = {
        "mean": float(np.mean(rfq_higher)) if rfq_higher else float("nan"),
        "median": float(np.median(rfq_higher)) if rfq_higher else float("nan"),
    }
    stats["rfq_higher_final_proportion"] = (
        float(np.mean(rfq_higher_final)) if rfq_higher_final else float("nan")
    )
    return stats


def calculate_common_window_writeoff_statistics(
    art_a: dict,
    art_b: dict,
    *,
    label_a: str = "den",
    label_b: str = "cen",
) -> dict:
    """
    Paired cumulative interbank writeoff on common window H_i = min(len_a, len_b).

    Avoids longevity bias: a longer-surviving mechanism must not look worse solely
    because its full-horizon cumulative sum covers more periods.
    """
    runs_a = _runs_by_seed(art_a)
    runs_b = _runs_by_seed(art_b)
    common_seeds = sorted(set(runs_a) & set(runs_b))
    if not common_seeds:
        raise ValueError("No paired Monte Carlo runs with common seeds were found.")

    cum_a: list[float] = []
    cum_b: list[float] = []
    deltas: list[float] = []
    common_H: list[int] = []
    full_a: list[float] = []
    full_b: list[float] = []
    used_seeds: list[int] = []

    for seed in common_seeds:
        ra, rb = runs_a[seed], runs_b[seed]
        ya = np.asarray(ra.get("interbank_writeoff") or [], dtype=float)
        yb = np.asarray(rb.get("interbank_writeoff") or [], dtype=float)
        # Fall back to SR length if writeoff series missing (legacy artifacts).
        if ya.size == 0:
            ya = np.zeros(len(ra.get("raw_sr") or ra.get("sr") or []), dtype=float)
        if yb.size == 0:
            yb = np.zeros(len(rb.get("raw_sr") or rb.get("sr") or []), dtype=float)
        Hi = int(min(len(ya), len(yb)))
        if Hi <= 0:
            continue
        used_seeds.append(int(seed))
        common_H.append(Hi)
        ca = float(np.sum(ya[:Hi]))
        cb = float(np.sum(yb[:Hi]))
        cum_a.append(ca)
        cum_b.append(cb)
        deltas.append(ca - cb)
        full_a.append(float(np.sum(ya)))
        full_b.append(float(np.sum(yb)))

    return {
        "n_pairs": len(cum_a),
        "common_seeds": used_seeds,
        "common_window_H": _diff_summary(common_H),
        "window_rule": (
            "H_i = min(len_writeoff_a, len_writeoff_b); "
            "report sum(writeoff[0:H_i]) — not full-horizon cumulative"
        ),
        "label_a": label_a,
        "label_b": label_b,
        f"{label_a}_common_window_cumulative": _diff_summary(cum_a),
        f"{label_b}_common_window_cumulative": _diff_summary(cum_b),
        "difference_common_window": _diff_summary(deltas),
        f"{label_a}_full_horizon_cumulative": _diff_summary(full_a),
        f"{label_b}_full_horizon_cumulative": _diff_summary(full_b),
        "note": (
            "Prefer difference_common_window for mechanism contrast; "
            "full_horizon_cumulative confounds longevity with loss intensity."
        ),
    }


def save_common_window_path_comparison(
    scenario_artifacts: dict[str, tuple[str, dict]],
    out_path: Path,
) -> Path:
    """Export common-window mean SR/CGR/AUC, writeoff, and collapse_step contrasts."""

    def _art(key: str):
        return scenario_artifacts[key][1] if key in scenario_artifacts else None

    pairs: dict[str, dict] = {}
    writeoff_pairs: dict[str, dict] = {}
    for suffix in (
        "rollover_on_support_on",
        "rollover_off_support_off",
        "rollover_on_support_off",
        "rollover_off_support_on",
    ):
        ka, kb = f"decentralized_{suffix}", f"centralized_{suffix}"
        if _art(ka) is not None and _art(kb) is not None:
            pairs[f"den_minus_cen__{suffix}"] = calculate_paired_path_statistics(
                _art(ka), _art(kb), keys=("sr", "cgr"), label_a="den", label_b="cen"
            )
            writeoff_pairs[f"den_minus_cen__{suffix}"] = (
                calculate_common_window_writeoff_statistics(
                    _art(ka), _art(kb), label_a="den", label_b="cen"
                )
            )

    for mech in ("decentralized", "centralized"):
        ka, kb = f"{mech}_rollover_on_support_on", f"{mech}_rollover_off_support_off"
        if _art(ka) is not None and _art(kb) is not None:
            pairs[f"onon_minus_offoff__{mech}"] = calculate_paired_path_statistics(
                _art(ka), _art(kb), keys=("sr", "cgr"), label_a="onon", label_b="offoff"
            )
            writeoff_pairs[f"onon_minus_offoff__{mech}"] = (
                calculate_common_window_writeoff_statistics(
                    _art(ka), _art(kb), label_a="onon", label_b="offoff"
                )
            )

    def _jsonable(stats: dict) -> dict:
        out = dict(stats)
        out.pop("delta_paths", None)
        out.pop("delta_paths_by_key", None)
        return out

    payload = {
        "primary_path_metrics": [
            "common_window_mean_SR",
            "common_window_mean_CGR",
            "common_window_AUC_SR",
            "common_window_AUC_CGR",
            "common_window_cumulative_interbank_writeoff",
            "collapse_step",
        ],
        "note": (
            "Do not rank scenarios by padded final-horizon SR_T. "
            "For each seed use H_i=min(len_a,len_b) on native series. "
            "Writeoff contrasts use common-window cumulative sums to avoid longevity bias."
        ),
        "paired_tests": {k: _jsonable(v) for k, v in pairs.items()},
        "paired_writeoff_tests": writeoff_pairs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"Saved: {out_path}")
    for name, res in pairs.items():
        sr = (res.get("metrics") or {}).get("sr") or {}
        cgr = (res.get("metrics") or {}).get("cgr") or {}
        col = (res.get("collapse_step") or {}).get("difference") or {}
        wo = (writeoff_pairs.get(name) or {}).get("difference_common_window") or {}
        print(
            f"[common-window] {name}: "
            f"ΔmeanSR={sr.get('mean_path_difference', {}).get('mean', float('nan')):+.4f} "
            f"ΔmeanCGR={cgr.get('mean_path_difference', {}).get('mean', float('nan')):+.4f} "
            f"ΔAUC_SR={sr.get('auc_difference', {}).get('mean', float('nan')):+.4f} "
            f"Δwriteoff={wo.get('mean', float('nan')):+.2f} "
            f"ΔT={col.get('mean', float('nan')):+.2f}"
        )
    return out_path


def plot_common_window_metrics(
    scenario_artifacts: dict[str, tuple[str, dict]],
    out_dir: Path,
) -> list[Path]:
    """Bar charts for common-window mean SR/CGR/AUC and collapse_step (no final-SR ranking)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def _art(key: str):
        return scenario_artifacts[key][1] if key in scenario_artifacts else None

    # DEN−CEN by feature suffix
    rows = []
    for suffix in (
        "rollover_on_support_on",
        "rollover_off_support_off",
        "rollover_on_support_off",
        "rollover_off_support_on",
    ):
        ka, kb = f"decentralized_{suffix}", f"centralized_{suffix}"
        if _art(ka) is None or _art(kb) is None:
            continue
        stats = calculate_paired_path_statistics(
            _art(ka), _art(kb), keys=("sr", "cgr"), label_a="den", label_b="cen"
        )
        rows.append((suffix, stats))

    if rows:
        labels = [r[0].replace("_", "\n") for r in rows]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        panels = [
            (axes[0, 0], "sr", "mean_path_difference", r"Δ mean SR (DEN−CEN)"),
            (axes[0, 1], "cgr", "mean_path_difference", r"Δ mean CGR (DEN−CEN)"),
            (axes[1, 0], "sr", "auc_difference", r"Δ AUC SR (DEN−CEN)"),
            (axes[1, 1], None, "collapse", r"Δ collapse_step (DEN−CEN)"),
        ]
        x = np.arange(len(rows))
        for ax, key, field, title in panels:
            vals, lo, hi = [], [], []
            for _, stats in rows:
                if field == "collapse":
                    d = (stats.get("collapse_step") or {}).get("difference") or {}
                else:
                    d = ((stats.get("metrics") or {}).get(key) or {}).get(field) or {}
                vals.append(float(d.get("mean", np.nan)))
                lo.append(float(d.get("ci95_lower", np.nan)))
                hi.append(float(d.get("ci95_upper", np.nan)))
            yerr = np.vstack([
                np.asarray(vals) - np.asarray(lo),
                np.asarray(hi) - np.asarray(vals),
            ])
            ax.bar(x, vals, yerr=yerr, capsize=4, alpha=0.85)
            ax.axhline(0.0, lw=1.0, linestyle="--", color="gray")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.3)
        fig.suptitle("Common-window path metrics (no padded final-SR ranking)", fontsize=13)
        fig.tight_layout()
        out_path = out_dir / "compare_common_window_den_minus_cen.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
        saved.append(out_path)

    # ON/ON − OFF/OFF within each mechanism
    onoff_rows = []
    for mech in ("decentralized", "centralized"):
        ka, kb = f"{mech}_rollover_on_support_on", f"{mech}_rollover_off_support_off"
        if _art(ka) is None or _art(kb) is None:
            continue
        stats = calculate_paired_path_statistics(
            _art(ka), _art(kb), keys=("sr", "cgr"), label_a="onon", label_b="offoff"
        )
        onoff_rows.append((mech, stats))

    if onoff_rows:
        labels = [m for m, _ in onoff_rows]
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        panels = [
            (axes[0, 0], "sr", "mean_path_difference", r"Δ mean SR (ON/ON−OFF/OFF)"),
            (axes[0, 1], "cgr", "mean_path_difference", r"Δ mean CGR (ON/ON−OFF/OFF)"),
            (axes[1, 0], "sr", "auc_difference", r"Δ AUC SR (ON/ON−OFF/OFF)"),
            (axes[1, 1], None, "collapse", r"Δ collapse_step (ON/ON−OFF/OFF)"),
        ]
        x = np.arange(len(onoff_rows))
        for ax, key, field, title in panels:
            vals, lo, hi = [], [], []
            for _, stats in onoff_rows:
                if field == "collapse":
                    d = (stats.get("collapse_step") or {}).get("difference") or {}
                else:
                    d = ((stats.get("metrics") or {}).get(key) or {}).get(field) or {}
                vals.append(float(d.get("mean", np.nan)))
                lo.append(float(d.get("ci95_lower", np.nan)))
                hi.append(float(d.get("ci95_upper", np.nan)))
            yerr = np.vstack([
                np.asarray(vals) - np.asarray(lo),
                np.asarray(hi) - np.asarray(vals),
            ])
            ax.bar(x, vals, yerr=yerr, capsize=4, alpha=0.85, color=["#4C72B0", "#DD8452"][: len(vals)])
            ax.axhline(0.0, lw=1.0, linestyle="--", color="gray")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.3)
        fig.suptitle("Common-window ON/ON vs OFF/OFF (same mechanism)", fontsize=13)
        fig.tight_layout()
        out_path = out_dir / "compare_common_window_onon_minus_offoff.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
        saved.append(out_path)

    return saved


def plot_scenario_theta_final_sr(
    scenario_artifacts: dict[str, tuple[str, dict]],
    out_path: Path,
    *,
    title: str = r"θ sweep — final $SR_T$",
    legend_ncol: int | None = None,
) -> Path | None:
    """Deprecated: padded/early-stop final SR_T is not comparable across scenarios."""
    print(
        f"[skip] final-SR ranking plot disabled ({out_path.name}); "
        "use common-window mean SR/CGR/AUC + collapse_step instead"
    )
    return None


def plot_all_scenario_theta_final_sr(
    scenario_artifacts: dict[str, tuple[str, dict]],
    out_dir: Path,
) -> list[Path]:
    """Deprecated wrapper kept for call-site compatibility; emits no ranking plots."""
    print(
        "[skip] plot_all_scenario_theta_final_sr disabled "
        "(cross early-stop final-SR ranking is invalid)"
    )
    return []


def plot_paired_sr_difference(
    dec_art: dict,
    cen_art: dict,
    out_path: Path,
) -> Path:

    stats = calculate_paired_sr_statistics(
        dec_art,
        cen_art,
    )

    paths = stats["delta_paths"]

    if not paths:
        raise ValueError(
            "No paired SR trajectories."
        )

    min_len = min(len(x) for x in paths)

    mat = np.vstack([
        np.asarray(x[:min_len], dtype=float)
        for x in paths
    ])

    mean = np.mean(mat, axis=0)
    std = np.std(mat, axis=0, ddof=1)

    se = std / np.sqrt(mat.shape[0])
    half = 1.96 * se

    lower = mean - half
    upper = mean + half

    xs = np.arange(1, min_len + 1)

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.plot(
        xs,
        mean,
        lw=2.2,
        label="Mean paired difference",
    )

    ax.fill_between(
        xs,
        lower,
        upper,
        alpha=0.2,
        label="95% CI",
    )

    ax.axhline(
        0.0,
        lw=1.2,
        linestyle="--",
    )

    ax.set_xlabel("Time Step")
    ax.set_ylabel(
        r"$\Delta SR_t$ (RFQ - Centralized)"
    )

    ax.set_title(
        "Paired Difference in Systemic Risk"
    )

    ax.grid(True, alpha=0.35)
    ax.legend()

    fig.tight_layout()

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {out_path}")

    return out_path


def save_paired_statistics(
    stats: dict,
    out_path: Path,
) -> Path:

    clean = {
        k: v
        for k, v in stats.items()
        if k not in ("delta_paths", "delta_paths_by_key")
    }

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            clean,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved: {out_path}")

    return out_path


def calculate_paired_network_statistics(
    dec_art: dict,
    cen_art: dict,
) -> dict:

    dec_runs = {
        int(x["seed"]): x
        for x in dec_art.get(
            "network_summary_runs", []
        )
    }

    cen_runs = {
        int(x["seed"]): x
        for x in cen_art.get(
            "network_summary_runs", []
        )
    }

    common_seeds = sorted(
        set(dec_runs) & set(cen_runs)
    )

    metrics = [
        "active_links",
        "network_density",
        "mean_degree",
        "degree_std",
        "mean_weighted_degree",
        "weighted_degree_std",
        "largest_bilateral_exposure",
        "exposure_hhi",
        "largest_component_size",
        "largest_component_share",
        "repeated_counterparty_share",
        "cumulative_transaction_volume",
        "funding_satisfaction_ratio",
        "cumulative_unmet_demand",
        "cumulative_clearing_shortfall",
        "cumulative_interbank_writeoff",
        "funded_q_weighted",
        "project_default_principal",
        "project_default_lgd_loss",
        "project_default_q_weighted_lgd_loss",
        "project_negative_return_loss",
        "funding_gap",
    ]

    def summary(x):
        x = np.asarray(x, dtype=float)
        n = len(x)

        if n == 0:
            return {
                "n": 0,
                "mean": 0.0,
                "std": 0.0,
                "ci95_lower": 0.0,
                "ci95_upper": 0.0,
            }

        mean = float(np.mean(x))

        if n > 1:
            std = float(
                np.std(x, ddof=1)
            )
            half = (
                1.96
                * std
                / np.sqrt(n)
            )
        else:
            std = 0.0
            half = 0.0

        return {
            "n": n,
            "mean": mean,
            "std": std,
            "ci95_lower": mean - half,
            "ci95_upper": mean + half,
        }

    output = {
        "n_pairs": len(common_seeds),
        "metrics": {},
    }

    for metric in metrics:
        rfq_vals = []
        centralized_vals = []
        for s in common_seeds:
            dv = dec_runs[s].get(metric)
            cv = cen_runs[s].get(metric)
            if dv is None or cv is None:
                continue
            try:
                dvf = float(dv)
                cvf = float(cv)
            except (TypeError, ValueError):
                continue
            if dvf != dvf or cvf != cvf:
                continue
            rfq_vals.append(dvf)
            centralized_vals.append(cvf)

        rfq = np.asarray(rfq_vals, dtype=float)
        centralized = np.asarray(centralized_vals, dtype=float)
        delta = rfq - centralized

        output["metrics"][metric] = {
            "centralized":
                summary(centralized),

            "rfq":
                summary(rfq),

            "paired_difference_rfq_minus_centralized":
                summary(delta),
        }

    def _reason_counts(runs):
        events = []
        for s in common_seeds:
            events.extend(runs[s].get("first_failures") or [])
        from bank_econ_shared import first_failure_reason_counts
        return first_failure_reason_counts(events)

    output["first_failure_reason_counts"] = {
        "rfq": _reason_counts(dec_runs),
        "centralized": _reason_counts(cen_runs),
    }
    output["first_failures_by_seed"] = {
        str(s): {
            "rfq": list(dec_runs[s].get("first_failures") or []),
            "centralized": list(cen_runs[s].get("first_failures") or []),
        }
        for s in common_seeds
    }

    return output


def _run0_shock_paths(art: dict) -> dict:
    extra, _ = _extra_plot_bundle(art)
    base = art.get("baseline_run0") or {}
    return {
        "common_liquidity_shock": list(
            extra.get("common_liquidity_shock")
            or _artifact_shock_flags(art)
            or []
        ),
        "sr": [float(x) for x in (base.get("sr") or [])],
        "fr": [float(x) for x in (base.get("fr") or [])],
        "cbs": [float(x) for x in (base.get("cbs") or [])],
        "cgr": [float(x) for x in (base.get("cgr") or [])],
        "unmet_demand_rate": [float(x) for x in (extra.get("unmet_demand_rate") or [])],
        "mean_lcr": [float(x) for x in (extra.get("mean_lcr") or [])],
        "funding_gap": [float(x) for x in (extra.get("funding_gap") or [])],
        "total_volume": [float(x) for x in (extra.get("total_volume") or [])],
    }


def save_network_statistics(
    stats: dict,
    out_path: Path,
) -> Path:

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved: {out_path}")
    return out_path


def _paired_collapse_same(dec_art: dict, cen_art: dict) -> bool | None:
    dec = {
        int(r["seed"]): r.get("collapse_step")
        for r in (dec_art.get("baseline_runs") or [])
    }
    cen = {
        int(r["seed"]): r.get("collapse_step")
        for r in (cen_art.get("baseline_runs") or [])
    }
    common = sorted(set(dec) & set(cen))
    if not common:
        return None
    return all(dec[s] == cen[s] for s in common)


def print_screening_diagnosis(dec_art: dict, cen_art: dict, network_stats: dict) -> None:
    """Print funded_q / loss / first-fail diagnostics and the three judgment rules."""
    from bank_econ_shared import diagnose_screening_channel

    metrics = network_stats.get("metrics") or {}

    def _mean(name: str, side: str):
        rec = (metrics.get(name) or {}).get(side) or {}
        if int(rec.get("n", 0) or 0) <= 0:
            return None
        val = rec.get("mean")
        try:
            x = float(val)
        except (TypeError, ValueError):
            return None
        if x != x:
            return None
        return x

    fq_d = _mean("funded_q_weighted", "rfq")
    fq_c = _mean("funded_q_weighted", "centralized")
    lgd_d = _mean("project_default_lgd_loss", "rfq")
    lgd_c = _mean("project_default_lgd_loss", "centralized")
    qlgd_d = _mean("project_default_q_weighted_lgd_loss", "rfq")
    qlgd_c = _mean("project_default_q_weighted_lgd_loss", "centralized")
    prin_d = _mean("project_default_principal", "rfq")
    prin_c = _mean("project_default_principal", "centralized")
    neg_d = _mean("project_negative_return_loss", "rfq")
    neg_c = _mean("project_negative_return_loss", "centralized")
    wo_d = _mean("cumulative_interbank_writeoff", "rfq")
    wo_c = _mean("cumulative_interbank_writeoff", "centralized")
    gap_d = _mean("funding_gap", "rfq")
    gap_c = _mean("funding_gap", "centralized")

    def _fmt(x):
        return "nan" if x is None else f"{x:.4g}"

    print(
        "[diag] screening paired "
        f"funded_q DEN={_fmt(fq_d)} CEN={_fmt(fq_c)} "
        f"default_prin DEN={_fmt(prin_d)} CEN={_fmt(prin_c)} "
        f"lgd DEN={_fmt(lgd_d)} CEN={_fmt(lgd_c)} "
        f"q_lgd DEN={_fmt(qlgd_d)} CEN={_fmt(qlgd_c)} "
        f"neg_ret DEN={_fmt(neg_d)} CEN={_fmt(neg_c)} "
        f"ib_wo DEN={_fmt(wo_d)} CEN={_fmt(wo_c)} "
        f"gap DEN={_fmt(gap_d)} CEN={_fmt(gap_c)}"
    )
    collapse_same = _paired_collapse_same(dec_art, cen_art)
    print(f"[diag] screening collapse_step_same={collapse_same}")
    for line in diagnose_screening_channel(
        funded_q_den=fq_d,
        funded_q_cen=fq_c,
        project_lgd_den=lgd_d,
        project_lgd_cen=lgd_c,
        collapse_same=collapse_same,
    ):
        print(line)
    counts = network_stats.get("first_failure_reason_counts") or {}
    print(
        "[diag] screening first_fail "
        f"DEN={counts.get('rfq') or {}} "
        f"CEN={counts.get('centralized') or {}}"
    )
    by_seed = network_stats.get("first_failures_by_seed") or {}
    for seed, pair in by_seed.items():
        den_ev = list(pair.get("rfq") or [])
        cen_ev = list(pair.get("centralized") or [])

        def _first(events):
            if not events:
                return "none"
            ev = events[0]
            return f"{ev.get('bank_idx')}:{ev.get('reason')}@t{ev.get('step')}"

        print(
            f"[diag] screening first_fail seed={seed} "
            f"DEN_first={_first(den_ev)} DEN_n={len(den_ev)} "
            f"CEN_first={_first(cen_ev)} CEN_n={len(cen_ev)}"
        )


def _horizon_pad(latest: int, *, pad_frac: float = 0.08, pad_min: int = 40) -> int:
    """Extra blank after the latest data point."""
    latest = max(0, int(latest))
    if latest <= 0:
        return 0
    return max(int(pad_min), int(round(float(pad_frac) * latest)))


def _artifact_data_horizon(artifact: dict, key: str = "sr") -> int:
    """Longest observed_until among runs (not every replication's end)."""
    runs = artifact.get("baseline_runs") or []
    lengths = []
    for run in runs:
        observed = run.get("observed_until")
        if observed is not None:
            lengths.append(int(observed))
        else:
            series = run.get(key) or []
            if series:
                lengths.append(len(series))
    if lengths:
        return int(max(lengths))
    baseline = (artifact.get("baseline") or {}).get(key) or []
    if baseline:
        return int(len(baseline))
    ph = artifact.get("plot_horizon")
    return int(ph) if ph is not None else 0


def _artifact_plot_horizon(
    artifact: dict,
    key: str = "sr",
    *,
    pad_frac: float = 0.08,
    pad_min: int = 40,
) -> int:
    """Axis end = latest data end + pad (blank margin; never T_cap)."""
    latest = _artifact_data_horizon(artifact, key=key)
    if latest <= 0:
        return 0
    return int(latest + _horizon_pad(latest, pad_frac=pad_frac, pad_min=pad_min))


def plot_baseline_comparison(
    dec_art: dict,
    cen_art: dict,
    out_path: Path,
) -> Path:
    dec_series = dec_art["baseline"]
    cen_series = cen_art["baseline"]

    dec_runs = dec_art.get("baseline_runs", [])
    cen_runs = cen_art.get("baseline_runs", [])

    cen_auction = (cen_art.get("extra_metrics") or {}).get("is_auction_day") or []
    if not cen_auction:
        cen_auction = (cen_art.get("extra_metrics_run0") or {}).get("is_auction_day") or []
    shock_flags = _artifact_shock_flags(dec_art) or _artifact_shock_flags(cen_art)
    dec_run0 = dec_art.get("baseline_run0") or {}
    cen_run0 = cen_art.get("baseline_run0") or {}
    weights = dec_art.get("weights", (0.5, 0.3, 0.2))
    theta = dec_art.get("theta", THETA)

    # Data through latest collapse; axis a bit longer so endings are not flush to the frame.
    D_dec = _artifact_data_horizon(dec_art)
    D_cen = _artifact_data_horizon(cen_art)
    H = max(_artifact_plot_horizon(dec_art), _artifact_plot_horizon(cen_art))
    if H <= 0:
        H = max(
            len(np.asarray(dec_series.get("sr", []), dtype=float)),
            len(np.asarray(cen_series.get("sr", []), dtype=float)),
        )

    labels = [
        ("Systemic Risk (SR)", "raw_sr"),
        ("FR (Failure Rate)", "fr"),
        ("CBS (active banks with CAR<θ / active banks)", "cbs"),
        ("CGR (gap / required)", "cgr"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.ravel()

    for ax, (title, key) in zip(axes, labels):
        # Prefer full Monte Carlo paths (baseline may have been truncated on older exports).
        if dec_runs:
            dec_stats = _mean_ci95_from_runs(dec_runs, key)
            y_dec = np.asarray(dec_stats["mean"], dtype=float)
            if D_dec > 0:
                y_dec = y_dec[:D_dec]
        else:
            series_key = key if key in dec_series else ("sr" if key in ("raw_sr", "sr") else key)
            y_dec = np.asarray(dec_series.get(series_key, []), dtype=float)
            if D_dec > 0:
                y_dec = y_dec[:D_dec]
            dec_stats = None

        if cen_runs:
            cen_stats = _mean_ci95_from_runs(cen_runs, key)
            y_cen = np.asarray(cen_stats["mean"], dtype=float)
            if D_cen > 0:
                y_cen = y_cen[:D_cen]
        else:
            series_key = key if key in cen_series else ("sr" if key in ("raw_sr", "sr") else key)
            y_cen = np.asarray(cen_series.get(series_key, []), dtype=float)
            if D_cen > 0:
                y_cen = y_cen[:D_cen]
            cen_stats = None

        _mark_auction_days(ax, cen_auction[: max(len(y_cen), len(y_dec), 0)])
        _mark_shock_days(ax, shock_flags[: max(len(y_cen), len(y_dec), 0)])

        if len(y_dec):
            ax.plot(
                np.arange(1, len(y_dec) + 1),
                y_dec,
                lw=2.0,
                label="Decentralized mean",
            )

        if len(y_cen):
            ax.plot(
                np.arange(1, len(y_cen) + 1),
                y_cen,
                lw=2.0,
                label="Centralized mean",
            )

        r0_key = "sr" if key == "raw_sr" else key
        y_dec_r0 = np.asarray(dec_run0.get(r0_key, []), dtype=float)
        y_cen_r0 = np.asarray(cen_run0.get(r0_key, []), dtype=float)
        if D_dec > 0 and len(y_dec_r0):
            y_dec_r0 = y_dec_r0[:D_dec]
        if D_cen > 0 and len(y_cen_r0):
            y_cen_r0 = y_cen_r0[:D_cen]
        if len(y_dec_r0):
            ax.plot(
                np.arange(1, len(y_dec_r0) + 1),
                y_dec_r0,
                lw=1.05,
                ls=":",
                alpha=0.9,
                label="Decentralized run0",
            )
        if len(y_cen_r0):
            ax.plot(
                np.arange(1, len(y_cen_r0) + 1),
                y_cen_r0,
                lw=1.05,
                ls=":",
                alpha=0.9,
                label="Centralized run0",
            )

        if dec_stats is not None and len(y_dec):
            nd = min(len(y_dec), len(dec_stats["lower"]))
            if nd > 0:
                ax.fill_between(
                    np.arange(1, nd + 1),
                    dec_stats["lower"][:nd],
                    dec_stats["upper"][:nd],
                    alpha=0.16,
                    label="Decentralized 95% CI",
                )

        if cen_stats is not None and len(y_cen):
            nc = min(len(y_cen), len(cen_stats["lower"]))
            if nc > 0:
                ax.fill_between(
                    np.arange(1, nc + 1),
                    cen_stats["lower"][:nc],
                    cen_stats["upper"][:nc],
                    alpha=0.16,
                    label="Centralized 95% CI",
                )

        ax.set_title(title)
        ax.set_ylabel("Value (0–1)")
        ax.set_ylim(-0.02, 1.02)
        if H > 0:
            ax.set_xlim(1, H)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Baseline Trajectory Comparison — W={tuple(weights)}, θ={theta} "
        f"(grey=auction day, red=common liquidity shock)\n"
        f"solid=cross-seed mean; dotted=paired-seed run0; longest Dec≤{D_dec}, Cen≤{D_cen}; axis H={H}",
        fontsize=12,
    )
    axes[-1].set_xlabel(f"Time Step (1–{H})")
    axes[-2].set_xlabel(f"Time Step (1–{H})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_collapse_index_comparison(
    dec_art: dict,
    cen_art: dict,
    out_path: Path,
) -> Path:
    """Monotonic cumulative max of SR (display only; not used by policy)."""
    dec_runs = dec_art.get("baseline_runs", [])
    cen_runs = cen_art.get("baseline_runs", [])
    dec_series = dec_art.get("baseline") or {}
    cen_series = cen_art.get("baseline") or {}
    D_dec = _artifact_data_horizon(dec_art, key="collapse_index")
    D_cen = _artifact_data_horizon(cen_art, key="collapse_index")
    if D_dec <= 0:
        D_dec = _artifact_data_horizon(dec_art, key="sr")
    if D_cen <= 0:
        D_cen = _artifact_data_horizon(cen_art, key="sr")
    H = max(
        _artifact_plot_horizon(dec_art, key="collapse_index"),
        _artifact_plot_horizon(cen_art, key="collapse_index"),
    )
    if H <= 0:
        H = max(D_dec, D_cen)

    fig, ax = plt.subplots(figsize=(12, 6))
    if dec_runs:
        dec_stats = _mean_ci95_from_runs(dec_runs, "collapse_index")
        y_dec = np.asarray(dec_stats["mean"], dtype=float)
        if D_dec > 0:
            y_dec = y_dec[:D_dec]
    else:
        y_dec = np.asarray(
            dec_series.get("collapse_index") or dec_series.get("sr") or [],
            dtype=float,
        )
        dec_stats = None
    if cen_runs:
        cen_stats = _mean_ci95_from_runs(cen_runs, "collapse_index")
        y_cen = np.asarray(cen_stats["mean"], dtype=float)
        if D_cen > 0:
            y_cen = y_cen[:D_cen]
    else:
        y_cen = np.asarray(
            cen_series.get("collapse_index") or cen_series.get("sr") or [],
            dtype=float,
        )
        cen_stats = None

    if len(y_dec):
        ax.plot(np.arange(1, len(y_dec) + 1), y_dec, lw=2.0, label="Decentralized mean")
    if len(y_cen):
        ax.plot(np.arange(1, len(y_cen) + 1), y_cen, lw=2.0, label="Centralized mean")
    if dec_stats is not None and len(y_dec):
        nd = min(len(y_dec), len(dec_stats["lower"]))
        if nd > 0:
            ax.fill_between(
                np.arange(1, nd + 1),
                dec_stats["lower"][:nd],
                dec_stats["upper"][:nd],
                alpha=0.16,
                label="Decentralized 95% CI",
            )
    if cen_stats is not None and len(y_cen):
        nc = min(len(y_cen), len(cen_stats["lower"]))
        if nc > 0:
            ax.fill_between(
                np.arange(1, nc + 1),
                cen_stats["lower"][:nc],
                cen_stats["upper"][:nc],
                alpha=0.16,
                label="Centralized 95% CI",
            )
    ax.set_title("Cumulative Collapse Index")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Collapse Index")
    ax.set_ylim(-0.02, 1.02)
    if H > 0:
        ax.set_xlim(1, H)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def _series_or_fallback(primary: dict, fallback: dict, key: str) -> np.ndarray:
    y = primary.get(key)
    if y is not None and len(y) > 0:
        return np.asarray(y, dtype=float)
    return np.asarray(fallback.get(key, []), dtype=float)


def plot_extra_metrics_comparison(
    dec_art: dict,
    cen_art: dict,
    out_path: Path,
) -> Path:
    dec_x, dec_run0 = _extra_plot_bundle(dec_art)
    cen_x, cen_run0 = _extra_plot_bundle(cen_art)
    dec_mean = dec_art.get("extra_metrics") or {}
    cen_mean = cen_art.get("extra_metrics") or {}
    panels = [
        ("Total Trade Volume", "total_volume"),
        ("Unmet Demand Rate", "unmet_demand_rate"),
        ("Funding Gap", "funding_gap"),
        ("Mean LCR", "mean_lcr"),
        ("Network Density", "network_density"),
        ("Num Trades", "num_trades"),
        ("Interbank Writeoff", "interbank_writeoff_amount"),
        ("EN Unpaid Amount", "en_unpaid_amount"),
    ]
    H = max(_artifact_plot_horizon(dec_art), _artifact_plot_horizon(cen_art))
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
    axes = axes.ravel()
    cen_auction = (
        cen_x.get("is_auction_day")
        or (cen_art.get("extra_metrics_run0") or {}).get("is_auction_day")
        or []
    )
    shock_flags = _artifact_shock_flags(dec_art) or _artifact_shock_flags(cen_art)
    src_note = "run0" if (dec_run0 or cen_run0) else "cross-seed mean"
    for ax, (title, key) in zip(axes, panels):
        y_dec = _series_or_fallback(dec_x, dec_mean, key)
        y_cen = _series_or_fallback(cen_x, cen_mean, key)
        n_dec = len(y_dec)
        n_cen = len(y_cen)
        if n_dec == 0 and n_cen == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        else:
            _mark_auction_days(ax, cen_auction[: max(n_cen, n_dec, 0)])
            _mark_shock_days(ax, shock_flags[: max(n_cen, n_dec, 0)])
            if n_dec:
                ax.plot(np.arange(1, n_dec + 1), y_dec, lw=1.8, label="Decentralized (RFQ)")
            if n_cen:
                ax.plot(np.arange(1, n_cen + 1), y_cen, lw=1.8, label="Centralized")
            ax.legend(fontsize=7)
            n_plot = max(H, n_dec, n_cen)
            if n_plot > 0:
                ax.set_xlim(1, n_plot)
        ax.set_title(title)
        ax.grid(True, alpha=0.35)

    ax = axes[8]
    ax.axis("off")
    dec_defs = dec_mean.get("defaulted_bank_ids", [])
    cen_defs = cen_mean.get("defaulted_bank_ids", [])
    dec_last = dec_defs[-1] if dec_defs else []
    cen_last = cen_defs[-1] if cen_defs else []
    dec_h = int(len(_series_or_fallback(dec_x, dec_mean, "total_volume")))
    cen_h = int(len(_series_or_fallback(cen_x, cen_mean, "total_volume")))
    n_shock = int(np.sum(np.asarray(shock_flags, dtype=float) >= 0.5)) if shock_flags else 0
    ax.text(
        0.02, 0.55,
        f"Shock-aligned extra metrics ({src_note})\n"
        f"red = common liquidity-shock days (n={n_shock})\n"
        f"Defaulted bank ids (run0, last step)\n"
        f"Decentralized: {dec_last}\n"
        f"Centralized: {cen_last}\n"
        f"horizon: Dec={dec_h}, Cen={cen_h}",
        transform=ax.transAxes, va="center", fontsize=10,
    )
    fig.suptitle(
        f"Extra Metrics Comparison ({src_note}) — red=common liquidity shock; "
        f"horizon Dec={dec_h}, Cen={cen_h}",
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


EVENT_STUDY_KEYS = (
    "unmet_demand_rate",
    "total_volume",
    "funding_gap",
    "mean_lcr",
    "network_density",
    "sr",
)


def _pool_event_study(
    shock_runs: list,
    key: str,
    *,
    pre: int = 2,
    post: int = 5,
    series_by_seed: dict | None = None,
) -> dict:
    from bank_econ_shared import event_study_around_flags

    obs: dict[int, list[float]] = {}
    n_events = 0
    n_runs = 0
    for run in shock_runs or []:
        flags = run.get("common_liquidity_shock") or []
        if series_by_seed is not None:
            series = series_by_seed.get(int(run.get("seed", -1)), [])
        else:
            series = run.get(key) or []
        rec = event_study_around_flags(flags, series, pre=pre, post=post)
        n_events += int(rec.get("n_events") or 0)
        n_runs += 1
        for tau, vals in (rec.get("obs") or {}).items():
            obs.setdefault(int(tau), []).extend(list(vals))
    taus = list(range(-int(pre), int(post) + 1))
    means = []
    n_obs = []
    for tau in taus:
        vals = obs.get(tau) or []
        n_obs.append(len(vals))
        means.append(float(np.mean(vals)) if vals else None)
    return {
        "n_runs": n_runs,
        "n_events": n_events,
        "tau": taus,
        "mean": means,
        "n_obs": n_obs,
    }


def calculate_shock_event_study(
    dec_art: dict,
    cen_art: dict,
    *,
    pre: int = 2,
    post: int = 5,
) -> dict:
    dec_shock = dec_art.get("shock_path_runs") or []
    cen_shock = cen_art.get("shock_path_runs") or []
    dec_sr = {
        int(r["seed"]): r.get("raw_sr") or r.get("sr") or []
        for r in (dec_art.get("baseline_runs") or [])
    }
    cen_sr = {
        int(r["seed"]): r.get("raw_sr") or r.get("sr") or []
        for r in (cen_art.get("baseline_runs") or [])
    }
    out = {
        "pre": int(pre),
        "post": int(post),
        "note": (
            "Days aligned to common liquidity-shock flags. "
            "This is the shock-response path; network_statistics means are period averages."
        ),
        "metrics": {},
    }
    for key in EVENT_STUDY_KEYS:
        if key == "sr":
            out["metrics"][key] = {
                "rfq": _pool_event_study(
                    dec_shock, key, pre=pre, post=post, series_by_seed=dec_sr
                ),
                "centralized": _pool_event_study(
                    cen_shock, key, pre=pre, post=post, series_by_seed=cen_sr
                ),
            }
        else:
            out["metrics"][key] = {
                "rfq": _pool_event_study(dec_shock, key, pre=pre, post=post),
                "centralized": _pool_event_study(cen_shock, key, pre=pre, post=post),
            }
    return out


def plot_shock_event_study(
    study: dict,
    out_path: Path,
) -> Path | None:
    metrics = study.get("metrics") or {}
    if not metrics:
        print(f"[skip] empty shock event study; skip {out_path.name}")
        return None
    n_events = 0
    for rec in metrics.values():
        n_events = max(
            n_events,
            int(((rec or {}).get("rfq") or {}).get("n_events") or 0),
            int(((rec or {}).get("centralized") or {}).get("n_events") or 0),
        )
    if n_events <= 0:
        print(f"[skip] no common-shock events recorded; skip {out_path.name}")
        return None

    titles = {
        "unmet_demand_rate": "Unmet Demand Rate",
        "total_volume": "Total Trade Volume",
        "funding_gap": "Funding Gap",
        "mean_lcr": "Mean LCR",
        "network_density": "Network Density",
        "sr": "Systemic Risk (SR)",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.ravel()
    for ax, key in zip(axes, EVENT_STUDY_KEYS):
        rec = metrics.get(key) or {}
        den = rec.get("rfq") or {}
        cen = rec.get("centralized") or {}
        tau_d = np.asarray(den.get("tau") or [], dtype=float)
        tau_c = np.asarray(cen.get("tau") or [], dtype=float)
        y_d = np.asarray(
            [np.nan if v is None else float(v) for v in (den.get("mean") or [])],
            dtype=float,
        )
        y_c = np.asarray(
            [np.nan if v is None else float(v) for v in (cen.get("mean") or [])],
            dtype=float,
        )
        ax.axvline(0.0, color="#c0392b", lw=1.1, alpha=0.7)
        if len(tau_d) and len(y_d):
            ax.plot(tau_d, y_d, lw=2.0, marker="o", label="Decentralized")
        if len(tau_c) and len(y_c):
            ax.plot(tau_c, y_c, lw=2.0, marker="s", label="Centralized")
        ax.set_title(titles.get(key, key))
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=7)
        ax.set_xlabel("Days from shock (0 = shock day)")
    fig.suptitle(
        f"Common-shock event study — pooled events n≈{n_events} "
        f"(window −{study.get('pre', 2)}/+{study.get('post', 5)})",
        fontsize=13,
    )
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
) -> Path | None:
    dec_sweep = dec_art.get("theta_sweep") or {}
    cen_sweep = cen_art.get("theta_sweep") or {}
    if (
        not dec_sweep.get("theta_grid")
        or not dec_sweep.get("sr_curves")
        or not cen_sweep.get("theta_grid")
        or not cen_sweep.get("sr_curves")
    ):
        print(f"[skip] θ sweep disabled; skip {out_path.name}")
        return None

    dec_theta = np.asarray(dec_sweep["theta_grid"], dtype=float)
    cen_theta = np.asarray(cen_sweep["theta_grid"], dtype=float)
    dec_curves = [np.asarray(y, dtype=float) for y in dec_sweep["sr_curves"]]
    cen_curves = [np.asarray(y, dtype=float) for y in cen_sweep["sr_curves"]]
    theta = float(dec_art.get("theta", THETA))

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap_dec = plt.get_cmap("Blues")
    cmap_cen = plt.get_cmap("Oranges")
    norm_dec = plt.Normalize(vmin=float(dec_theta.min()), vmax=float(dec_theta.max()))
    norm_cen = plt.Normalize(vmin=float(cen_theta.min()), vmax=float(cen_theta.max()))

    # Draw native curve lengths; do NOT clip to padded axis (that erased the blank margin).
    D_dec = _artifact_data_horizon(dec_art)
    D_cen = _artifact_data_horizon(cen_art)
    data_end = 0

    for th, y in zip(dec_theta, dec_curves):
        if len(y) == 0:
            continue
        if D_dec > 0:
            y = y[:D_dec]
        data_end = max(data_end, len(y))
        xs = np.arange(1, len(y) + 1)
        ax.plot(xs, y, lw=1.6, alpha=0.85, color=cmap_dec(norm_dec(th)), linestyle="-")

    for th, y in zip(cen_theta, cen_curves):
        if len(y) == 0:
            continue
        if D_cen > 0:
            y = y[:D_cen]
        data_end = max(data_end, len(y))
        xs = np.arange(1, len(y) + 1)
        ax.plot(xs, y, lw=1.6, alpha=0.85, color=cmap_cen(norm_cen(th)), linestyle="--")

    dec_base_idx = int(np.argmin(np.abs(dec_theta - theta)))
    cen_base_idx = int(np.argmin(np.abs(cen_theta - theta)))
    if len(dec_curves[dec_base_idx]) > 0:
        y = np.asarray(dec_curves[dec_base_idx], dtype=float)
        if D_dec > 0:
            y = y[:D_dec]
        data_end = max(data_end, len(y))
        xs = np.arange(1, len(y) + 1)
        ax.plot(xs, y, lw=3.0, color="navy", label=f"Decentralized θ={theta:.2f}")
    if len(cen_curves[cen_base_idx]) > 0:
        y = np.asarray(cen_curves[cen_base_idx], dtype=float)
        if D_cen > 0:
            y = y[:D_cen]
        data_end = max(data_end, len(y))
        xs = np.arange(1, len(y) + 1)
        ax.plot(
            xs, y, lw=3.0, color="darkorange",
            linestyle="--", label=f"Centralized θ={theta:.2f}",
        )

    axis_H = int(data_end + _horizon_pad(data_end)) if data_end > 0 else 0
    if axis_H > 0:
        ax.set_xlim(1, axis_H)

    ax.set_title(
        "θ Measure Sweep Comparison (SR)\n"
        f"(same recorded batch as baseline/w1; curves end ≤{data_end}; axis to {axis_H})"
    )
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


def plot_weight_sweep_comparison(
    dec_art: dict,
    cen_art: dict,
    out_path: Path,
) -> Path | None:
    """Compare FR-weight sweeps.

    w2:w3 remains fixed at 3:2 while w1 varies.
    """
    dec_ws = dec_art.get("weight_sweep") or {}
    cen_ws = cen_art.get("weight_sweep") or {}
    dec_w1 = np.asarray(dec_ws.get("w1_grid", []), dtype=float)
    cen_w1 = np.asarray(cen_ws.get("w1_grid", []), dtype=float)
    dec_curves = [np.asarray(y, dtype=float) for y in dec_ws.get("sr_curves", [])]
    cen_curves = [np.asarray(y, dtype=float) for y in cen_ws.get("sr_curves", [])]
    if len(dec_w1) == 0 or len(cen_w1) == 0 or not dec_curves or not cen_curves:
        print(f"[warn] missing weight_sweep in artifacts; skip {out_path.name}")
        return None

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap_dec = plt.get_cmap("Blues")
    cmap_cen = plt.get_cmap("Oranges")
    norm_dec = plt.Normalize(vmin=float(dec_w1.min()), vmax=float(dec_w1.max()))
    norm_cen = plt.Normalize(vmin=float(cen_w1.min()), vmax=float(cen_w1.max()))

    for w1, y in zip(dec_w1, dec_curves):
        if len(y) == 0:
            continue
        xs = np.arange(1, len(y) + 1)
        ax.plot(xs, y, lw=1.6, alpha=0.85, color=cmap_dec(norm_dec(w1)), linestyle="-")

    for w1, y in zip(cen_w1, cen_curves):
        if len(y) == 0:
            continue
        xs = np.arange(1, len(y) + 1)
        ax.plot(xs, y, lw=1.6, alpha=0.85, color=cmap_cen(norm_cen(w1)), linestyle="--")

    # Highlight baseline w1=0.5 if present.
    baseline_w1 = 0.5
    for w1_grid, curves, color, ls, name in (
        (dec_w1, dec_curves, "navy", "-", "Decentralized"),
        (cen_w1, cen_curves, "darkorange", "--", "Centralized"),
    ):
        idx = int(np.argmin(np.abs(w1_grid - baseline_w1)))
        if 0 <= idx < len(curves) and len(curves[idx]) > 0:
            xs = np.arange(1, len(curves[idx]) + 1)
            ax.plot(
                xs,
                curves[idx],
                lw=3.0,
                color=color,
                linestyle=ls,
                label=f"{name} w1={float(w1_grid[idx]):.2f}",
            )

    ax.set_title(r"Failure-weight sweep comparison — $SR_t=w_1FR_t+w_2CBS_t+w_3CGR_t$")
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


def plot_pair_shock_baselines(
    dec_art: dict,
    cen_art: dict,
    out_path: Path,
    *,
    title: str,
) -> Path:
    """Paired-seed paths with common-shock marks.

    Recorded risk states stay fixed after an early stop; liquidity panels show the
    contemporaneous shock response that those three series cannot.
    """
    dec_b0 = dec_art.get("baseline_run0") or {}
    cen_b0 = cen_art.get("baseline_run0") or {}
    dec_x, _ = _extra_plot_bundle(dec_art)
    cen_x, _ = _extra_plot_bundle(cen_art)
    dec_mean = dec_art.get("baseline") or {}
    cen_mean = cen_art.get("baseline") or {}
    shock = _artifact_shock_flags(dec_art) or _artifact_shock_flags(cen_art)

    def _y(run0: dict, mean: dict, key: str) -> np.ndarray:
        src = run0.get(key) or mean.get(key) or []
        return np.asarray(src, dtype=float)

    panels = [
        ("SR (solid=run0, faint=mean)", "sr", "unit", True),
        ("FR (solid=run0, faint=mean)", "fr", "unit", True),
        ("Unmet Demand Rate (run0)", "unmet_demand_rate", "unit", False),
        ("Mean LCR (run0)", "mean_lcr", "auto", False),
        ("Funding Gap (run0)", "funding_gap", "auto", False),
        ("CBS (solid=run0, faint=mean)", "cbs", "unit", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=False)
    axes = axes.ravel()
    n_plot = 0
    for ax, (panel_title, key, scale, from_baseline) in zip(axes, panels):
        if from_baseline:
            y_dec = _y(dec_b0, dec_mean, key)
            y_cen = _y(cen_b0, cen_mean, key)
            y_dec_mean = np.asarray(dec_mean.get(key) or [], dtype=float)
            y_cen_mean = np.asarray(cen_mean.get(key) or [], dtype=float)
        else:
            y_dec = np.asarray(dec_x.get(key) or [], dtype=float)
            y_cen = np.asarray(cen_x.get(key) or [], dtype=float)
            y_dec_mean = np.asarray([], dtype=float)
            y_cen_mean = np.asarray([], dtype=float)
        n = max(len(y_dec), len(y_cen), len(y_dec_mean), len(y_cen_mean))
        n_plot = max(n_plot, n)
        _mark_shock_days(ax, shock[:n] if n else shock)
        if len(y_dec_mean):
            ax.plot(
                np.arange(1, len(y_dec_mean) + 1),
                y_dec_mean,
                lw=1.0,
                alpha=0.28,
                color="C0",
            )
        if len(y_cen_mean):
            ax.plot(
                np.arange(1, len(y_cen_mean) + 1),
                y_cen_mean,
                lw=1.0,
                alpha=0.28,
                color="C1",
            )
        if len(y_dec):
            ax.plot(
                np.arange(1, len(y_dec) + 1),
                y_dec,
                lw=1.8,
                color="C0",
                label="Decentralized (RFQ)",
            )
        if len(y_cen):
            ax.plot(
                np.arange(1, len(y_cen) + 1),
                y_cen,
                lw=1.8,
                color="C1",
                label="Centralized",
            )
        ax.set_title(panel_title)
        if scale == "unit":
            ax.set_ylim(-0.02, 1.02)
        if n > 0:
            ax.set_xlim(1, n + max(8, int(0.04 * n)))
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=7)
    for ax in axes[-3:]:
        ax.set_xlabel("Time Step (paired seed 0)")
    n_shock = int(np.sum(np.asarray(shock, dtype=float) >= 0.5)) if shock else 0
    fig.suptitle(
        f"{title}\n"
        f"red = common liquidity-shock days (n={n_shock}); "
        "risk state freezes only after early stop — look at unmet/LCR/gap for shock response",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_scenario_baselines(
    scenario_artifacts: dict[str, tuple[str, dict]],
    out_path: Path,
    *,
    title: str = "Baseline Trajectories",
    legend_ncol: int | None = None,
) -> Path:
    """Descriptive multi-series baselines with panel-specific visible ranges."""
    panel_labels = [
        ("SR (Systemic Risk)", "sr"),
        ("FR (Failure Rate)", "fr"),
        ("CBS (active banks with CAR<θ / active banks)", "cbs"),
        ("CGR (gap / required)", "cgr"),
    ]
    n_series = max(1, len(scenario_artifacts))
    if legend_ncol is None:
        legend_ncol = 2 if n_series >= 4 else n_series
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()
    for ax, (panel_title, key) in zip(axes, panel_labels):
        data_end = 0
        panel_curves = []
        for _, (label, art) in scenario_artifacts.items():
            runs = art.get("baseline_runs") or []
            if runs:
                y = np.asarray(_mean_ci95_from_runs(runs, key)["mean"], dtype=float)
            else:
                y = np.asarray(art["baseline"][key], dtype=float)
            D = _artifact_data_horizon(art)
            if D > 0:
                y = y[:D]
            panel_curves.append(y)
            data_end = max(data_end, len(y))
            xs = np.arange(1, len(y) + 1)
            ax.plot(xs, y, lw=1.7, alpha=0.88, label=label)
        ax.set_title(panel_title)
        observed_max = max(
            (float(np.nanmax(y)) for y in panel_curves if len(y)),
            default=0.0,
        )
        if observed_max <= 0.285:
            y_max = 0.30
        else:
            y_max = min(1.0, max(0.40, np.ceil(observed_max * 11.0) / 10.0))
        ax.set_ylim(0.0, y_max)
        if data_end > 0:
            ax.set_xlim(1, data_end)
        ax.grid(True, alpha=0.35)
    axes[-1].set_xlabel("Time Step")
    axes[-2].set_xlabel("Time Step")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=legend_ncol,
        fontsize=8,
    )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_all_scenario_baselines(
    scenario_artifacts: dict[str, tuple[str, dict]],
    out_dir: Path,
) -> list[Path]:
    """
    Replace the old 8-line overview with:
    - 4 pairwise Dec vs Cen charts (one per rollover×support),
    - 2 four-line charts (Decentralized×4 features; Centralized×4 features).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Four pairwise Dec vs Cen comparisons
    for rollover_enabled, policy_support_enabled in FEATURE_SCENARIOS:
        suffix = _feature_suffix(rollover_enabled, policy_support_enabled)
        dec_key = f"decentralized_{suffix}"
        cen_key = f"centralized_{suffix}"
        subset = {
            k: scenario_artifacts[k]
            for k in (dec_key, cen_key)
            if k in scenario_artifacts
        }
        if len(subset) < 2:
            continue
        r = "on" if rollover_enabled else "off"
        s = "on" if policy_support_enabled else "off"
        saved.append(
            plot_pair_shock_baselines(
                subset[dec_key][1],
                subset[cen_key][1],
                out_dir / f"compare_pair_baselines_{suffix}.png",
                title=f"Baseline Trajectories — Dec vs Cen (R:{r}, Support:{s})",
            )
        )

    # Two four-line mechanism charts
    dec_only = {
        k: v for k, v in scenario_artifacts.items() if k.startswith("decentralized_")
    }
    cen_only = {
        k: v for k, v in scenario_artifacts.items() if k.startswith("centralized_")
    }
    if dec_only:
        saved.append(
            plot_scenario_baselines(
                dec_only,
                out_dir / "compare_four_baselines_decentralized.png",
                title="Decentralized (RFQ) — Four Feature Scenarios",
                legend_ncol=2,
            )
        )
    if cen_only:
        saved.append(
            plot_scenario_baselines(
                cen_only,
                out_dir / "compare_four_baselines_centralized.png",
                title="Centralized — Four Feature Scenarios",
                legend_ncol=2,
            )
        )
    return saved


def plot_policy_support_comparison(
    scenario_artifacts: dict[str, tuple[str, dict]],
    out_path: Path,
    *,
    title: str = "Central Bank Policy Support Total",
    legend_ncol: int | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(14, 7))
    any_series = False
    n_series = max(1, len(scenario_artifacts))
    if legend_ncol is None:
        legend_ncol = 2 if n_series >= 4 else n_series
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
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Liquidity + Capital Support")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8, ncol=legend_ncol)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_all_policy_support_comparisons(
    scenario_artifacts: dict[str, tuple[str, dict]],
    out_dir: Path,
) -> list[Path]:
    """4 pairwise + 2 four-line policy-support charts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for rollover_enabled, policy_support_enabled in FEATURE_SCENARIOS:
        suffix = _feature_suffix(rollover_enabled, policy_support_enabled)
        dec_key = f"decentralized_{suffix}"
        cen_key = f"centralized_{suffix}"
        subset = {
            k: scenario_artifacts[k]
            for k in (dec_key, cen_key)
            if k in scenario_artifacts
        }
        if len(subset) < 2:
            continue
        r = "on" if rollover_enabled else "off"
        s = "on" if policy_support_enabled else "off"
        saved.append(
            plot_policy_support_comparison(
                subset,
                out_dir / f"compare_pair_policy_support_{suffix}.png",
                title=f"Policy Support Total — Dec vs Cen (R:{r}, Support:{s})",
                legend_ncol=2,
            )
        )

    dec_only = {
        k: v for k, v in scenario_artifacts.items() if k.startswith("decentralized_")
    }
    cen_only = {
        k: v for k, v in scenario_artifacts.items() if k.startswith("centralized_")
    }
    if dec_only:
        saved.append(
            plot_policy_support_comparison(
                dec_only,
                out_dir / "compare_four_policy_support_decentralized.png",
                title="Policy Support Total — Decentralized (four features)",
                legend_ncol=2,
            )
        )
    if cen_only:
        saved.append(
            plot_policy_support_comparison(
                cen_only,
                out_dir / "compare_four_policy_support_centralized.png",
                title="Policy Support Total — Centralized (four features)",
                legend_ncol=2,
            )
        )
    return saved


def _first_hit_step(series, threshold: float) -> float | None:
    """1-based time step of first FR >= threshold (matches trajectory x-axis)."""
    y = np.asarray(series, dtype=float)
    if y.size == 0:
        return None
    hits = np.where(y >= float(threshold))[0]
    if hits.size == 0:
        return None
    return float(hits[0] + 1)


def failure_arrival_stats(artifact: dict, thresholds=(0.5, 0.9)) -> dict:
    """Per-run first hitting times of FR thresholds + summaries.

    Means are conditional on hitting the threshold within the horizon;
    see hit_rate for the share of runs that hit.
    """
    runs = artifact.get("baseline_runs") or []
    out: dict[str, dict] = {}
    for thr in thresholds:
        times: list[float] = []
        for run in runs:
            t = _first_hit_step(run.get("fr", []), thr)
            if t is not None:
                times.append(float(t))
        key = f"time_to_fr_{str(thr).replace('.', 'p')}"
        if times:
            summary = _summary_scalar(times)
        else:
            summary = {
                "n": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "ci95_lower": float("nan"),
                "ci95_upper": float("nan"),
            }
        summary["hit_rate"] = float(len(times) / len(runs)) if runs else 0.0
        out[key] = summary
    # Also export mean-path hitting times for quick reference.
    fr_mean = (artifact.get("baseline") or {}).get("fr") or []
    out["mean_path"] = {
        f"time_to_fr_{str(thr).replace('.', 'p')}": _first_hit_step(fr_mean, thr)
        for thr in thresholds
    }
    feats = artifact.get("features") or {}
    out["features"] = {
        "rollover_enabled": bool(feats.get("rollover_enabled", True)),
        "policy_support_enabled": bool(feats.get("policy_support_enabled", True)),
    }
    return out


def _run_mean_alive(run: dict) -> float:
    if "mean_alive" in run and run["mean_alive"] is not None:
        return float(run["mean_alive"])
    fr = np.asarray(run.get("fr", []), dtype=float)
    if fr.size == 0:
        return float("nan")
    return float(np.mean(1.0 - fr))


def _run_bank_days_alive(run: dict, *, n_commercial: int = 29) -> float:
    if "bank_days_alive" in run and run["bank_days_alive"] is not None:
        return float(run["bank_days_alive"])
    fr = np.asarray(run.get("fr", []), dtype=float)
    if fr.size == 0:
        return float("nan")
    # Fallback: share-days × N_commercial → bank-days.
    return float(np.sum(1.0 - fr) * max(1, int(n_commercial)))


def _run_rmst_fixed(run: dict, *, horizon: int = 1000) -> float:
    """Restricted collapse-free time: min(observed collapse time, T)."""
    T = max(1, int(horizon))
    collapse = run.get("collapse_step")
    hit = bool(run.get("collapse_hit") or collapse is not None)
    if not hit or collapse is None or float(collapse) > T:
        return float(T)
    return float(max(0.0, min(float(collapse), float(T))))


def _run_bank_days_fixed(
    run: dict, *, horizon: int = 1000, n_commercial: int = 29
) -> float:
    """Alive-bank area through T, padding a stopped path with its last state."""
    T = max(1, int(horizon))
    fr = np.asarray(run.get("fr", []), dtype=float)
    if fr.size == 0:
        return float("nan")
    path = np.clip(1.0 - fr[:T], 0.0, 1.0)
    if path.size < T:
        path = np.pad(path, (0, T - path.size), constant_values=float(path[-1]))
    return float(np.sum(path) * max(1, int(n_commercial)))


def _run_censored_fixed(run: dict, *, horizon: int = 1000) -> bool:
    collapse = run.get("collapse_step")
    return not bool(
        (run.get("collapse_hit") or collapse is not None)
        and collapse is not None
        and float(collapse) <= int(horizon)
    )


def survival_run_metrics(artifact: dict) -> dict:
    """Primary fixed-horizon survival metrics from baseline_runs."""
    runs = artifact.get("baseline_runs") or []
    meta = artifact.get("batch_meta") or {}
    surv_block = artifact.get("survival") or {}
    H = int(surv_block.get("horizon") or meta.get("T_cap") or meta.get("T") or 0)
    mean_alives = [_run_mean_alive(r) for r in runs]
    bank_days = [_run_bank_days_alive(r) for r in runs]
    t50_vals = [r.get("t50") for r in runs]
    t50_hits = [v for v in t50_vals if v is not None]
    t90_vals = [r.get("t90") for r in runs]
    t90_hits = [v for v in t90_vals if v is not None]
    alive_H = [
        float(r["alive_H"]) if r.get("alive_H") is not None
        else (float(1.0 - np.asarray(r.get("fr", [1.0]), dtype=float)[-1]) if r.get("fr") else float("nan"))
        for r in runs
    ]
    t_first = [r.get("t_first") for r in runs]
    t_first_hits = [v for v in t_first if v is not None]
    surv_times = [_run_survival_time(r) for r in runs]
    surv_times_observed = [t for t in surv_times if np.isfinite(t)]
    collapse_hits = [bool(r.get("collapse_hit") or r.get("collapse_step") is not None) for r in runs]
    fixed_T = 1000
    fixed_rmst = [_run_rmst_fixed(r, horizon=fixed_T) for r in runs]
    fixed_bank_days = [_run_bank_days_fixed(r, horizon=fixed_T) for r in runs]
    fixed_censored = [_run_censored_fixed(r, horizon=fixed_T) for r in runs]
    return {
        "horizon": H,
        "n": len(runs),
        "survival_time_conditional": _summary_scalar(surv_times_observed) if surv_times_observed else {},
        "n_observed_collapse": int(len(surv_times_observed)),
        "collapse_hit_rate": float(np.mean(collapse_hits)) if collapse_hits else 0.0,
        "stop_alive_threshold": (
            int(runs[0]["stop_alive_threshold"])
            if runs and runs[0].get("stop_alive_threshold") is not None
            else None
        ),
        "mean_alive": _summary_scalar(mean_alives) if mean_alives else {},
        "bank_days_alive": _summary_scalar(bank_days) if bank_days else {},
        "alive_H": _summary_scalar(alive_H) if alive_H else {},
        "t50_hit_rate": float(len(t50_hits) / len(runs)) if runs else 0.0,
        "t50_mean_conditional": float(np.mean(t50_hits)) if t50_hits else None,
        "t90_hit_rate": float(len(t90_hits) / len(runs)) if runs else 0.0,
        "t90_mean_conditional": float(np.mean(t90_hits)) if t90_hits else None,
        "t_first_mean_conditional": float(np.mean(t_first_hits)) if t_first_hits else None,
        "mean_alive_by_seed": {
            str(int(r.get("seed", i))): float(a)
            for i, (r, a) in enumerate(zip(runs, mean_alives))
        },
        "fixed_horizon": {
            "T": fixed_T,
            "rmst": _summary_scalar(fixed_rmst) if fixed_rmst else {},
            "bank_days_alive": _summary_scalar(fixed_bank_days) if fixed_bank_days else {},
            "censoring_rate": float(np.mean(fixed_censored)) if fixed_censored else 0.0,
        },
    }


def _paired_bootstrap_mean_diff(diff: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {
            "mean": float("nan"),
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
            "win_rate": float("nan"),
        }
    rng = np.random.default_rng(int(seed))
    boots = []
    n = len(diff)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.mean(diff[idx])))
    boots = np.sort(np.asarray(boots, dtype=float))
    return {
        "mean": float(np.mean(diff)),
        "ci95_lower": float(np.quantile(boots, 0.025)),
        "ci95_upper": float(np.quantile(boots, 0.975)),
        "win_rate": float(np.mean(diff > 0.0)),
        "n": int(n),
    }



def _run_survival_time(run: dict) -> float:
    """Observed collapse time only; a horizon stop remains right-censored."""
    collapse_step = run.get("collapse_step")
    collapse_hit = bool(run.get("collapse_hit") or collapse_step is not None)
    if not collapse_hit:
        return float("nan")
    if collapse_step is not None:
        return float(collapse_step)
    if run.get("survival_time") is not None:
        return float(run["survival_time"])
    return float("nan")


def paired_survival_time_comparison(art_a: dict, art_b: dict) -> dict:
    """Paired by seed: survival_time_a - survival_time_b (e.g. ΔT4 = DEN - CEN)."""
    runs_a = {
        int(r.get("seed")): _run_survival_time(r)
        for r in (art_a.get("baseline_runs") or [])
    }
    runs_b = {
        int(r.get("seed")): _run_survival_time(r)
        for r in (art_b.get("baseline_runs") or [])
    }
    common_all = sorted(set(runs_a) & set(runs_b))
    common = [s for s in common_all if np.isfinite(runs_a[s]) and np.isfinite(runs_b[s])]
    censored = [s for s in common_all if s not in common]
    diffs = np.asarray([runs_a[s] - runs_b[s] for s in common], dtype=float)
    boot = _paired_bootstrap_mean_diff(diffs)
    return {
        "n_common_seeds": len(common_all),
        "n_paired_observed": len(common),
        "paired_observed_rate": float(len(common) / len(common_all)) if common_all else 0.0,
        "censored_common_seeds": censored,
        "survival_time_a_mean": float(np.mean([runs_a[s] for s in common])) if common else float("nan"),
        "survival_time_b_mean": float(np.mean([runs_b[s] for s in common])) if common else float("nan"),
        "delta_mean": boot.get("mean", float("nan")),
        "delta_median": float(np.median(diffs)) if len(diffs) else float("nan"),
        "win_rate": boot.get("win_rate", float("nan")),
        "ci95_lower": boot.get("ci95_lower", float("nan")),
        "ci95_upper": boot.get("ci95_upper", float("nan")),
        "n": boot.get("n", 0),
        "observed_seeds": common,
        "deltas_by_seed": {str(s): float(runs_a[s] - runs_b[s]) for s in common},
    }

def paired_mean_alive_comparison(art_a: dict, art_b: dict) -> dict:
    """Paired by seed: mean_alive_a - mean_alive_b."""
    runs_a = {int(r.get("seed")): _run_mean_alive(r) for r in (art_a.get("baseline_runs") or [])}
    runs_b = {int(r.get("seed")): _run_mean_alive(r) for r in (art_b.get("baseline_runs") or [])}
    common = sorted(set(runs_a) & set(runs_b))
    diffs = np.asarray([runs_a[s] - runs_b[s] for s in common], dtype=float)
    return {
        "n_paired": len(common),
        "mean_alive_a": float(np.mean([runs_a[s] for s in common])) if common else float("nan"),
        "mean_alive_b": float(np.mean([runs_b[s] for s in common])) if common else float("nan"),
        **_paired_bootstrap_mean_diff(diffs),
        "seeds": common,
    }


def paired_bank_days_comparison(art_a: dict, art_b: dict) -> dict:
    """Paired by seed: bank_days_alive_a - bank_days_alive_b."""
    runs_a = {int(r.get("seed")): _run_bank_days_alive(r) for r in (art_a.get("baseline_runs") or [])}
    runs_b = {int(r.get("seed")): _run_bank_days_alive(r) for r in (art_b.get("baseline_runs") or [])}
    common = sorted(set(runs_a) & set(runs_b))
    diffs = np.asarray([runs_a[s] - runs_b[s] for s in common], dtype=float)
    return {
        "n_paired": len(common),
        "bank_days_a": float(np.mean([runs_a[s] for s in common])) if common else float("nan"),
        "bank_days_b": float(np.mean([runs_b[s] for s in common])) if common else float("nan"),
        **_paired_bootstrap_mean_diff(diffs),
        "seeds": common,
    }


def paired_rmst_fixed_comparison(
    art_a: dict, art_b: dict, *, horizon: int = 1000
) -> dict:
    """Paired fixed-window RMST and bank-days; positive deltas favor A."""
    T = max(1, int(horizon))
    rows_a = {int(r.get("seed")): r for r in (art_a.get("baseline_runs") or [])}
    rows_b = {int(r.get("seed")): r for r in (art_b.get("baseline_runs") or [])}
    common = sorted(set(rows_a) & set(rows_b))
    rmst_a = np.asarray([_run_rmst_fixed(rows_a[s], horizon=T) for s in common])
    rmst_b = np.asarray([_run_rmst_fixed(rows_b[s], horizon=T) for s in common])
    days_a = np.asarray([_run_bank_days_fixed(rows_a[s], horizon=T) for s in common])
    days_b = np.asarray([_run_bank_days_fixed(rows_b[s], horizon=T) for s in common])
    return {
        "T": T,
        "n_paired": len(common),
        "rmst_a_mean": float(np.mean(rmst_a)) if common else float("nan"),
        "rmst_b_mean": float(np.mean(rmst_b)) if common else float("nan"),
        "rmst_delta": _paired_bootstrap_mean_diff(rmst_a - rmst_b),
        "bank_days_a_mean": float(np.mean(days_a)) if common else float("nan"),
        "bank_days_b_mean": float(np.mean(days_b)) if common else float("nan"),
        "bank_days_delta": _paired_bootstrap_mean_diff(days_a - days_b),
        "censoring_rate_a": (
            float(np.mean([_run_censored_fixed(rows_a[s], horizon=T) for s in common]))
            if common else float("nan")
        ),
        "censoring_rate_b": (
            float(np.mean([_run_censored_fixed(rows_b[s], horizon=T) for s in common]))
            if common else float("nan")
        ),
        "seeds": common,
    }


def save_survival_comparison(
    scenario_artifacts: dict[str, tuple[str, dict]],
    out_path: Path,
) -> Path:
    """Export mean_alive / bank-days / t50 / t90 for den>cen and ON/ON>OFF/OFF."""
    per_scenario = {
        key: {"label": label, **survival_run_metrics(art)}
        for key, (label, art) in scenario_artifacts.items()
    }

    def _art(key: str):
        return scenario_artifacts[key][1] if key in scenario_artifacts else None

    pairs = {}
    # Primary: ΔT = survival_time_den - survival_time_cen (collapse when alive <= threshold)
    for suffix in ("rollover_on_support_on", "rollover_off_support_off",
                   "rollover_on_support_off", "rollover_off_support_on"):
        ka, kb = f"decentralized_{suffix}", f"centralized_{suffix}"
        if _art(ka) is not None and _art(kb) is not None:
            pairs[f"delta_T__den_minus_cen__{suffix}"] = paired_survival_time_comparison(_art(ka), _art(kb))
            pairs[f"den_minus_cen__mean_alive__{suffix}"] = paired_mean_alive_comparison(_art(ka), _art(kb))
            pairs[f"den_minus_cen__bank_days__{suffix}"] = paired_bank_days_comparison(_art(ka), _art(kb))
            pairs[f"den_minus_cen__rmst_T1000__{suffix}"] = paired_rmst_fixed_comparison(_art(ka), _art(kb))

    for mech in ("decentralized", "centralized"):
        ka = f"{mech}_rollover_on_support_on"
        for off_suffix in (
            "rollover_off_support_off",
            "rollover_on_support_off",
            "rollover_off_support_on",
        ):
            kb = f"{mech}_{off_suffix}"
            if _art(ka) is not None and _art(kb) is not None:
                tag = f"onon_minus_{off_suffix}__{mech}"
                pairs[f"delta_T__{tag}"] = paired_survival_time_comparison(_art(ka), _art(kb))
                pairs[f"{tag}__mean_alive"] = paired_mean_alive_comparison(_art(ka), _art(kb))
                pairs[f"{tag}__bank_days"] = paired_bank_days_comparison(_art(ka), _art(kb))
                pairs[f"{tag}__rmst_T1000"] = paired_rmst_fixed_comparison(_art(ka), _art(kb))

    payload = {
        "primary_metric": "survival_time (collapse_step; remaining commercial banks <= stop_alive_threshold)",
        "primary_contrast": "ΔT = T_DEN - T_CEN (paired by seed); ΔT>0 favors DEN",
        "note": "Do not infer survival time from FR padded to the safety cap. Use collapse_step; horizon-only runs are right-censored.",
        "criteria": {
            "den_gt_cen": "among paired observed collapses, ΔT mean > 0, win_rate > 0.5; prefer bootstrap 95% CI lower > 0",
            "onon_gt_offoff": "among paired observed collapses, ΔT (ON/ON - OFF/OFF) mean > 0 and win_rate > 0.5",
            "censoring_check": "paired_observed_rate should be close to 1; otherwise extend the cap or use censored survival analysis",
            "robustness": "repeat with stop_alive_threshold=3",
        },
        "scenarios": per_scenario,
        "paired_tests": pairs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")
    # Concise console verdict
    for name, res in pairs.items():
        display = res.get("rmst_delta") if isinstance(res.get("rmst_delta"), dict) else res
        print(
            f"[survival] {name}: Δ={display.get('delta_mean', display.get('mean', float('nan'))):+.4f} "
            f"win_rate={display.get('win_rate', float('nan')):.2f} "
            f"CI95=[{display.get('ci95_lower', float('nan')):+.4f}, "
            f"{display.get('ci95_upper', float('nan')):+.4f}]"
        )
    return out_path


def save_failure_arrival_statistics(
    scenario_artifacts: dict[str, tuple[str, dict]],
    out_path: Path,
) -> Path:
    payload = {
        key: {"label": label, **failure_arrival_stats(art)}
        for key, (label, art) in scenario_artifacts.items()
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")
    return out_path


def plot_failure_arrival_comparison(
    scenario_artifacts: dict[str, tuple[str, dict]],
    out_path: Path,
) -> Path:
    """Bar chart: mean first hitting time of FR=0.5 / FR=0.9 across scenarios."""
    labels: list[str] = []
    t50: list[float] = []
    t50_lo: list[float] = []
    t50_hi: list[float] = []
    t90: list[float] = []
    t90_lo: list[float] = []
    t90_hi: list[float] = []

    for _, (label, art) in scenario_artifacts.items():
        stats = failure_arrival_stats(art)
        s50 = stats.get("time_to_fr_0p5") or {}
        s90 = stats.get("time_to_fr_0p9") or {}
        if not s50.get("n") and not s90.get("n"):
            continue
        labels.append(label)
        t50.append(float(s50.get("mean", np.nan)))
        t50_lo.append(float(s50.get("ci95_lower", np.nan)))
        t50_hi.append(float(s50.get("ci95_upper", np.nan)))
        t90.append(float(s90.get("mean", np.nan)))
        t90_lo.append(float(s90.get("ci95_lower", np.nan)))
        t90_hi.append(float(s90.get("ci95_upper", np.nan)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x = np.arange(len(labels))

    for ax, means, lo, hi, title in (
        (
            axes[0],
            t50,
            t50_lo,
            t50_hi,
            r"Conditional mean time to $FR\geq 0.5$ (hit runs only)",
        ),
        (
            axes[1],
            t90,
            t90_lo,
            t90_hi,
            r"Conditional mean time to $FR\geq 0.9$ (hit runs only)",
        ),
    ):
        if len(labels) == 0:
            ax.text(0.5, 0.5, "No FR arrival data", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        means_a = np.asarray(means, dtype=float)
        lo_a = np.asarray(lo, dtype=float)
        hi_a = np.asarray(hi, dtype=float)
        valid = np.isfinite(means_a)
        yerr = np.vstack([
            np.where(valid, np.maximum(0.0, means_a - lo_a), 0.0),
            np.where(valid, np.maximum(0.0, hi_a - means_a), 0.0),
        ])
        heights = np.where(valid, means_a, 0.0)
        bars = ax.bar(x, heights, yerr=yerr, capsize=3, alpha=0.85, color="#4C78A8")
        for bar, ok in zip(bars, valid):
            if not ok:
                bar.set_alpha(0.15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Time step (1-based)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.35)

    fig.suptitle(
        "Failure arrival times (conditional on hit; see hit_rate in JSON)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def _summary_scalar(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "ci95_lower": 0.0,
            "ci95_upper": 0.0,
        }
    mean = float(np.mean(x))
    if n > 1:
        std = float(np.std(x, ddof=1))
        half = 1.96 * std / np.sqrt(n)
    else:
        std = 0.0
        half = 0.0
    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
    }


def _artifact_sr_end_summary(artifact: dict) -> dict[str, float]:
    finals: list[float] = []
    for run in artifact.get("baseline_runs", []):
        # End-of-path summary for policy/tests uses raw three-weight SR when available.
        sr = run.get("raw_sr") or run.get("sr", [])
        if sr:
            finals.append(float(sr[-1]))
    if finals:
        return _summary_scalar(finals)
    # Fallback to exported mean path if replication paths are absent.
    baseline = artifact.get("baseline", {}) or {}
    sr = baseline.get("raw_sr") or baseline.get("sr", []) or []
    if not sr:
        return _summary_scalar([])
    end = float(sr[-1])
    return {
        "n": 1,
        "mean": end,
        "std": 0.0,
        "ci95_lower": end,
        "ci95_upper": end,
    }


def _artifact_network_mean(artifact: dict, key: str) -> float:
    vals = [
        float(run.get(key, 0.0))
        for run in artifact.get("network_summary_runs", [])
        if key in run
    ]
    return float(np.mean(vals)) if vals else 0.0


def _json_default(value):
    """Convert NumPy values retained in comparison summaries for JSON export."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def plot_loan_cap_robustness(
    rows: list[dict],
    out_path: Path,
) -> Path:
    Bs = [float(r["B"]) for r in rows]
    cen_mean = [float(r["centralized"]["sr_end"]["mean"]) for r in rows]
    cen_lo = [float(r["centralized"]["sr_end"]["ci95_lower"]) for r in rows]
    cen_hi = [float(r["centralized"]["sr_end"]["ci95_upper"]) for r in rows]
    dec_mean = [float(r["decentralized"]["sr_end"]["mean"]) for r in rows]
    dec_lo = [float(r["decentralized"]["sr_end"]["ci95_lower"]) for r in rows]
    dec_hi = [float(r["decentralized"]["sr_end"]["ci95_upper"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(Bs, cen_mean, "o-", lw=2.0, label="Centralized")
    ax.fill_between(Bs, cen_lo, cen_hi, alpha=0.18, label="Centralized 95% CI")
    ax.plot(Bs, dec_mean, "s-", lw=2.0, label="Decentralized (RFQ)")
    ax.fill_between(Bs, dec_lo, dec_hi, alpha=0.18, label="RFQ 95% CI")
    ax.set_xlabel("Loan-size cap B")
    ax.set_ylabel(r"Final-horizon $SR_T$")
    ax.set_title("Loan-cap robustness: final systemic risk")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    ax = axes[1]
    delta_mean = [
        float(r["paired"]["final_difference"]["mean"])
        for r in rows
    ]
    delta_lo = [
        float(r["paired"]["final_difference"]["ci95_lower"])
        for r in rows
    ]
    delta_hi = [
        float(r["paired"]["final_difference"]["ci95_upper"])
        for r in rows
    ]
    ax.plot(Bs, delta_mean, "D-", lw=2.0, color="purple", label="Mean paired ΔSR_T")
    ax.fill_between(Bs, delta_lo, delta_hi, alpha=0.2, color="purple", label="95% CI")
    ax.axhline(0.0, lw=1.2, ls="--", color="black")
    ax.set_xlabel("Loan-size cap B")
    ax.set_ylabel(r"$\Delta SR_T$ (RFQ − Centralized)")
    ax.set_title("Paired final-risk difference vs B")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def run_loan_cap_robustness(
    *,
    python_exe: str,
    T: int = DEFAULT_SIM_T_CAP,
    nsim: int = 20,
    den_matcher_mode: str = "load",
    cen_matcher_mode: str = "off",
    skip_run: bool = False,
    B_values: tuple[float, ...] = B_VALUES,
    out_json: Path | None = None,
    out_png: Path | None = None,
) -> dict:
    """
    Sweep loan-size cap B for both models under the baseline feature setting
    (rollover on, policy support on), then export JSON + figure.
    """
    out_json = out_json or (COMPARE_DIR / "loan_cap_robustness.json")
    out_png = out_png or (COMPARE_DIR / "loan_cap_robustness.png")

    rows: list[dict] = []
    for B_test in B_values:
        B_test = float(B_test)
        print(f"\n=== Loan-cap robustness: B={B_test:g} ===")

        if not skip_run:
            for model_key, (_, script) in MODEL_SPECS.items():
                mode = den_matcher_mode if model_key == "decentralized" else cen_matcher_mode
                fig_dir = _loan_cap_fig_dir(model_key, B_test, matcher_mode=mode)
                fig_dir.mkdir(parents=True, exist_ok=True)
                _run_model_script(
                    script,
                    fig_dir,
                    python_exe=python_exe,
                    T=int(T),
                    nsim=int(nsim),
                    B=B_test,
                    matcher_mode=mode,
                    rollover_enabled=True,
                    policy_support_enabled=True,
                    stop_alive_threshold=STOP_ALIVE_THRESHOLD,
                )

        dec_art = _load_artifacts(
            _loan_cap_fig_dir("decentralized", B_test, matcher_mode=den_matcher_mode)
        )
        cen_art = _load_artifacts(
            _loan_cap_fig_dir("centralized", B_test, matcher_mode=cen_matcher_mode)
        )

        try:
            paired = calculate_paired_sr_statistics(dec_art, cen_art)
        except ValueError as exc:
            print(f"[warn] paired SR stats unavailable for B={B_test:g}: {exc}")
            paired = {
                "n_pairs": 0,
                "final_difference": _summary_scalar([]),
                "rfq_higher_final_proportion": 0.0,
            }

        # Drop bulky path arrays from the exported robustness table.
        paired_clean = {
            k: v
            for k, v in paired.items()
            if k not in ("delta_paths", "common_seeds")
        }

        def _net_block(art: dict) -> dict:
            keys = (
                "active_links",
                "network_density",
                "mean_degree",
                "exposure_hhi",
                "largest_bilateral_exposure",
                "largest_component_size",
                "funding_satisfaction_ratio",
                "cumulative_transaction_volume",
            )
            out = {"sr_end": _artifact_sr_end_summary(art)}
            for key in keys:
                out[key] = _artifact_network_mean(art, key)
            out["n_runs"] = int(len(art.get("baseline_runs") or []))
            return out

        row = {
            "B": B_test,
            "T": int(T),
            "nsim": int(nsim),
            "features": {
                "rollover_enabled": True,
                "policy_support_enabled": True,
            },
            "centralized": _net_block(cen_art),
            "decentralized": _net_block(dec_art),
            "paired": paired_clean,
        }
        rows.append(row)

    payload = {
        "B_values": [float(b) for b in B_values],
        "T": int(T),
        "nsim": int(nsim),
        "results": rows,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    print(f"Saved: {out_json}")

    plot_loan_cap_robustness(rows, out_png)
    return payload


def main():
    parser = argparse.ArgumentParser(description="对比 decentralized 与 centralized 仿真结果图")
    parser.add_argument("--python", dest="python_exe", default=sys.executable,
                        help="用于运行仿真脚本的 Python（需已安装 torch）")
    parser.add_argument(
        "--T",
        type=int,
        default=DEFAULT_SIM_T_CAP,
        help="安全上限（默认 4000）；商业银行只剩 stop-alive-threshold 家时早停，存活时间用 collapse_step",
    )
    parser.add_argument("--nsim", type=int, default=20, help="plot batch 轨迹条数")
    parser.add_argument("--seed-root", type=int, default=42, help="命名种子流的根种子")
    parser.add_argument(
        "--seed-namespace",
        choices=["CALIBRATION", "FORMAL"],
        default="FORMAL",
        help="校准与正式实验使用互不重叠的确定性种子流",
    )
    parser.add_argument(
        "--run-scope", choices=["calibration", "formal"], default="formal",
        help="calibration 自动使用 T=1000、小样本并跳过 loan-cap sweep",
    )
    parser.add_argument("--calibration-nsim", type=int, default=4)
    parser.add_argument(
        "--stop-alive-threshold",
        type=int,
        default=STOP_ALIVE_THRESHOLD,
        help="剩余商业银行数早停阈值（主实验 4；稳健性用 3）",
    )
    parser.add_argument("--B", type=float, default=DEFAULT_MATCH_B, help="单笔成交上限 B（默认 1200）")
    parser.add_argument("--skip-run", action="store_true",
                        help="跳过仿真，仅从已有 compare_artifacts.json 生成对比图")
    parser.add_argument(
        "--den-matcher-mode",
        choices=["train", "load", "off"],
        default="load",
        help="DEN：load=局部 GNN-RFQ v6（默认）；train=训练 gnn_pair_matcher_v6_local；off=仅调试",
    )
    parser.add_argument(
        "--cen-matcher-mode",
        choices=["off"],
        default="off",
        help="CEN：必须为 off（全局利率撮合，禁止 GNN）",
    )
    parser.add_argument(
        "--allow-den-rate-only",
        action="store_true",
        help="调试：允许 DEN --den-matcher-mode off（正式论文结果禁止）",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="(兼容) 仅强制 CEN matcher-mode off；不再把 DEN 静默切到 rate-only",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="跑全部 4 种 rollover/support 组合；对比图写入 comparison/base_on_off/",
    )
    parser.add_argument(
        "--base-on-off",
        action="store_true",
        help="只跑 base 的 rollover/support on–off 四种对比（等价于 --all-scenarios --skip-loan-cap；结果写入 comparison/base_on_off/）",
    )
    parser.add_argument(
        "--skip-loan-cap",
        action="store_true",
        help="跳过 B∈{600,1200,1800} 的 loan-cap robustness sweep",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="续跑：跳过已完成的情景；一旦遇到未完成项，其后情景一律重跑（避免新旧结果混用）",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="强制重跑所有情景（覆盖 --resume）",
    )
    args = parser.parse_args()

    if args.run_scope == "calibration":
        args.T = 1000
        args.nsim = max(1, int(args.calibration_nsim))
        args.seed_namespace = "CALIBRATION"
        args.skip_loan_cap = True
    paired_seed0 = make_seed_stream(args.seed_namespace, args.seed_root, 1)[0]
    print(
        f"[seed-protocol] namespace={args.seed_namespace} root={args.seed_root} "
        f"paired_seed0={paired_seed0}"
    )

    if args.base_on_off:
        args.all_scenarios = True
        args.skip_loan_cap = True
        # Mid-run exits are common for long base-on-off suites.
        if not args.force_rerun:
            args.resume = True

    den_mode = str(args.den_matcher_mode).strip().lower()
    cen_mode = str(args.cen_matcher_mode).strip().lower()
    if cen_mode != "off":
        raise SystemExit(
            "CEN 禁止 GNN：--cen-matcher-mode 必须为 off（全局利率撮合）。"
        )
    if args.no_train:
        cen_mode = "off"
    if den_mode == "off" and not args.allow_den_rate_only:
        raise SystemExit(
            "正式 DEN 必须使用 --den-matcher-mode load|train；"
            "调试才可加 --allow-den-rate-only。"
        )
    if args.no_train:
        print("[warn] --no-train 仅影响 CEN；DEN 仍使用 --den-matcher-mode（默认 load）")

    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    DECENTRALIZED_FIG_DIR.mkdir(parents=True, exist_ok=True)
    CENTRALIZED_FIG_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = FEATURE_SCENARIOS if args.all_scenarios else ((True, True),)
    compare_out_dir = (
        BASE_ON_OFF_COMPARE_DIR if args.all_scenarios else COMPARE_DIR
    )
    if args.run_scope == "calibration":
        compare_out_dir = OUTPUT_ROOT / "calibration" / "comparison" / (
            "base_on_off" if args.all_scenarios else "baseline"
        )
    compare_out_dir.mkdir(parents=True, exist_ok=True)

    def _ckpt_hash_for_mode(mode: str) -> str | None:
        if str(mode).lower() in ("load", "train"):
            return checkpoint_sha256(GNN_PAIR_MATCHER_V6_LOCAL_PATH)
        return None

    if args.skip_run:
        # --skip-run must still reject stale v2/v3 / wrong-schema artifacts.
        for rollover_enabled, policy_support_enabled in scenarios:
            for model_key in MODEL_SPECS:
                mode = den_mode if model_key == "decentralized" else cen_mode
                fig_dir = _scenario_dir(
                    model_key,
                    rollover_enabled,
                    policy_support_enabled,
                    matcher_mode=mode,
                    run_scope=args.run_scope,
                )
                if not _artifacts_ready(
                    fig_dir,
                    model_key=model_key,
                    rollover_enabled=rollover_enabled,
                    policy_support_enabled=policy_support_enabled,
                    nsim=args.nsim,
                    T=args.T,
                    B=args.B,
                    stop_alive_threshold=args.stop_alive_threshold,
                    matcher_mode=mode,
                    matcher_checkpoint_sha256=_ckpt_hash_for_mode(mode),
                ):
                    raise SystemExit(
                        f"--skip-run 拒绝过期/不完整结果: {fig_dir / 'compare_artifacts.json'}。"
                        "请用不带 --skip-run 的命令重跑，正式 DEN 用 "
                        "--den-matcher-mode train --cen-matcher-mode off --force-rerun。"
                    )

    if not args.skip_run:
        if not _check_dependencies(args.python_exe):
            raise SystemExit(1)
        resume = bool(args.resume) and not bool(args.force_rerun)
        seen_incomplete = False
        for rollover_enabled, policy_support_enabled in scenarios:
            for model_key, (_, script) in MODEL_SPECS.items():
                mode = (
                    den_mode
                    if model_key == "decentralized"
                    else cen_mode
                )
                fig_dir = _scenario_dir(
                    model_key,
                    rollover_enabled,
                    policy_support_enabled,
                    matcher_mode=mode,
                    run_scope=args.run_scope,
                )
                suffix = _feature_suffix(
                    rollover_enabled, policy_support_enabled
                )
                ckpt_hash = _ckpt_hash_for_mode(mode)
                ready = _artifacts_ready(
                    fig_dir,
                    model_key=model_key,
                    rollover_enabled=rollover_enabled,
                    policy_support_enabled=policy_support_enabled,
                    nsim=args.nsim,
                    T=args.T,
                    B=args.B,
                    stop_alive_threshold=args.stop_alive_threshold,
                    matcher_mode=mode,
                    matcher_checkpoint_sha256=ckpt_hash,
                )
                if resume and ready and not seen_incomplete:
                    print(
                        f"[resume] skip {model_key} {suffix} matcher={mode} "
                        f"(artifacts ready: {fig_dir / 'compare_artifacts.json'})"
                    )
                    continue
                if resume and not ready:
                    seen_incomplete = True
                    print(
                        f"[resume] need {model_key} {suffix} matcher={mode} "
                        f"(incomplete/missing under {fig_dir})"
                    )
                elif resume and seen_incomplete:
                    print(
                        f"[resume] rerun {model_key} {suffix} matcher={mode} "
                        f"(after incomplete earlier scenario)"
                    )
                fig_dir.mkdir(parents=True, exist_ok=True)
                _run_model_script(
                    script,
                    fig_dir,
                    python_exe=args.python_exe,
                    T=args.T,
                    nsim=args.nsim,
                    B=args.B,
                    matcher_mode=mode,
                    rollover_enabled=rollover_enabled,
                    policy_support_enabled=policy_support_enabled,
                    stop_alive_threshold=args.stop_alive_threshold,
                    allow_den_rate_only=bool(args.allow_den_rate_only),
                    seed0=paired_seed0,
                )

    scenario_artifacts: dict[str, tuple[str, dict]] = {}
    for rollover_enabled, policy_support_enabled in scenarios:
        suffix = _feature_suffix(rollover_enabled, policy_support_enabled)
        dec_mode = den_mode
        cen_mode_sc = cen_mode
        dec_fig = _scenario_dir(
            "decentralized", rollover_enabled, policy_support_enabled,
            matcher_mode=dec_mode, run_scope=args.run_scope,
        )
        cen_fig = _scenario_dir(
            "centralized", rollover_enabled, policy_support_enabled,
            matcher_mode=cen_mode_sc, run_scope=args.run_scope,
        )
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
            compare_out_dir / f"compare_baseline_{suffix}.png",
        )
        plot_collapse_index_comparison(
            dec_art, cen_art,
            compare_out_dir / f"compare_collapse_index_{suffix}.png",
        )

        paired_stats = calculate_paired_sr_statistics(
            dec_art,
            cen_art,
        )

        save_paired_statistics(
            paired_stats,
            compare_out_dir
            / f"paired_statistics_{suffix}.json",
        )

        plot_paired_sr_difference(
            dec_art,
            cen_art,
            compare_out_dir
            / f"paired_sr_difference_{suffix}.png",
        )

        network_stats = (
            calculate_paired_network_statistics(
                dec_art,
                cen_art,
            )
        )
        shock_study = calculate_shock_event_study(dec_art, cen_art)
        network_stats["shock_event_study"] = shock_study
        network_stats["note"] = (
            "metrics.* are full-horizon means. Shock response is in "
            "shock_event_study and run0_shock_paths, not in those means."
        )
        network_stats["run0_shock_paths"] = {
            "rfq": _run0_shock_paths(dec_art),
            "centralized": _run0_shock_paths(cen_art),
        }

        save_network_statistics(
            network_stats,
            compare_out_dir
            / f"network_statistics_{suffix}.json",
        )
        save_network_statistics(
            shock_study,
            compare_out_dir
            / f"shock_event_study_{suffix}.json",
        )
        print_screening_diagnosis(dec_art, cen_art, network_stats)

        plot_extra_metrics_comparison(
            dec_art, cen_art,
            compare_out_dir / f"compare_extra_metrics_{suffix}.png",
        )
        plot_shock_event_study(
            shock_study,
            compare_out_dir / f"compare_shock_event_study_{suffix}.png",
        )
        plot_theta_sweep_comparison(
            dec_art, cen_art,
            compare_out_dir / f"compare_theta_sweep_sr_{suffix}.png",
        )
        plot_weight_sweep_comparison(
            dec_art, cen_art,
            compare_out_dir / f"compare_weight_sweep_sr_{suffix}.png",
        )
        plot_network_panel_comparison(
            dec_fig / "network_decentralized_panel.png",
            cen_fig / "network_centralized_panel.png",
            compare_out_dir / f"compare_network_panel_{suffix}.png",
        )

    plot_all_scenario_baselines(
        scenario_artifacts,
        compare_out_dir,
    )
    # Cross early-stop final-SR ranking is invalid; use common-window metrics instead.
    plot_all_scenario_theta_final_sr(
        scenario_artifacts,
        compare_out_dir,
    )
    save_common_window_path_comparison(
        scenario_artifacts,
        compare_out_dir / "common_window_path_comparison.json",
    )
    plot_common_window_metrics(
        scenario_artifacts,
        compare_out_dir,
    )
    plot_all_policy_support_comparisons(
        scenario_artifacts,
        compare_out_dir,
    )
    save_failure_arrival_statistics(
        scenario_artifacts,
        compare_out_dir / "failure_arrival_statistics.json",
    )
    save_survival_comparison(
        scenario_artifacts,
        compare_out_dir / "survival_collapse_time_comparison.json",
    )
    plot_failure_arrival_comparison(
        scenario_artifacts,
        compare_out_dir / "compare_failure_arrival_times.png",
    )

    if not args.skip_loan_cap:
        try:
            run_loan_cap_robustness(
                python_exe=args.python_exe,
                T=args.T,
                nsim=args.nsim,
                den_matcher_mode=den_mode,
                cen_matcher_mode=cen_mode,
                skip_run=args.skip_run,
            )
        except OSError as exc:
            print(
                f"[warn] loan-cap robustness skipped after I/O error: {exc}\n"
                "  主对比结果仍有效。可稍后加 --skip-loan-cap 跳过，"
                "或等 OneDrive 同步结束后重跑该 sweep。"
            )

    print(f"\n[done] 对比图目录: {compare_out_dir.resolve()}")
    if args.all_scenarios:
        print(f"[done] base on/off 专用目录: {BASE_ON_OFF_COMPARE_DIR.resolve()}")
    dec_primary = DECENTRALIZED_FIG_DIR
    cen_primary = CENTRALIZED_FIG_DIR
    print("[done] 单模型图目录（默认仅 rollover on + support on）:")
    for label, d in (("Decentralized", dec_primary), ("Centralized", cen_primary)):
        figs = _list_primary_model_figures(d)
        print(f"  {label}: {d.resolve()}")
        if not figs:
            print("    （未找到图；请先不加 --skip-run 跑仿真，或检查输出目录）")
        else:
            for p in figs:
                print(f"    - {p.name}")
    print(
        "\n提示：默认 DEN=局部 GNN-load（--den-matcher-mode load）、CEN=全局利率（off）。"
        "首次需 --den-matcher-mode train 生成 gnn_pair_matcher_v6_local.pth（勿用 v2/v3/v4/v5）。"
        "DZ 不进入主比较。base on/off 用 --base-on-off；强制全量重跑用 --force-rerun。"
    )


if __name__ == "__main__":
    main()
