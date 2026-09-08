"""仿真与对比图输出目录（固定在本仓库 git/输出/figures 下）。"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

OUTPUT_ROOT = REPO_ROOT / "输出" / "figures"
DECENTRALIZED_FIG_DIR = OUTPUT_ROOT / "decentralized"
CENTRALIZED_FIG_DIR = OUTPUT_ROOT / "centralized"
COMPARE_FIG_DIR = OUTPUT_ROOT / "comparison"
# Rollover/support on–off ablation comparison figures (kept separate from baseline).
BASE_ON_OFF_COMPARE_DIR = COMPARE_FIG_DIR / "base_on_off"

# baseline_runs keep native early-stop length (no pad to T_cap).
# v12: per-step common-liquidity-shock flag, mean LCR, shock_path_runs.
ARTIFACT_SCHEMA_VERSION = 13

_CODE_FINGERPRINT_FILES = (
    "bank_simulation_model_decentralized_central_policy.py",
    "bank_simulation_model_centralized_central_policy.py",
    "bank_regulatory.py",
    "bank_econ_shared.py",
    "interbank_installment_rollover.py",
    "interbank_intentions.py",
    "interbank_matcher_shared.py",
    "interbank_resolution.py",
    "project_common_random.py",
)


def simulation_code_fingerprint() -> str:
    """Short hash of core sim sources; resume must match or force rerun."""
    h = hashlib.sha256()
    for name in _CODE_FINGERPRINT_FILES:
        path = REPO_ROOT / name
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        if path.is_file():
            h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def model_figure_dir(model_key: str) -> Path:
    if model_key == "decentralized":
        return DECENTRALIZED_FIG_DIR
    if model_key == "centralized":
        return CENTRALIZED_FIG_DIR
    raise ValueError(f"unknown model_key: {model_key!r}")


def matcher_mode_dir_token(
    matcher_mode: str | None,
    *,
    model_key: str | None = None,
) -> str:
    """Directory token: DEN train/load → matcher_{mode}_den_gnn_v6_local."""
    mode = str(matcher_mode or "off").strip().lower()
    if mode not in ("train", "load", "off"):
        mode = "off"
    key = str(model_key or "").strip().lower()
    if key in ("decentralized", "den") and mode in ("train", "load"):
        return f"matcher_{mode}_den_gnn_v6_local"
    return f"matcher_{mode}"


def scenario_figure_dir(
    model_key: str,
    rollover_enabled: bool,
    policy_support_enabled: bool,
    matcher_mode: str = "off",
) -> Path:
    """单模型图目录：含 matcher 模式，避免 DEN-rate / DEN-GNN 互相覆盖。"""
    base = model_figure_dir(model_key) / matcher_mode_dir_token(
        matcher_mode, model_key=model_key
    )
    if rollover_enabled and policy_support_enabled:
        return base
    suffix = (
        f"{'rollover_on' if rollover_enabled else 'rollover_off'}_"
        f"{'support_on' if policy_support_enabled else 'support_off'}"
    )
    return base / suffix
