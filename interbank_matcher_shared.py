# -*- coding: utf-8 -*-
"""Shared fixed GNN/matcher training artifacts for formal DEN.

v6 (normalized, competition-aware local RFQ):
  * RFQ choice labels with concentration / gross-exposure teacher
  * Pair decoder receives bilateral exposure ratio (not node-only)
  * Public CAR/LCR grades from lagged / pre-trade info

Do not reuse v2/v3/v4/v5 caches for formal paper runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "输入"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATASET_SCHEMA_VERSION = 6
MATCHER_SCHEMA_VERSION = 6
LABEL_DEFINITION_V6 = "rfq_systemic_safe_competition_score_v6"
LOCAL_GRAPH_RULE_V6 = (
    "rfq_time_borrower+hist+candidates_public_grades_pair_exposure_"
    "repay_feats_teacher_scores_all_negative"
)
RFQ_EVENT_SCHEMA_VERSION = 3
REPAY_FEATURE_DIM = 2
# Backward-compatible import names; values intentionally point at v6.
LABEL_DEFINITION_V5 = LABEL_DEFINITION_V6
LOCAL_GRAPH_RULE_V5 = LOCAL_GRAPH_RULE_V6
LABEL_DEFINITION_V4 = LABEL_DEFINITION_V6
LOCAL_GRAPH_RULE_V4 = LOCAL_GRAPH_RULE_V6

FEATURE_ORDER_15 = [
    "core_capital",
    "liquid_assets",
    "current_liabilities",
    "interbank_assets",
    "interbank_liabilities",
    "solvency_ratio",
    "is_active",
    "capital_adequacy_ratio",
    "liquidity_coverage_ratio",
    "leverage_ratio",
    "market_volatility",
    "loan_interest_rate",
    "investment_interest_rate",
    "risk_appetite",
    "outflow_rate",
]

# Raw private slots zeroed for non-observer nodes; 3/4/7/8 get public overlays.
# Index 9 stays zeroed — bilateral exposure is a pair feature for the decoder.
PRIVATE_FEATURE_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9)
PAIR_FEATURE_DIM = 1  # bilateral gross exposure / lender core ∈ [0,1]

NODE_FEATURE_NORMALIZATION = {
    "name": "matcher_node_v6",
    "version": 1,
    "feature_order": list(FEATURE_ORDER_15),
    "amount_reference": 10000.0,
    "car_reference": 1.5,
    "lcr_reference": 3.0,
    "leverage_reference": 10.0,
    "daily_rate_reference": 0.001,
    "output_clip": [0.0, 5.0],
}


def normalize_matcher_feature_vector(values) -> list[float]:
    """Normalize one raw 15D bank vector by feature semantics.

    Amount inputs are raw currency values. Public overlays are applied later
    and already live in [0,1].
    """
    import math

    row = [float(x or 0.0) for x in values]
    if len(row) != len(FEATURE_ORDER_15):
        raise ValueError(f"matcher feature dim {len(row)} != {len(FEATURE_ORDER_15)}")
    ref = float(NODE_FEATURE_NORMALIZATION["amount_reference"])
    for idx in (0, 1, 2, 3, 4):
        row[idx] = math.log1p(max(0.0, row[idx])) / math.log1p(ref)
    row[5] = max(0.0, min(row[5] / 1.5, 1.0))
    row[6] = 1.0 if row[6] >= 0.5 else 0.0
    row[7] = max(0.0, min(row[7] / 1.5, 1.0))
    row[8] = max(0.0, min(row[8] / 3.0, 1.0))
    row[9] = max(0.0, min(row[9] / 10.0, 1.0))
    row[10] = max(0.0, min(row[10], 1.0))
    for idx in (11, 12):
        row[idx] = max(-1.0, min(row[idx] / 0.001, 1.0))
    row[13] = max(0.0, min(row[13], 1.0))
    row[14] = max(0.0, min(row[14], 1.0))
    return row


def sanitize_matcher_node_tensor(x):
    """Identical final tensor sanitation for training and inference."""
    import torch

    return torch.clamp(
        torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0),
        -5.0,
        5.0,
    )

SHARED_DATA_FILE = MODEL_DIR / "bank_contagion_data_shared.json"
SHARED_LOCAL_DATA_FILE = MODEL_DIR / "bank_contagion_data_local_v6.json"
SHARED_LOCAL_DATA_FILE_V5 = MODEL_DIR / "bank_contagion_data_local_v5.json"
SHARED_LOCAL_DATA_FILE_V4 = MODEL_DIR / "bank_contagion_data_local_v4.json"
SHARED_LOCAL_DATA_FILE_V3 = MODEL_DIR / "bank_contagion_data_local_v3.json"

GNN_PAIR_MATCHER_V6_LOCAL_PATH = MODEL_DIR / "gnn_pair_matcher_v6_local.pth"
GNN_PAIR_MATCHER_V5_LOCAL_PATH = MODEL_DIR / "gnn_pair_matcher_v5_local.pth"
GNN_PAIR_MATCHER_V4_LOCAL_PATH = MODEL_DIR / "gnn_pair_matcher_v4_local.pth"
GNN_PAIR_MATCHER_V3_LOCAL_PATH = MODEL_DIR / "gnn_pair_matcher_v3_local.pth"
GNN_PAIR_MATCHER_V2_PATH = MODEL_DIR / "gnn_pair_matcher_v2.pth"
SHARED_MATCHER_PATH = GNN_PAIR_MATCHER_V6_LOCAL_PATH
SHARED_GNN_LSTM_PATH = MODEL_DIR / "gnn_lstm_model_shared.pth"
LEGACY_BANK_PAIR_MATCHER_PATH = MODEL_DIR / "bank_pair_matcher_shared.pth"

DATA_FILE_DECENTRALIZED = MODEL_DIR / "bank_contagion_data_decentralized.json"
DATA_FILE_CENTRALIZED = MODEL_DIR / "bank_contagion_data_centralized.json"

DEFAULT_MIN_SHARED_SAMPLES = 1000
DEN_GNN_DIR_TOKEN = "den_gnn_v6_local"

SEED_NAMESPACES = ("MATCHER_TRAIN", "MATCHER_VALID", "CALIBRATION", "FORMAL")


def make_seed_stream(namespace: str, root_seed: int, n: int) -> list[int]:
    """Stable disjoint seed stream; never uses process-randomized ``hash``."""
    import numpy as np

    ns = str(namespace).strip().upper()
    if ns not in SEED_NAMESPACES:
        raise ValueError(f"unknown seed namespace {namespace!r}; use {SEED_NAMESPACES}")
    digest = hashlib.sha256(ns.encode("ascii")).digest()
    ns_words = [int.from_bytes(digest[i:i + 4], "little") for i in range(0, 16, 4)]
    seq = np.random.SeedSequence([int(root_seed), *ns_words])
    rng = np.random.default_rng(seq)
    return [int(x) for x in rng.integers(1, 2**31 - 1, size=max(0, int(n)))]


def _file_sha256(path: Path | str | None, *, nhex: int = 16) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[: int(nhex)]


checkpoint_sha256 = _file_sha256


def load_json_dataset(path: Path | str) -> list:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        samples = data.get("samples")
        return samples if isinstance(samples, list) else []
    return []


def load_dataset_payload(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"schema_version": 0, "samples": data}
    return data if isinstance(data, dict) else {}


def save_json_dataset(
    path: Path | str,
    data: list,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": int(DATASET_SCHEMA_VERSION),
        "feature_order": list(FEATURE_ORDER_15),
        "label_definition": LABEL_DEFINITION_V6,
        "local_graph_rule": LOCAL_GRAPH_RULE_V6,
        "node_feature_normalization": dict(NODE_FEATURE_NORMALIZATION),
        "pair_feature_dim": int(PAIR_FEATURE_DIM),
        "rfq_event_schema_version": int(RFQ_EVENT_SCHEMA_VERSION),
        "repay_feature_dim": int(REPAY_FEATURE_DIM),
        "samples": data,
    }
    if meta:
        payload.update(meta)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def should_regenerate_dataset(
    path: Path | str,
    *,
    force: bool = False,
    min_samples: int = DEFAULT_MIN_SHARED_SAMPLES,
    require_schema: int = DATASET_SCHEMA_VERSION,
    require_label: str = LABEL_DEFINITION_V6,
) -> bool:
    if force:
        return True
    payload = load_dataset_payload(path)
    if not payload:
        return True
    schema = int(payload.get("schema_version") or 0)
    if schema < int(require_schema):
        return True
    if str(payload.get("label_definition") or "") != str(require_label):
        return True
    feat = payload.get("feature_order")
    if list(feat or []) != list(FEATURE_ORDER_15):
        return True
    if str(payload.get("local_graph_rule") or "") != LOCAL_GRAPH_RULE_V6:
        return True
    if payload.get("node_feature_normalization") != NODE_FEATURE_NORMALIZATION:
        return True
    if int(payload.get("pair_feature_dim") or 0) != int(PAIR_FEATURE_DIM):
        return True
    if int(payload.get("rfq_event_schema_version") or 0) < int(RFQ_EVENT_SCHEMA_VERSION):
        return True
    if int(payload.get("repay_feature_dim") or 0) != int(REPAY_FEATURE_DIM):
        return True
    samples = payload.get("samples")
    if not isinstance(samples, list):
        return True
    return len(samples) < int(min_samples)


def save_matcher_checkpoint(
    path: Path | str,
    state_dict: dict,
    *,
    data_path: Path | str | None = None,
    train_config: dict[str, Any] | None = None,
) -> None:
    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "matcher_schema_version": int(MATCHER_SCHEMA_VERSION),
        "state_dict": state_dict,
        "feature_order": list(FEATURE_ORDER_15),
        "label_definition": LABEL_DEFINITION_V6,
        "local_graph_rule": LOCAL_GRAPH_RULE_V6,
        "node_feature_normalization": dict(NODE_FEATURE_NORMALIZATION),
        "pair_feature_dim": int(PAIR_FEATURE_DIM),
        "rfq_event_schema_version": int(RFQ_EVENT_SCHEMA_VERSION),
        "repay_feature_dim": int(REPAY_FEATURE_DIM),
        "data_path": str(data_path) if data_path is not None else None,
        "data_sha256": _file_sha256(data_path),
        "train_config": dict(train_config or {}),
    }
    # File handles avoid old PyTorch Windows Unicode-path corruption.
    with open(p, "wb") as f:
        torch.save(blob, f)


def load_matcher_state_dict(path: Path | str, *, device=None):
    import torch

    p = Path(path)
    # File handles avoid old PyTorch Windows Unicode-path corruption.
    with open(p, "rb") as f:
        raw = torch.load(f, map_location=device or "cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        meta = {k: v for k, v in raw.items() if k != "state_dict"}
        schema = int(meta.get("matcher_schema_version") or 0)
        if schema < int(MATCHER_SCHEMA_VERSION):
            raise RuntimeError(
                f"Matcher checkpoint schema {schema} < {MATCHER_SCHEMA_VERSION}: {p}. "
                "Retrain with --den-matcher-mode train --force-rerun."
            )
        if list(meta.get("feature_order") or []) != list(FEATURE_ORDER_15):
            raise RuntimeError(f"Matcher feature_order mismatch: {p}")
        if int(meta.get("pair_feature_dim") or 0) != int(PAIR_FEATURE_DIM):
            raise RuntimeError(
                f"Matcher pair_feature_dim mismatch: "
                f"checkpoint={meta.get('pair_feature_dim')}, "
                f"required={PAIR_FEATURE_DIM}: {p}. "
                "Retrain with --den-matcher-mode train --force-rerun."
            )
        if str(meta.get("label_definition") or "") != LABEL_DEFINITION_V6:
            raise RuntimeError(f"Matcher label_definition mismatch: {p}")
        if str(meta.get("local_graph_rule") or "") != LOCAL_GRAPH_RULE_V6:
            raise RuntimeError(
                f"Matcher local_graph_rule mismatch (stale pre-v6 checkpoint): {p}. "
                "Retrain with --den-matcher-mode train."
            )
        if meta.get("node_feature_normalization") != NODE_FEATURE_NORMALIZATION:
            raise RuntimeError(
                f"Matcher node normalization mismatch: {p}. Retrain v6."
            )
        if int(meta.get("rfq_event_schema_version") or 0) < int(RFQ_EVENT_SCHEMA_VERSION):
            raise RuntimeError(
                f"Matcher rfq_event_schema_version too old: {p}. "
                "Retrain after regenerating bank_contagion_data_local_v6.json."
            )
        if int(meta.get("repay_feature_dim") or 0) != int(REPAY_FEATURE_DIM):
            raise RuntimeError(f"Matcher repay_feature_dim mismatch: {p}")
        return raw["state_dict"], meta
    raise RuntimeError(
        f"Refusing legacy bare matcher weights at {p}. "
        "Train gnn_pair_matcher_v6_local.pth (dict checkpoint) instead."
    )


def matcher_meta(
    *,
    matcher_mode: str,
    matcher=None,
    checkpoint: Path | str | None = None,
    mechanism: str = "decentralized",
) -> dict:
    mode = str(matcher_mode).strip().lower()
    ckpt = str(checkpoint) if checkpoint is not None else None
    if mechanism.startswith("cen"):
        matching_mode = "cen_off" if mode == "off" else f"cen_{mode}"
    elif mode == "off":
        matching_mode = "den_rate"
    else:
        matching_mode = DEN_GNN_DIR_TOKEN
    return {
        "matching_mode": matching_mode,
        "matcher_mode": mode,
        "matcher_class": (None if matcher is None else type(matcher).__name__),
        "matcher_checkpoint": ckpt,
        "matcher_checkpoint_sha256": checkpoint_sha256(checkpoint),
        "matcher_info_scope": "local_subgraph" if mode in ("train", "load") else "none",
        "matcher_schema_version": int(MATCHER_SCHEMA_VERSION) if mode in ("train", "load") else None,
        "dataset_schema_version": int(DATASET_SCHEMA_VERSION) if mode in ("train", "load") else None,
        "label_definition": LABEL_DEFINITION_V6 if mode in ("train", "load") else None,
        "node_feature_normalization": (
            dict(NODE_FEATURE_NORMALIZATION) if mode in ("train", "load") else None
        ),
        "pair_feature_dim": int(PAIR_FEATURE_DIM) if mode in ("train", "load") else None,
        "rfq_event_schema_version": int(RFQ_EVENT_SCHEMA_VERSION) if mode in ("train", "load") else None,
        "repay_feature_dim": int(REPAY_FEATURE_DIM) if mode in ("train", "load") else None,
    }
