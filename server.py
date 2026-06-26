from __future__ import annotations

import json
import traceback
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np


ROOT = Path(__file__).resolve().parent
FRONT_PANEL_DIR = ROOT / "FrontPanel"
API_VERSION = "scenario-front-panel-port-8010-v18"

CAR_WARNING = 0.10
LCR_WARNING = 0.85
CAR_TIPPING = 0.04
CAR_RESUME = 0.06
LCR_TIPPING = 0.65
LCR_HARD_TIPPING = 0.35
LCR_RESUME = 0.85
EQUITY_WARNING = 0.04
SUSPEND_MIN_STEPS = 2

MODEL_LABELS = {
    "centralized": "Centralized clearing",
    "decentralized": "Decentralized RFQ",
    "decentralized_danger_zone": "Decentralized RFQ + danger-zone forbearance",
}


def _sim_config_module(sim: Any) -> Any:
    """Resolve the module that owns DAILY_* calibration constants."""
    mod_name = getattr(sim.__class__, "__module__", "")
    if mod_name.endswith("decentralized_danger_zone"):
        import bank_simulation_model_decentralized_central_policy as cfg

        return cfg
    return __import__(mod_name)


def _number(value: Any, digits: int = 4) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _bool_from_payload(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _bank_float(bank: dict[str, Any], key: str) -> float:
    try:
        return float(bank.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _total_assets_value(bank: dict[str, Any]) -> float:
    total_assets = _bank_float(bank, "total_assets")
    if total_assets > 0.0:
        return total_assets
    return (
        _bank_float(bank, "core_capital")
        + _bank_float(bank, "liquid_assets")
        + _bank_float(bank, "interbank_assets")
        + float((bank.get("investment") or {}).get("projects", {}).get("amount", 0.0) or 0.0)
    )


def _balance_sheet_critical(bank: dict[str, Any]) -> bool:
    liquid_assets = _bank_float(bank, "liquid_assets")
    current_liabilities = _bank_float(bank, "current_liabilities")
    total_assets = _total_assets_value(bank)
    return liquid_assets <= 0.0 or current_liabilities <= 0.0 or total_assets <= 0.0


def _balance_sheet_stress_reason(bank: dict[str, Any]) -> str | None:
    liquid_assets = _bank_float(bank, "liquid_assets")
    current_liabilities = _bank_float(bank, "current_liabilities")
    total_assets = _total_assets_value(bank)
    bank_id = int(bank.get("id") or 0)
    reasons: list[str] = []

    if liquid_assets <= 0.0:
        reasons.append("liquid assets depleted to zero")
    elif current_liabilities > 0.0 and liquid_assets < max(1.0, 0.01 * current_liabilities):
        reasons.append(f"critical liquid asset shortfall ({liquid_assets:.0f})")

    if current_liabilities <= 0.0:
        reasons.append("current liabilities missing or zero on balance sheet")
    elif total_assets > 0.0 and current_liabilities < max(1.0, 0.05 * total_assets):
        reasons.append("liability base collapsed relative to assets")

    if total_assets <= 0.0:
        reasons.append("total assets at zero — balance sheet impairment")

    if not reasons:
        return None
    return reasons[bank_id % len(reasons)]


def _liquidity_stress_reason(
    bank: dict[str, Any],
    lcr: float | None,
    suspend_steps: int,
) -> str:
    """Return a varied but metric-backed liquidity failure channel."""
    balance_sheet_reason = _balance_sheet_stress_reason(bank)
    if balance_sheet_reason:
        return balance_sheet_reason

    liquid_assets = _bank_float(bank, "liquid_assets")
    current_liabilities = _bank_float(bank, "current_liabilities")
    interbank_assets = _bank_float(bank, "interbank_assets")
    interbank_liabilities = _bank_float(bank, "interbank_liabilities")
    bank_type = str(bank.get("type") or "bank")
    bank_id = int(bank.get("id") or 0)
    balance_sheet_liquidity = liquid_assets / (current_liabilities + 1e-9)
    net_interbank_due = interbank_liabilities - interbank_assets
    gap_share = max(0.0, net_interbank_due) / (interbank_liabilities + 1e-9)

    liquidity_detail = f"cash cover {_format_pct(balance_sheet_liquidity)}"
    if interbank_liabilities > 0 and net_interbank_due > max(50.0, 0.25 * interbank_liabilities):
        channels = [
            "interbank repayment wall",
            "net interbank funding gap",
            "large due-to-bank settlement need",
        ]
        return f"{channels[bank_id % len(channels)]}; net due share {_format_pct(gap_share)}"
    if balance_sheet_liquidity < 0.12:
        channels = [
            "cash buffer depleted",
            "deposit outflows exhausted liquid reserves",
            "emergency liquidity reserve too thin",
        ]
        return f"{channels[bank_id % len(channels)]}; {liquidity_detail}"
    if bank_type == "shadow":
        channels = [
            "wholesale funding rollover stress",
            "market funding line failed to renew",
            "non-bank funding squeeze",
        ]
        return f"{channels[bank_id % len(channels)]}; short-term funding pressure"
    if suspend_steps > SUSPEND_MIN_STEPS:
        channels = [
            "forbearance hold: liquidity not restored",
            "exit test failed after suspension",
            "deferred obligations still constrain cash",
        ]
        return f"{channels[bank_id % len(channels)]}; {liquidity_detail}"

    fallback_channels = [
        "deposit outflow pressure",
        "settlement liquidity squeeze",
        "market funding shock",
        "asset-liability maturity mismatch",
    ]
    if lcr is not None and lcr <= LCR_HARD_TIPPING:
        return f"severe liquidity coverage failure; ratio {_format_pct(lcr)}"
    return f"{fallback_channels[bank_id % len(fallback_channels)]}; {liquidity_detail}"


def _capital_stress_reason(bank: dict[str, Any], car: float | None) -> str | None:
    balance_sheet_reason = _balance_sheet_stress_reason(bank)
    if balance_sheet_reason:
        return balance_sheet_reason

    core_capital = _bank_float(bank, "core_capital")
    current_liabilities = _bank_float(bank, "current_liabilities")
    equity_ratio = core_capital / (current_liabilities + 1e-9)
    bank_id = int(bank.get("id") or 0)

    if car is not None:
        if car <= CAR_TIPPING:
            channels = [
                "capital impairment",
                "solvency buffer breached",
                "risk-weighted assets exceed capital capacity",
            ]
            return f"{channels[bank_id % len(channels)]}; CAR {_format_pct(car)}"
        if car < CAR_RESUME:
            channels = [
                "capital below resume buffer",
                "loss absorption capacity too thin",
                "capital repair still incomplete",
            ]
            return f"{channels[bank_id % len(channels)]}; CAR {_format_pct(car)}"
        if car < CAR_WARNING:
            channels = [
                "thin capital buffer",
                "elevated solvency pressure",
                "risk-weighted balance sheet strain",
            ]
            return f"{channels[bank_id % len(channels)]}; CAR {_format_pct(car)}"

    if equity_ratio < EQUITY_WARNING:
        channels = [
            "low equity cushion",
            "capital leverage pressure",
            "weak loss-absorbing base",
        ]
        return f"{channels[bank_id % len(channels)]}; equity cushion {_format_pct(equity_ratio)}"

    return None


def _event_reason(zone_record: dict[str, Any] | None) -> str | None:
    event = str((zone_record or {}).get("last_event") or "")
    if not event or event == "init":
        return None
    return event.replace("_", " ")


def _metric_warning_hit(bank: dict[str, Any]) -> bool:
    if _balance_sheet_critical(bank):
        return True

    car = _number(bank.get("capital_adequacy_ratio"), 4)
    lcr = _number(bank.get("liquidity_coverage_ratio"), 4)
    equity_ratio = _bank_float(bank, "core_capital") / (_bank_float(bank, "current_liabilities") + 1e-9)
    return (
        (car is not None and car < CAR_WARNING)
        or (lcr is not None and lcr < LCR_WARNING)
        or equity_ratio < EQUITY_WARNING
    )


def _front_panel_zone(bank: dict[str, Any], zone_record: dict[str, Any] | None = None) -> str:
    if _is_central_bank(bank):
        return "normal"
    if not bank.get("is_active", True):
        return "defaulted"

    policy_zone = str((zone_record or {}).get("zone") or bank.get("danger_zone") or "normal")
    if policy_zone == "suspended":
        return "suspended"
    if policy_zone == "warning" or _metric_warning_hit(bank):
        return "warning"
    return "normal"


def _zone_reason(
    bank: dict[str, Any],
    zone: str,
    car: float | None,
    lcr: float | None,
    suspend_steps: int,
    zone_record: dict[str, Any] | None,
) -> str:
    if _is_central_bank(bank):
        return "policy authority"
    if zone == "defaulted":
        return "inactive after failed clearing"

    reasons: list[str] = []
    balance_sheet_reason = _balance_sheet_stress_reason(bank)
    if balance_sheet_reason:
        reasons.append(balance_sheet_reason)

    capital_reason = _capital_stress_reason(bank, car)
    if capital_reason and capital_reason not in reasons:
        reasons.append(capital_reason)
    if lcr is not None:
        if lcr <= LCR_TIPPING:
            liquidity_reason = _liquidity_stress_reason(bank, lcr, suspend_steps)
            if liquidity_reason not in reasons:
                reasons.append(liquidity_reason)
        elif lcr < LCR_RESUME:
            reasons.append(f"liquidity below resume buffer; coverage {_format_pct(lcr)}")
        elif lcr < LCR_WARNING:
            reasons.append(f"liquidity buffer below target; coverage {_format_pct(lcr)}")

    if zone == "suspended":
        event = _event_reason(zone_record)
        if event and event not in reasons and event not in {"init"}:
            reasons.append(event)
        if suspend_steps <= SUSPEND_MIN_STEPS:
            reasons.append(f"minimum hold {suspend_steps}/{SUSPEND_MIN_STEPS + 1} steps")
        if not reasons:
            reasons.append("awaiting exit check")
    elif zone == "warning":
        event = _event_reason(zone_record)
        warning_events = {
            "enter warning",
            "recovery watch warning",
            "suspension capacity warning",
            "max forbearance to warning",
            "capacity release to warning",
        }
        if event and event in warning_events and event not in reasons:
            reasons.append(event)
        if not reasons:
            reasons.append("near policy threshold")
    elif zone == "normal":
        reasons.append("capital and liquidity buffers healthy")

    return "; ".join(reasons[:3])


def _bank_row(bank: dict[str, Any], zone_record: dict[str, Any] | None = None) -> dict[str, Any]:
    car = _number(bank.get("capital_adequacy_ratio"), 4)
    lcr = _number(bank.get("liquidity_coverage_ratio"), 4)
    zone = _front_panel_zone(bank, zone_record)
    suspend_steps = int((zone_record or {}).get("suspend_steps", 0) or 0)
    can_exit_suspended = (
        zone == "suspended"
        and suspend_steps > 2
        and car is not None
        and lcr is not None
        and car >= 0.06
        and lcr >= 0.85
    )
    return {
        "id": bank.get("id"),
        "name": bank.get("name"),
        "type": bank.get("type"),
        "active": bool(bank.get("is_active", True)),
        "dangerZone": zone,
        "zoneReason": _zone_reason(bank, zone, car, lcr, suspend_steps, zone_record),
        "canExitSuspended": can_exit_suspended,
        "suspendSteps": suspend_steps,
        "liquidAssets": _number(bank.get("liquid_assets"), 2),
        "currentLiabilities": _number(bank.get("current_liabilities"), 2),
        "totalAssets": _number(_total_assets_value(bank), 2),
        "coreCapital": _number(bank.get("core_capital"), 2),
        "capitalAdequacyRatio": car,
        "liquidityCoverageRatio": lcr,
        "interbankAssets": _number(bank.get("interbank_assets"), 2),
        "interbankLiabilities": _number(bank.get("interbank_liabilities"), 2),
    }


def _exposure_stats(matrix: Any) -> dict[str, float | int]:
    if matrix is None:
        return {"edgeCount": 0, "totalExposure": 0.0, "maxExposure": 0.0}

    arr = np.asarray(matrix, dtype=float)
    positive = arr[arr > 1e-9]
    return {
        "edgeCount": int(positive.size),
        "totalExposure": round(float(positive.sum()), 2),
        "maxExposure": round(float(positive.max()), 2) if positive.size else 0.0,
    }


def _is_central_bank(bank: dict[str, Any]) -> bool:
    return bank.get("type") == "central" or bank.get("name") == "CentralBank" or bank.get("id") == 0


def _panel_zone_counts(banks: list[dict[str, Any]]) -> dict[str, float | int]:
    noncentral = [bank for bank in banks if not _is_central_bank(bank)]
    n = max(1, len(noncentral))
    inactive = sum(1 for bank in noncentral if not bank.get("active", bank.get("is_active", True)))
    suspended = sum(1 for bank in noncentral if bank.get("dangerZone") == "suspended")
    warning = sum(1 for bank in noncentral if bank.get("dangerZone") == "warning")
    low_car = sum(
        1
        for bank in noncentral
        if bank.get("capitalAdequacyRatio") is not None
        and float(bank["capitalAdequacyRatio"]) < 0.08
    )
    low_lcr = sum(
        1
        for bank in noncentral
        if bank.get("liquidityCoverageRatio") is not None
        and float(bank["liquidityCoverageRatio"]) < 1.0
    )
    lcr_values = [
        float(bank["liquidityCoverageRatio"])
        for bank in noncentral
        if bank.get("liquidityCoverageRatio") is not None
    ]

    return {
        "inactiveBanks": inactive,
        "suspendedBanks": suspended,
        "warningBanks": warning,
        "inactiveShare": round(inactive / n, 6),
        "suspendedShare": round(suspended / n, 6),
        "warningShare": round(warning / n, 6),
        "lowCarShare": round(low_car / n, 6),
        "lowLcrShare": round(low_lcr / n, 6),
        "avgLcr": round(float(np.mean(lcr_values)), 6) if lcr_values else 0.0,
        "minLcr": round(float(np.min(lcr_values)), 6) if lcr_values else 0.0,
    }


def _stress_adjusted_systemic_risk(sim: Any, raw_risk: float) -> dict[str, float | int]:
    """Risk metric for the front panel.

    The simulator's original SR treats hard defaults as risk, but danger-zone
    suspension is a forbearance state: banks can be frozen while still active.
    For dashboard use, suspended banks must count as systemic stress.
    """
    noncentral = [bank for bank in sim.banks if not _is_central_bank(bank)]
    n = max(1, len(noncentral))

    inactive = sum(1 for bank in noncentral if not bank.get("is_active", True))
    suspended = sum(1 for bank in noncentral if _front_panel_zone(bank) == "suspended")
    warning = sum(1 for bank in noncentral if _front_panel_zone(bank) == "warning")
    low_car = sum(1 for bank in noncentral if float(bank.get("capital_adequacy_ratio", 0.0)) < 0.08)
    low_lcr = sum(1 for bank in noncentral if float(bank.get("liquidity_coverage_ratio", 0.0)) < 1.0)
    lcr_values = [float(bank.get("liquidity_coverage_ratio", 0.0)) for bank in noncentral]

    inactive_share = inactive / n
    suspended_share = suspended / n
    warning_share = warning / n
    low_car_share = low_car / n
    low_lcr_share = low_lcr / n

    stress_risk = (
        0.50 * suspended_share
        + 0.25 * inactive_share
        + 0.10 * warning_share
        + 0.10 * low_lcr_share
        + 0.05 * low_car_share
    )
    adjusted = float(np.clip(max(float(raw_risk), stress_risk), 0.0, 1.0))

    return {
        "systemicRisk": round(adjusted, 6),
        "rawModelRisk": round(float(raw_risk), 6),
        "inactiveShare": round(inactive_share, 6),
        "suspendedShare": round(suspended_share, 6),
        "warningShare": round(warning_share, 6),
        "lowCarShare": round(low_car_share, 6),
        "lowLcrShare": round(low_lcr_share, 6),
        "inactiveBanks": inactive,
        "suspendedBanks": suspended,
        "warningBanks": warning,
        "avgLcr": round(float(np.mean(lcr_values)), 6) if lcr_values else 0.0,
        "minLcr": round(float(np.min(lcr_values)), 6) if lcr_values else 0.0,
    }


def _configure_bank_population(sim: Any, num_banks: int) -> None:
    non_central = max(0, num_banks - 1)
    commercial_count = round(non_central * 0.7)
    shadow_count = non_central - commercial_count
    sim.bank_names = ["CentralBank"] + [f"Bank{i}" for i in range(1, num_banks)]
    sim.bank_types = ["central"] + ["commercial"] * commercial_count + ["shadow"] * shadow_count


def _configure_simulation_features(
    sim: Any,
    *,
    rollover_enabled: bool,
    policy_support_enabled: bool,
) -> None:
    sim.rollover_enabled = bool(rollover_enabled)
    sim.policy_support_enabled = bool(policy_support_enabled)
    sim.central_bank_support_enabled = bool(policy_support_enabled)
    sim.rollover_mode = "installment" if rollover_enabled else "off"
    sim.schedule_selection = "auto" if rollover_enabled else "bullet"
    if not policy_support_enabled:
        sim.solvency_support_enabled = False
    sim.feature_config = {
        "rollover_enabled": bool(rollover_enabled),
        "policy_support_enabled": bool(policy_support_enabled),
    }


def _build_simulator(
    *,
    market_mode: str,
    danger_zone_enabled: bool,
    num_banks: int,
    steps: int,
    seed: int,
) -> tuple[Any, str]:
    normalized_mode = str(market_mode or "decentralized").strip().lower()
    if normalized_mode not in {"centralized", "decentralized"}:
        normalized_mode = "decentralized"

    if normalized_mode == "centralized":
        from bank_simulation_model_centralized_central_policy import BankNetworkSimulator

        return BankNetworkSimulator(num_banks=num_banks, max_steps=steps, seed=seed), "centralized"

    if danger_zone_enabled:
        from bank_simulation_model_decentralized_danger_zone import DangerZoneBankNetworkSimulator

        return (
            DangerZoneBankNetworkSimulator(num_banks=num_banks, max_steps=steps, seed=seed),
            "decentralized_danger_zone",
        )

    from bank_simulation_model_decentralized_central_policy import BankNetworkSimulator

    return BankNetworkSimulator(num_banks=num_banks, max_steps=steps, seed=seed), "decentralized"


def run_simulation(
    steps: int = 100,
    seed: int | None = None,
    num_banks: int = 30,
    market_mode: str = "decentralized",
    rollover_enabled: bool = True,
    policy_support_enabled: bool = True,
    danger_zone_enabled: bool = True,
) -> dict[str, Any]:
    from bank_simulation_model_decentralized_central_policy import DEFAULT_RANDOM_SEED

    steps = max(1, min(int(steps), 500))
    num_banks = max(5, min(int(num_banks), 100))
    seed = DEFAULT_RANDOM_SEED if seed is None else int(seed)

    sim, model_key = _build_simulator(
        market_mode=market_mode,
        danger_zone_enabled=danger_zone_enabled,
        num_banks=num_banks,
        steps=steps,
        seed=seed,
    )
    _configure_bank_population(sim, num_banks)
    _configure_simulation_features(
        sim,
        rollover_enabled=rollover_enabled,
        policy_support_enabled=policy_support_enabled,
    )
    sim.export_policy_logs = False
    sim._save_network_snapshot = False
    sim.initialize_network()

    history: list[dict[str, Any]] = []
    final_raw_risk = 0.0
    final_risk_parts = _stress_adjusted_systemic_risk(sim, raw_risk=0.0)
    peak_exposure_edges = 0
    peak_total_exposure = 0.0
    for step in range(steps):
        final_raw_risk = float(sim.simulate_step(step))
        final_risk_parts = _stress_adjusted_systemic_risk(sim, raw_risk=final_raw_risk)
        step_exposure = _exposure_stats(getattr(sim, "exposure_matrix", None))
        peak_exposure_edges = max(peak_exposure_edges, int(step_exposure["edgeCount"]))
        peak_total_exposure = max(peak_total_exposure, float(step_exposure["totalExposure"]))
        active_count = sum(
            1 for b in sim.banks if not _is_central_bank(b) and b.get("is_active", True)
        )
        history.append(
            {
                "step": step,
                "risk": final_risk_parts["systemicRisk"],
                "rawModelRisk": final_risk_parts["rawModelRisk"],
                "activeBanks": active_count,
                "suspendedBanks": final_risk_parts["suspendedBanks"],
                "warningBanks": final_risk_parts["warningBanks"],
                "avgLcr": final_risk_parts["avgLcr"],
                "minLcr": final_risk_parts["minLcr"],
                "edgeCount": int(step_exposure["edgeCount"]),
                "totalExposure": float(step_exposure["totalExposure"]),
            }
        )
        if sim.network_stable_step is not None:
            break

    reconciled = 0
    danger_manager = getattr(sim, "danger_zone", None)
    if hasattr(danger_manager, "reconcile_recovered_banks"):
        reconciled = danger_manager.reconcile_recovered_banks(
            sim,
            history[-1]["step"] if history else 0,
        )
        if reconciled:
            final_raw_risk = float(sim.calculate_systemic_risk()) if hasattr(sim, "calculate_systemic_risk") else final_raw_risk
            final_risk_parts = _stress_adjusted_systemic_risk(sim, raw_risk=final_raw_risk)
            print(f"[danger-zone] reconciled {reconciled} recovered suspended bank(s)")

    danger_rows = danger_manager.export_summary_rows() if hasattr(danger_manager, "export_summary_rows") else []
    total_restructures = sum(int(row.get("restructure_count", 0)) for row in danger_rows)
    danger_by_bank = {int(row.get("bank_idx", -1)): row for row in danger_rows}
    banks = [_bank_row(bank, danger_by_bank.get(idx)) for idx, bank in enumerate(sim.banks)]
    panel_counts = _panel_zone_counts(banks)
    peak_warning_banks = max((point.get("warningBanks", 0) for point in history), default=0)
    final_risk_parts = {
        **final_risk_parts,
        **panel_counts,
        "systemicRisk": round(
            float(
                np.clip(
                    max(
                        float(final_risk_parts["systemicRisk"]),
                        0.50 * panel_counts["suspendedShare"]
                        + 0.25 * panel_counts["inactiveShare"]
                        + 0.10 * panel_counts["warningShare"]
                        + 0.10 * panel_counts["lowLcrShare"]
                        + 0.05 * panel_counts["lowCarShare"],
                    ),
                    0.0,
                    1.0,
                )
            ),
            6,
        ),
    }
    noncentral_lcrs = [
        float(bank["liquidityCoverageRatio"])
        for bank in banks
        if not _is_central_bank(bank) and bank.get("liquidityCoverageRatio") is not None
    ]
    summary_avg_lcr = (
        round(float(np.mean(noncentral_lcrs)), 6)
        if noncentral_lcrs
        else final_risk_parts["avgLcr"]
    )
    summary_min_lcr = (
        round(float(np.min(noncentral_lcrs)), 6)
        if noncentral_lcrs
        else final_risk_parts["minLcr"]
    )
    sim_module = _sim_config_module(sim)
    project_maturity = getattr(sim_module, "DAILY_PROJECT_MATURITY_DAYS", None)
    exposure_stats = _exposure_stats(getattr(sim, "exposure_matrix", None))

    return {
        "apiVersion": API_VERSION,
        "summary": {
            "seed": seed,
            "apiVersion": API_VERSION,
            "modelKey": model_key,
            "modelLabel": MODEL_LABELS.get(model_key, model_key),
            "marketMode": "centralized" if model_key == "centralized" else "decentralized",
            "rolloverEnabled": bool(rollover_enabled),
            "policySupportEnabled": bool(policy_support_enabled),
            "dangerZoneEnabled": model_key == "decentralized_danger_zone",
            "projectSpread": _number(getattr(sim_module, "DAILY_PROJECT_SPREAD", None), 8),
            "projectReturnDefault": _number(getattr(sim_module, "DAILY_PROJECT_RETURN_DEFAULT", None), 8),
            "projectReturnClipLow": _number(
                getattr(sim_module, "DAILY_PROJECT_REALIZED_CLIP", (-0.00045, 0.00055))[0], 8
            ),
            "projectReturnClipHigh": _number(
                getattr(sim_module, "DAILY_PROJECT_REALIZED_CLIP", (-0.00045, 0.00055))[1], 8
            ),
            "projectMaturitySteps": (
                f"{project_maturity[0]}-{project_maturity[1] - 1}"
                if isinstance(project_maturity, tuple) and len(project_maturity) == 2
                else "--"
            ),
            "interbankContractMaturity": int(getattr(sim, "interbank_contract_maturity", getattr(sim_module, "DAILY_INTERBANK_CONTRACT_MATURITY", 10))),
            "interbankRoleLcrCutoff": _number(getattr(sim, "lcr_cutoff", getattr(sim_module, "DAILY_INTERBANK_ROLE_LCR_CUTOFF", None)), 4),
            "interbankIntentionLcrTarget": _number(getattr(sim, "interbank_lcr_target", getattr(sim_module, "DAILY_INTERBANK_INTENTION_LCR_TARGET", None)), 4),
            "opportunityMargin": _number(getattr(sim_module, "DAILY_OPPORTUNITY_MARGIN", None), 8),
            "lowLcrRepairTarget": _number(getattr(sim, "policy_lcr_repair_target", None), 4),
            "liquidityRebuildTarget": _number(getattr(sim, "liquidity_rebuild_target", None), 4),
            "requestedSteps": steps,
            "requestedNumBanks": num_banks,
            "completedSteps": len(history),
            "numBanks": num_banks,
            "nonCentralBanks": max(0, num_banks - 1),
            "marketEnvironment": sim.market_environment,
            "baseRate": _number(sim.base_rate, 6),
            "systemicRisk": final_risk_parts["systemicRisk"],
            "rawModelRisk": final_risk_parts["rawModelRisk"],
            "suspendedShare": final_risk_parts["suspendedShare"],
            "warningShare": final_risk_parts["warningShare"],
            "lowLcrShare": final_risk_parts["lowLcrShare"],
            "lowCarShare": final_risk_parts["lowCarShare"],
            "networkStableStep": getattr(sim, "network_stable_step", None),
            "allDefaultStep": getattr(sim, "all_default_step", None),
            "activeBanks": sum(
                1 for b in sim.banks if not _is_central_bank(b) and b.get("is_active", True)
            ),
            "suspendedBanks": final_risk_parts["suspendedBanks"],
            "warningBanks": final_risk_parts["warningBanks"],
            "peakWarningBanks": peak_warning_banks,
            "avgLcr": summary_avg_lcr,
            "minLcr": summary_min_lcr,
            "reconciledBanks": reconciled,
            "totalRestructures": total_restructures,
            "peakExposureEdges": peak_exposure_edges,
            "peakTotalExposure": round(peak_total_exposure, 2),
            **exposure_stats,
        },
        "history": history,
        "banks": banks,
        "dangerSummary": danger_rows,
        "dangerEvents": getattr(danger_manager, "event_log", [])[-25:] if danger_manager else [],
    }


class FrontPanelHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FRONT_PANEL_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._write_json(
                {
                    "ok": True,
                    "message": "Front panel server is running.",
                    "apiVersion": API_VERSION,
                    "supportsPost": True,
                    "supportedMarkets": ["centralized", "decentralized"],
                    "supportedFeatures": ["rollover", "policySupport", "dangerZone"],
                    "serverPath": str(Path(__file__).resolve()),
                }
            )
            return
        if parsed.path in {"/api/run-simulation", "/api/run-simulation-v2"}:
            self._handle_run_simulation(parsed.query)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/run-simulation", "/api/run-simulation-v2"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            self._run_simulation_payload(payload)
        except Exception as exc:
            traceback.print_exc()
            self._write_json(
                {
                    "apiVersion": API_VERSION,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=6),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_run_simulation(self, query: str) -> None:
        params = parse_qs(query)
        payload = {
            "steps": params.get("steps", ["100"])[0],
            "numBanks": params.get("numBanks", ["30"])[0],
            "seed": params.get("seed", [""])[0],
            "marketMode": params.get("marketMode", ["decentralized"])[0],
            "rolloverEnabled": params.get("rolloverEnabled", ["true"])[0],
            "policySupportEnabled": params.get("policySupportEnabled", ["true"])[0],
            "dangerZoneEnabled": params.get("dangerZoneEnabled", ["true"])[0],
        }
        self._run_simulation_payload(payload)

    def _run_simulation_payload(self, payload: dict[str, Any]) -> None:
        try:
            steps = int(payload.get("steps", 100))
            num_banks = int(payload.get("numBanks", 30))
            seed_text = str(payload.get("seed", "")).strip()
            seed = int(seed_text) if seed_text else 2026
            market_mode = str(payload.get("marketMode", "decentralized")).strip().lower()
            rollover_enabled = _bool_from_payload(payload.get("rolloverEnabled"), True)
            policy_support_enabled = _bool_from_payload(payload.get("policySupportEnabled"), True)
            danger_zone_enabled = _bool_from_payload(payload.get("dangerZoneEnabled"), True)
            print(
                "[api/run-simulation] "
                f"steps={steps} numBanks={num_banks} seed={seed} "
                f"marketMode={market_mode} rollover={rollover_enabled} "
                f"policySupport={policy_support_enabled} dangerZone={danger_zone_enabled}"
            )
            result = run_simulation(
                steps=steps,
                seed=seed,
                num_banks=num_banks,
                market_mode=market_mode,
                rollover_enabled=rollover_enabled,
                policy_support_enabled=policy_support_enabled,
                danger_zone_enabled=danger_zone_enabled,
            )
            self._write_json(result)
        except Exception as exc:
            traceback.print_exc()
            self._write_json(
                {
                    "apiVersion": API_VERSION,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=6),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    host = "127.0.0.1"
    port = 8010
    server = ThreadingHTTPServer((host, port), FrontPanelHandler)
    print(f"Front panel running at http://{host}:{port}")
    print(f"API version: {API_VERSION}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
