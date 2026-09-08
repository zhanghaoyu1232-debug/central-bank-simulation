"""
Centralized clearing simulator with interbank danger-zone forbearance.

Extends bank_simulation_model_centralized_central_policy.py without modifying it.
"""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from interbank_danger_zone_policy import (
    DangerZoneConfig,
    DangerZoneManager,
    contract_counts_in_exposure,
)
from interbank_installment_rollover import (
    bullet_maturity_due,
    effective_notional,
    installment_coupon_due,
    outstanding_principal,
)

_BASE_PATH = Path(__file__).resolve().parent / "bank_simulation_model_centralized_central_policy.py"
_spec = importlib.util.spec_from_file_location("bank_sim_centralized_base", _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)


@dataclass
class Contract(base.Contract):
    status: str = "active"
    suspended_at_step: int | None = None
    deferred_count: int = 0

    def cashflow_at_maturity(self) -> tuple[float, float]:
        principal = outstanding_principal(self)
        arrears = max(0.0, float(getattr(self, "arrears_due", 0.0) or 0.0))
        rate = float(self.settlement_rate or self.rate)
        return (principal * rate, principal + arrears)


base.Contract = Contract


def aggregate_contracts_to_exposure_matrix_at_step(book, n: int, current_step: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=float)
    for contract in book.contracts:
        if not contract_counts_in_exposure(contract, current_step):
            continue
        principal = effective_notional(contract)
        matrix[contract.lender_idx, contract.borrower_idx] += principal
        matrix[contract.borrower_idx, contract.lender_idx] -= principal
    return matrix


base.aggregate_contracts_to_exposure_matrix_at_step = aggregate_contracts_to_exposure_matrix_at_step


class DangerZoneBankNetworkSimulator(base.BankNetworkSimulator):
    """Centralized BankNetworkSimulator with tipping-point suspension and exit restructuring."""

    def __init__(self, *args, danger_zone_config: DangerZoneConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.danger_zone_enabled = True
        self.danger_zone = DangerZoneManager(danger_zone_config or DangerZoneConfig())
        self.initial_state_export_prefix = "centralized_danger_zone"
        self.low_lcr_deleveraging_threshold = 0.85
        self.low_lcr_market_borrow_floor = 0.20
        self.low_lcr_lending_preserve_threshold = 1.05
        self.low_lcr_lending_floor = 0.0
        self.low_lcr_project_investment_floor = 0.10
        self.policy_lcr_repair_target = 0.90
        self.policy_lcr_repair_cap_share = 0.35
        self.liquidity_rebuild_target = 0.90
        self.liquidity_rebuild_cashflow_bull = 0.0012
        self.liquidity_rebuild_cashflow_bear = 0.0005
        self.short_liability_termout_rate = 0.006
        self.project_liquidation_rate = 0.010
        self.project_liquidation_haircut = 0.03
        self.low_lcr_bear_shock_floor = -0.00015

    def initialize_network(self):
        super().initialize_network()
        self.danger_zone.reset(self.num_banks)
        self.initial_state_export_prefix = "centralized_danger_zone"
        for bank in self.banks:
            bank["danger_zone"] = "normal"

    def _bank_is_flow_frozen(self, bank_idx: int) -> bool:
        if not getattr(self, "danger_zone_enabled", False):
            return False
        if bank_idx >= len(self.banks) or not self.banks[bank_idx].get("is_active", True):
            return True
        return self.danger_zone.is_flow_blocked(bank_idx)

    def _bank_lcr_value(self, bank: dict) -> float:
        try:
            return float(bank.get("liquidity_coverage_ratio", 1.0))
        except (TypeError, ValueError):
            return 1.0

    def _project_investment_scale(self, bank: dict) -> float:
        lcr = self._bank_lcr_value(bank)
        threshold = float(getattr(self, "low_lcr_deleveraging_threshold", 0.85))
        if lcr >= threshold:
            return 1.0
        floor = float(getattr(self, "low_lcr_project_investment_floor", 0.10))
        return float(np.clip(floor + (1.0 - floor) * (lcr / max(threshold, 1e-9)), floor, 1.0))

    def _defer_due_for_frozen_banks(self, step: int) -> None:
        defer_list = []
        for contract in self.contract_book.contracts:
            due_now = installment_coupon_due(contract, step) or bullet_maturity_due(contract, step)
            if not due_now:
                continue
            if (
                self.danger_zone.is_flow_blocked(contract.lender_idx)
                or self.danger_zone.is_flow_blocked(contract.borrower_idx)
                or getattr(contract, "status", "active") == "suspended"
            ):
                defer_list.append(contract)
        if defer_list:
            self.danger_zone.defer_contracts(defer_list, step)

    def _settle_due_interbank_contracts(self, step: int) -> list[int]:
        if getattr(self, "danger_zone_enabled", False):
            self.danger_zone.accrue_suspended_interest(self.contract_book, step)
            self._defer_due_for_frozen_banks(step)
        return super()._settle_due_interbank_contracts(step)

    def _sparse_bipartite_update(self, roles: np.ndarray) -> None:
        if getattr(self, "danger_zone_enabled", False):
            roles = np.array(roles, copy=True)
            for bank_idx in range(len(roles)):
                if self._bank_is_flow_frozen(bank_idx):
                    roles[bank_idx] = 0
        super()._sparse_bipartite_update(roles)

    def _rebuild_low_lcr_liquidity_buffers(self, step: int) -> None:
        target_lcr = float(getattr(self, "liquidity_rebuild_target", 0.90))
        for bank_idx, bank in enumerate(self.banks):
            if bank_idx == 0 or not bank.get("is_active", True):
                continue
            if self._bank_is_flow_frozen(bank_idx):
                continue
            lcr = self._bank_lcr_value(bank)
            if lcr >= target_lcr:
                continue
            liabilities = float(bank.get("current_liabilities", 0.0))
            if liabilities <= 1e-9:
                continue
            stress = float(np.clip((target_lcr - lcr) / max(target_lcr, 1e-9), 0.0, 1.0))
            cashflow_rate = (
                float(getattr(self, "liquidity_rebuild_cashflow_bull", 0.0012))
                if self.market_environment == "bull"
                else float(getattr(self, "liquidity_rebuild_cashflow_bear", 0.0005))
            )
            retained_cash = cashflow_rate * liabilities * stress
            bank["liquid_assets"] = float(bank.get("liquid_assets", 0.0)) + retained_cash
            bank["core_capital"] = float(bank.get("core_capital", 0.0)) + 0.15 * retained_cash
            termout = float(getattr(self, "short_liability_termout_rate", 0.006)) * liabilities * stress
            bank["current_liabilities"] = max(0.0, liabilities - termout)
            bank["termed_out_liabilities"] = float(bank.get("termed_out_liabilities", 0.0)) + termout
            bank["risk_appetite"] = float(bank.get("risk_appetite", 0.5)) * (1.0 - 0.10 * stress)

    def _adjust_market_liquidity_shock(self, bank_idx: int, market_adjustment: float) -> float:
        if self._bank_is_flow_frozen(bank_idx):
            return 0.0
        bank = self.banks[bank_idx]
        adj = float(market_adjustment)
        if self.market_environment == "bear" and self._bank_lcr_value(bank) < self.liquidity_rebuild_target:
            adj = max(adj, float(getattr(self, "low_lcr_bear_shock_floor", -0.00015)))
        return adj

    def _lender_invest_frac(self, bank_idx: int, base_frac: float = 0.05) -> float:
        if self._bank_is_flow_frozen(bank_idx):
            return 0.0
        return float(base_frac) * self._project_investment_scale(self.banks[bank_idx])

    def allocate_borrowed_to_projects(self, i, *args, **kwargs):
        if self._bank_is_flow_frozen(i):
            self.borrowed_cash[i] = 0.0
            if getattr(self, "borrowed_origination_risk_sum", None) is not None:
                self.borrowed_origination_risk_sum[i] = 0.0
            return None
        return super().allocate_borrowed_to_projects(i, *args, **kwargs)

    def _post_project_phase_hook(self, step: int) -> None:
        self._rebuild_low_lcr_liquidity_buffers(step)

    def _post_regulatory_refresh_hook(self, step: int) -> None:
        if getattr(self, "danger_zone_enabled", False):
            self.danger_zone.update_bank_states(self, step)

    def _history_extra_fields(self) -> dict:
        return {"danger_zone": self.danger_zone.export_summary_rows()}

    def maybe_save_network_snapshot(self, step, risk, tag="centralized", **kw):
        return super().maybe_save_network_snapshot(step, risk, tag="centralized_danger_zone", **kw)

    def simulate_step(self, step):
        return super().simulate_step(step)


BankNetworkSimulator = DangerZoneBankNetworkSimulator

DEFAULT_RANDOM_SEED = base.DEFAULT_RANDOM_SEED
FIG_DIR = base.FIG_DIR
OUTPUT_DIR = base.OUTPUT_DIR
