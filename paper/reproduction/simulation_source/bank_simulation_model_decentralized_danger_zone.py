"""
Decentralized central-policy bank simulator with interbank danger-zone forbearance.

This file extends bank_simulation_model_decentralized_central_policy.py without
modifying the original script.  Core policy logic lives in interbank_danger_zone_policy.py.
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
from interbank_installment_rollover import effective_notional

_BASE_PATH = Path(__file__).resolve().parent / "bank_simulation_model_decentralized_central_policy.py"
_spec = importlib.util.spec_from_file_location("bank_sim_decentralized_base", _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)


@dataclass
class Contract(base.Contract):
    status: str = "active"
    suspended_at_step: int | None = None
    deferred_count: int = 0


base.Contract = Contract


def aggregate_contracts_to_exposure_matrix_at_step(book, n: int, current_step: int) -> np.ndarray:
    L = np.zeros((n, n), dtype=float)
    for c in book.contracts:
        if contract_counts_in_exposure(c, current_step):
            claim = effective_notional(c)
            L[c.lender_idx, c.borrower_idx] += claim
            L[c.borrower_idx, c.lender_idx] -= claim
    return L


def total_interbank_assets_liabilities_from_book(book, bank_idx: int, current_step: int) -> tuple[float, float]:
    assets = 0.0
    liabilities = 0.0
    for c in book.contracts:
        if not contract_counts_in_exposure(c, current_step):
            continue
        claim = effective_notional(c)
        if c.lender_idx == bank_idx:
            assets += claim
        if c.borrower_idx == bank_idx:
            liabilities += claim
    return assets, liabilities


base.aggregate_contracts_to_exposure_matrix_at_step = aggregate_contracts_to_exposure_matrix_at_step
base.total_interbank_assets_liabilities_from_book = total_interbank_assets_liabilities_from_book


class DangerZoneBankNetworkSimulator(base.BankNetworkSimulator):
    """BankNetworkSimulator with tipping-point suspension and exit restructuring."""

    def __init__(self, *args, danger_zone_config: DangerZoneConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.danger_zone_enabled = True
        self.danger_zone = DangerZoneManager(danger_zone_config or DangerZoneConfig())
        self.initial_state_export_prefix = "decentralized_danger_zone"
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
        self.initial_state_export_prefix = "decentralized_danger_zone"
        for b in self.banks:
            b["danger_zone"] = "normal"

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

    def _low_lcr_deleveraging_scale(self, bank: dict) -> float:
        lcr = self._bank_lcr_value(bank)
        threshold = float(getattr(self, "low_lcr_deleveraging_threshold", 0.85))
        if lcr >= threshold:
            return 1.0
        floor = float(getattr(self, "low_lcr_market_borrow_floor", 0.20))
        return float(np.clip(floor + (1.0 - floor) * (lcr / max(threshold, 1e-9)), floor, 1.0))

    def _delever_low_lcr_intentions(self, intentions: list) -> list:
        adjusted = []
        for it in intentions:
            bank = self.banks[it.bank_idx]
            if it.role == "borrower":
                scale = self._low_lcr_deleveraging_scale(bank)
            elif it.role == "lender":
                lcr = self._bank_lcr_value(bank)
                threshold = float(getattr(self, "low_lcr_lending_preserve_threshold", 1.05))
                floor = float(getattr(self, "low_lcr_lending_floor", 0.0))
                scale = 1.0 if lcr >= threshold else float(np.clip(lcr / max(threshold, 1e-9), floor, 1.0))
            else:
                scale = 1.0
            if scale < 1.0:
                qty = float(it.quantity) * scale
                if qty <= 1e-6:
                    continue
                adjusted.append(type(it)(
                    bank_idx=it.bank_idx,
                    role=it.role,
                    reserve_bid=it.reserve_bid,
                    reserve_ask=it.reserve_ask,
                    quantity=qty,
                ))
            else:
                adjusted.append(it)
        return adjusted

    def _project_investment_scale(self, bank: dict) -> float:
        lcr = self._bank_lcr_value(bank)
        threshold = float(getattr(self, "low_lcr_deleveraging_threshold", 0.85))
        if lcr >= threshold:
            return 1.0
        floor = float(getattr(self, "low_lcr_project_investment_floor", 0.10))
        return float(np.clip(floor + (1.0 - floor) * (lcr / max(threshold, 1e-9)), floor, 1.0))

    def _liquidate_project_assets_for_cash(self, bank_idx: int, sale_amount: float) -> float:
        if sale_amount <= 1e-9 or bank_idx >= len(self.project_book):
            return 0.0
        remaining_sale = float(sale_amount)
        new_book = []
        sold = 0.0
        for loan in self.project_book[bank_idx]:
            principal = float(getattr(loan, "principal", 0.0))
            if principal <= 1e-9:
                continue
            if remaining_sale <= 1e-9:
                new_book.append(loan)
                continue
            cut = min(principal, remaining_sale)
            loan.principal = principal - cut
            sold += cut
            remaining_sale -= cut
            if loan.principal > 1e-9:
                new_book.append(loan)
        self.project_book[bank_idx] = new_book
        bank = self.banks[bank_idx]
        bank["investment"]["projects"]["amount"] = max(
            0.0,
            float(bank["investment"]["projects"].get("amount", 0.0)) - sold,
        )
        cash = sold * (1.0 - float(getattr(self, "project_liquidation_haircut", 0.03)))
        bank["liquid_assets"] = float(bank.get("liquid_assets", 0.0)) + cash
        return cash

    def _rebuild_low_lcr_liquidity_buffers(self, step: int) -> None:
        target_lcr = float(getattr(self, "liquidity_rebuild_target", 0.90))
        for i, bank in enumerate(self.banks):
            if i == 0 or not bank.get("is_active", True):
                continue
            if self._bank_is_flow_frozen(i):
                continue
            lcr = self._bank_lcr_value(bank)
            if lcr >= target_lcr:
                continue

            lia = float(bank.get("current_liabilities", 0.0))
            if lia <= 1e-9:
                continue
            stress = float(np.clip((target_lcr - lcr) / max(target_lcr, 1e-9), 0.0, 1.0))

            # Retained operating cashflow represents asset income not paid out while rebuilding liquidity.
            cashflow_rate = (
                float(getattr(self, "liquidity_rebuild_cashflow_bull", 0.0012))
                if self.market_environment == "bull"
                else float(getattr(self, "liquidity_rebuild_cashflow_bear", 0.0005))
            )
            retained_cash = cashflow_rate * lia * stress
            bank["liquid_assets"] = float(bank.get("liquid_assets", 0.0)) + retained_cash
            bank["core_capital"] = float(bank.get("core_capital", 0.0)) + 0.15 * retained_cash

            # Terming out short funding lowers the LCR denominator without creating free cash.
            termout = float(getattr(self, "short_liability_termout_rate", 0.006)) * lia * stress
            bank["current_liabilities"] = max(0.0, lia - termout)
            bank["termed_out_liabilities"] = float(bank.get("termed_out_liabilities", 0.0)) + termout

            projects_amt = float(bank["investment"]["projects"].get("amount", 0.0))
            if projects_amt > 1e-9:
                sale = float(getattr(self, "project_liquidation_rate", 0.010)) * projects_amt * stress
                self._liquidate_project_assets_for_cash(i, sale)

            bank["risk_appetite"] = float(bank.get("risk_appetite", 0.5)) * (1.0 - 0.10 * stress)

    def _settle_interbank_installment_period(self, step: int) -> list[int]:
        """Only suspend/defer before EN; never pre-pay cash outside the unified EN."""
        if getattr(self, "danger_zone_enabled", False):
            from interbank_installment_rollover import (
                installment_payment_due,
                single_payment_due,
            )
            self.danger_zone.accrue_suspended_interest(self.contract_book, step)
            due = [
                c for c in self.contract_book.contracts
                if installment_payment_due(c, step) or single_payment_due(c, step)
            ]
            if due:
                _, defer_due = self.danger_zone.split_due_contracts(due, self, step)
                self.danger_zone.defer_contracts(defer_due, step)
        return super()._settle_interbank_installment_period(step)

    def _register_trade_contracts(self, trades: list, step: int) -> None:
        filtered = [
            t for t in trades
            if not self._bank_is_flow_frozen(t.lender_idx)
            and not self._bank_is_flow_frozen(t.borrower_idx)
        ]
        return super()._register_trade_contracts(filtered, step)

    def _filter_intentions_for_matching(self, intentions):
        intentions = self._delever_low_lcr_intentions(intentions)
        return [
            it for it in intentions
            if not self._bank_is_flow_frozen(it.bank_idx)
        ]

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
        # Liquidity rebuild can use pre-refresh buffers; zone transitions wait for CAR/LCR refresh.
        self._rebuild_low_lcr_liquidity_buffers(step)

    def _post_regulatory_refresh_hook(self, step: int) -> None:
        if getattr(self, "danger_zone_enabled", False):
            self.danger_zone.update_bank_states(self, step)

    def _history_extra_fields(self) -> dict:
        return {"danger_zone": self.danger_zone.export_summary_rows()}

    def maybe_save_network_snapshot(self, step, risk, tag="rfq", **kw):
        return super().maybe_save_network_snapshot(step, risk, tag="danger_zone", **kw)

    def simulate_step(self, step):
        return super().simulate_step(step)


BankNetworkSimulator = DangerZoneBankNetworkSimulator


def export_danger_zone_logs(sim: DangerZoneBankNetworkSimulator, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or (base.OUTPUT_DIR / "danger_zone_logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "decentralized_danger_zone_summary.csv"
    events_path = output_dir / "decentralized_danger_zone_events.csv"
    base.pd.DataFrame(sim.danger_zone.export_summary_rows()).to_csv(summary_path, index=False, encoding="utf-8-sig")
    base.pd.DataFrame(sim.danger_zone.event_log).to_csv(events_path, index=False, encoding="utf-8-sig")
    print(f"Saved danger-zone logs: {summary_path}")
    print(f"Saved danger-zone logs: {events_path}")
    return summary_path


def run_danger_zone_demo(T: int = 200, seed: int | None = None) -> DangerZoneBankNetworkSimulator:
    sim = DangerZoneBankNetworkSimulator(max_steps=T, seed=seed)
    sim.export_policy_logs = False
    sim._save_network_snapshot = False
    sim.initialize_network()
    for step in range(T):
        sim.simulate_step(step)
        if sim.network_stable_step is not None:
            break
    export_danger_zone_logs(sim)
    return sim


# Re-export commonly used symbols from the base module for plotting / training helpers.
BankContagionDataset = base.BankContagionDataset
train_model = base.train_model
train_matcher_from_dataset = base.train_matcher_from_dataset
plot_baseline_trajectory = base.plot_baseline_trajectory
plot_scenario_comparison = base.plot_scenario_comparison
run_sensitivity_analysis = base.run_sensitivity_analysis
plot_weight_sweep_lines = base.plot_weight_sweep_lines
plot_theta_measure_sweep_lines = base.plot_theta_measure_sweep_lines
plot_theta_policy_scenario_lines = base.plot_theta_policy_scenario_lines
generate_gnn_panel = base.generate_gnn_panel
run_and_report = base.run_and_report
measure_single_run_time = base.measure_single_run_time
DEFAULT_RANDOM_SEED = base.DEFAULT_RANDOM_SEED
FIG_DIR = base.FIG_DIR
OUTPUT_DIR = base.OUTPUT_DIR


if __name__ == "__main__":
    import time

    t_all = time.perf_counter()
    sim = run_danger_zone_demo(T=200, seed=DEFAULT_RANDOM_SEED)
    if sim.network_stable_step is not None:
        print(f"[SUMMARY] NETWORK_STABLE at step={sim.network_stable_step}")
    suspended = [r for r in sim.danger_zone.export_summary_rows() if r["zone"] == "suspended"]
    restructured = sum(r["restructure_count"] for r in sim.danger_zone.export_summary_rows())
    print(f"[SUMMARY] banks currently suspended: {len(suspended)}")
    print(f"[SUMMARY] total restructures on exit: {restructured}")
    print(f"[time] TOTAL: {time.perf_counter() - t_all:.2f}s")
