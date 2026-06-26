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
            L[c.lender_idx, c.borrower_idx] += c.principal
            L[c.borrower_idx, c.lender_idx] -= c.principal
    return L


def total_interbank_assets_liabilities_from_book(book, bank_idx: int, current_step: int) -> tuple[float, float]:
    assets = 0.0
    liabilities = 0.0
    for c in book.contracts:
        if not contract_counts_in_exposure(c, current_step):
            continue
        if c.lender_idx == bank_idx:
            assets += c.principal
        if c.borrower_idx == bank_idx:
            liabilities += c.principal
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

    def _repair_low_lcr_with_policy_support(self, step: int) -> None:
        if not getattr(self, "central_bank_support_enabled", True):
            return
        corridor = getattr(self, "central_corridor", None) or base.CentralBankCorridor(
            deposit_rate=max(0.0, self.base_rate - base.DAILY_CB_DEPOSIT_SPREAD),
            lending_rate=self.base_rate + base.DAILY_FACILITY_SPREAD_DEFENSIVE,
            base_rate=self.base_rate,
        )
        target_lcr = float(getattr(self, "policy_lcr_repair_target", 0.90))
        cap_share = float(getattr(self, "policy_lcr_repair_cap_share", 0.35))
        for i, bank in enumerate(self.banks):
            if i == 0 or not bank.get("is_active", True):
                continue
            if self._bank_is_flow_frozen(i):
                continue
            if self._bank_lcr_value(bank) >= target_lcr:
                continue
            if self._bank_equity(bank) <= 0.0:
                continue
            if float(bank.get("capital_adequacy_ratio", 0.0)) < float(getattr(self, "policy_car_floor", 0.03)):
                continue
            lia = float(bank.get("current_liabilities", 0.0))
            liq = float(bank.get("liquid_assets", 0.0))
            outflow_rate = float(bank.get("outflow_rate", 0.4))
            if lia <= 1e-9 or outflow_rate <= 1e-9:
                continue
            # Policy support is a loan, so it raises both liquid assets and liabilities.
            denom = max(1e-6, 1.0 - target_lcr * outflow_rate)
            amount = max(0.0, (target_lcr * outflow_rate * lia - liq) / denom)
            amount = min(amount, cap_share * lia)
            if amount <= 1e-6:
                continue
            injected = self._issue_central_bank_liquidity_support(
                i,
                amount,
                step,
                rate=corridor.lending_rate,
                tenor=max(2, int(getattr(self, "cb_loan_tenor", 2))),
                kind="lcr_repair_window",
            )
            if injected > 0.0:
                bank["risk_appetite"] = float(bank.get("risk_appetite", 0.5)) * 0.85

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

    def simulate_step(self, step):
        try:
            step = int(step)
            self.current_step = step
            self._reset_policy_step_budget()
            self._run_central_bank_policy_cycle(step)
            self._settle_central_bank_loans(step)
            n = self.num_banks
            book = self.contract_book
            banks = self.banks
            dz = self.danger_zone

            if self.exposure_matrix is not None:
                self.prev_exposure_matrix = self.exposure_matrix.copy()

            if getattr(self, "danger_zone_enabled", False):
                dz.accrue_suspended_interest(book, step)

            due = book.contracts_due_at(step)
            failed = []
            if due:
                if getattr(self, "danger_zone_enabled", False):
                    settle_due, defer_due = dz.split_due_contracts(due, self, step)
                    dz.defer_contracts(defer_due, step)
                else:
                    settle_due, defer_due = due, []

                if settle_due:
                    L_due = np.zeros((n, n), dtype=float)
                    for c in settle_due:
                        total = c.principal * (1.0 + c.rate)
                        L_due[c.lender_idx, c.borrower_idx] += total
                        L_due[c.borrower_idx, c.lender_idx] -= total
                    shortfall = base.liquidity_default_candidates(L_due, banks, n, use_core=False)
                    corridor = getattr(self, "central_corridor", None) or base.CentralBankCorridor(
                        deposit_rate=self.base_rate - 0.01,
                        lending_rate=self.base_rate + 0.02,
                        base_rate=self.base_rate,
                    )
                    for i in range(n):
                        if i == 0 or not shortfall[i] or self._bank_is_flow_frozen(i):
                            continue
                        p_bar_i = np.maximum(-L_due[i], 0.0).sum()
                        need = max(0.0, p_bar_i - float(banks[i].get("liquid_assets", 0.0)))
                        if need > 1e-6:
                            amt = min(need, 2000.0)
                            self._issue_central_bank_liquidity_support(
                                i,
                                amt,
                                step,
                                rate=corridor.lending_rate,
                                tenor=1,
                                kind="settlement_backstop",
                            )
                    _, failed = base.run_en_clearing_and_recovery(L_due, banks, n, use_core=False)
                    for c in settle_due:
                        book.remove_contract(c)

            for i in failed:
                if 0 < i < len(banks) and not self._bank_is_flow_frozen(i):
                    banks[i]["is_active"] = False
                    banks[i]["liquid_assets"] *= 0.8

            if hasattr(self, "adjust_base_rate"):
                self.adjust_base_rate()

            for _b in banks:
                if "pending_endowment" in _b:
                    _b["liquid_assets"] += _b.pop("pending_endowment")

            self.market_duration += 1
            if self.market_duration >= self.market_duration_limit:
                self.prev_market_environment = self.market_environment
                self.market_environment = "bull" if base.random.random() < 0.6 else "bear"
                self.market_duration = 0
                self.market_duration_limit = base.random.randint(2, 5)

            if self.market_environment == "bull":
                self.base_rate = max(base.DAILY_POLICY_RATE_FLOOR, self.base_rate + base.random.uniform(-0.00001, 0.00001))
                self.long_term_rate = self.base_rate + base.random.uniform(*base.DAILY_LONG_RATE_SPREAD_BULL)
                market_volatility = base.random.uniform(10, 20) / 50
                market_adjustment = base.random.uniform(*base.DAILY_MARKET_ADJUSTMENT_BULL)
            else:
                self.base_rate = min(base.DAILY_POLICY_RATE_CEILING, self.base_rate + base.random.uniform(0.0, 0.000015))
                self.long_term_rate = self.base_rate + base.random.uniform(*base.DAILY_LONG_RATE_SPREAD_BEAR)
                market_volatility = base.random.uniform(30, 50) / 50
                market_adjustment = base.random.uniform(*base.DAILY_MARKET_ADJUSTMENT_BEAR)

            for i, bank in enumerate(banks):
                bank["market_volatility"] = market_volatility
                bank["loan_interest_rate"] = (
                    self.base_rate + base.random.uniform(*base.DAILY_LOAN_SPREAD_BULL)
                    if self.market_environment == "bull"
                    else self.base_rate + base.random.uniform(*base.DAILY_LOAN_SPREAD_BEAR)
                )
                bank["investment_interest_rate"] = self.long_term_rate + base.random.uniform(*base.DAILY_INVESTMENT_SPREAD_STEP)
                bank.setdefault("risk_appetite", 0.5)
                bank.setdefault("hurdle_rate", base.DAILY_HURDLE_RATE)
                bank.setdefault("pending_endowment", 0.0)
                bank["current_liabilities"] += base.DAILY_LIABILITY_GROWTH * bank["current_liabilities"]
                if not self._bank_is_flow_frozen(i):
                    bank_adjustment = market_adjustment
                    if self.market_environment == "bear" and self._bank_lcr_value(bank) < self.liquidity_rebuild_target:
                        bank_adjustment = max(
                            bank_adjustment,
                            float(getattr(self, "low_lcr_bear_shock_floor", -0.00015)),
                        )
                    bank["liquid_assets"] *= (1 + bank_adjustment)
                bank["outflow_rate"] = (
                    0.2 if bank["type"] == "central"
                    else (
                        base.random.uniform(0.4, 0.6) if self.market_environment == "bear"
                        else base.random.uniform(0.3, 0.5)
                    )
                )
                if bank["liquid_assets"] < bank["current_liabilities"] * bank["outflow_rate"]:
                    bank["risk_appetite"] *= 0.9

            if not any(self._bank_is_flow_frozen(i) for i in range(n)):
                if base.random.random() < 0.05:
                    for i, bank in enumerate(banks):
                        bank["liquid_assets"] *= 0.9
                        for loan in self.project_book[i]:
                            loan.pd = min(1.0, loan.pd * 1.2)
            if self.market_environment == "bear" and base.random.random() < 0.1:
                for i, bank in enumerate(banks):
                    if not self._bank_is_flow_frozen(i):
                        bank["liquid_assets"] += 0.01 * bank["current_liabilities"]

            if (not getattr(self, "one_shot_default_done", False)) and (int(step) == 0):
                if base.random.random() < 0.05:
                    fail_bank = base.random.randint(1, n - 1)
                    banks[fail_bank]["is_active"] = False
                    banks[fail_bank]["liquid_assets"] *= 0.85
                self.one_shot_default_done = True

            if getattr(self, "free_market", False):
                self.roles = self.assign_roles_balanced(frac_lenders=0.5)
            else:
                self.roles = (
                    self.assign_roles_by_risk(
                        car_cutoff=getattr(self, "car_cutoff", 0.08),
                        lcr_cutoff=getattr(self, "lcr_cutoff", base.DAILY_INTERBANK_ROLE_LCR_CUTOFF),
                    )
                    if hasattr(self, "assign_roles_by_risk")
                    else self.assign_roles()
                )

            intentions = base.collect_intentions(
                banks, n, self.roles, self.reserve_buffer,
                self.base_rate,
                lcr_target=float(getattr(self, "interbank_lcr_target", base.DAILY_INTERBANK_INTENTION_LCR_TARGET)),
                last_avg_rate=getattr(self, "last_avg_rate", None),
            )
            intentions = self._delever_low_lcr_intentions(intentions)
            if getattr(self, "danger_zone_enabled", False):
                intentions = [
                    it for it in intentions
                    if not dz.is_flow_blocked(it.bank_idx)
                ]

            rfq = getattr(self, "rfq_market", None) or base.RFQMarket(max_rounds=3, min_trade_size=10.0)
            trades = rfq.run(intentions, banks, system=self, step=step, B_max=float(getattr(self, "B", 1200.0)))
            if trades:
                total_amt = sum(t.amount for t in trades)
                if total_amt > 1e-9:
                    self.last_avg_rate = sum(t.amount * t.rate for t in trades) / total_amt
                else:
                    self.last_avg_rate = sum(t.rate for t in trades) / len(trades)
            else:
                self.last_avg_rate = None

            maturity_periods = int(getattr(self, "interbank_contract_maturity", base.DAILY_INTERBANK_CONTRACT_MATURITY))
            for t in trades:
                if self._bank_is_flow_frozen(t.lender_idx) or self._bank_is_flow_frozen(t.borrower_idx):
                    continue
                book.add_from_trade(t, maturity_in_periods=maturity_periods)
                banks[t.lender_idx]["liquid_assets"] = float(banks[t.lender_idx].get("liquid_assets", 0.0)) - t.amount
                banks[t.borrower_idx]["liquid_assets"] = float(banks[t.borrower_idx].get("liquid_assets", 0.0)) + t.amount
                self.borrowed_cash[t.borrower_idx] += t.amount

            self.exposure_matrix = aggregate_contracts_to_exposure_matrix_at_step(book, n, step)
            np.fill_diagonal(self.exposure_matrix, 0.0)

            for i in range(n):
                if (
                    self.roles[i] == +1
                    and banks[i].get("is_active", True)
                    and not self._bank_is_flow_frozen(i)
                ):
                    invest_frac = 0.05 * self._project_investment_scale(banks[i])
                    if invest_frac > 1e-6:
                        self.invest_free_cash_into_projects(i, invest_frac=invest_frac)

            for i in range(n):
                if self._bank_is_flow_frozen(i):
                    continue
                if self._project_investment_scale(banks[i]) >= 0.50:
                    self.allocate_borrowed_to_projects(i)
                self.update_project_book(i)

            self._rebuild_low_lcr_liquidity_buffers(step)

            base.update_bank_states_from_contract_book(banks, book, n, step)
            for idx, b in enumerate(banks):
                cap_ratio = (b["core_capital"] + b["liquid_assets"]) / (b["current_liabilities"] + 1e-9)
                b["solvency_ratio"] = cap_ratio
                proj_amt = b["investment"]["projects"]["amount"]
                b["capital_adequacy_ratio"] = self._safe_car_value(
                    b["core_capital"], b["interbank_assets"], proj_amt
                )
                b["liquidity_coverage_ratio"] = b["liquid_assets"] / (
                    b["current_liabilities"] * b["outflow_rate"] + 1e-9
                )
                b["leverage_ratio"] = b["core_capital"] / (
                    b["liquid_assets"] + b["interbank_assets"] + 1e-9
                )
                b["capital_ratio_history"].append(cap_ratio)

            self._repair_low_lcr_with_policy_support(step)

            for idx, b in enumerate(banks):
                b["liquidity_coverage_ratio"] = b["liquid_assets"] / (
                    b["current_liabilities"] * b["outflow_rate"] + 1e-9
                )
                b["solvency_ratio"] = (b["core_capital"] + b["liquid_assets"]) / (
                    b["current_liabilities"] + 1e-9
                )

            if getattr(self, "danger_zone_enabled", False):
                dz.update_bank_states(self, step)

            for i in range(n):
                if self.bank_types[i] == "central":
                    continue
                if self._bank_is_flow_frozen(i):
                    continue
                b = banks[i]
                equity = self._bank_equity(b)
                if equity < 0:
                    b["is_active"] = False
                    b["liquid_assets"] *= 0.8

            risk = self.calculate_systemic_risk() if hasattr(self, "calculate_systemic_risk") else 0.0
            self._record_systemic_risk(risk)
            if getattr(self, "record_history", False):
                self.simulation_history.append({
                    "step": step,
                    "systemic_risk": risk,
                    "policy_note": getattr(self, "last_policy_note", ""),
                    "exposure_matrix": self.exposure_matrix.copy(),
                    "bank_states": [deepcopy(b) for b in banks],
                    "danger_zone": dz.export_summary_rows(),
                })
            if self.all_default_step is None:
                alive_noncentral = [k for k in range(1, n) if banks[k].get("is_active", True)]
                if len(alive_noncentral) == 0:
                    self.all_default_step = int(step)
                    print(f"[ALL DEFAULT] step={self.all_default_step} (all non-central banks defaulted)")
            self._update_network_stability(step, risk)
            self.maybe_save_network_snapshot(step, risk, tag="danger_zone", edge_quantile=0.0)
            return risk

        except Exception as e:
            print(f"Error in simulate_step: {e}")
            raise


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
