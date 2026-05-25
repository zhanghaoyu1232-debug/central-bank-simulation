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
                self.base_rate = max(0.01, self.base_rate + base.random.uniform(-0.005, 0.005))
                self.long_term_rate = self.base_rate + base.random.uniform(0.01, 0.02)
                market_volatility = base.random.uniform(10, 20) / 50
                market_adjustment = base.random.uniform(0.05, 0.1)
            else:
                self.base_rate = min(0.06, self.base_rate + base.random.uniform(0.0, 0.01))
                self.long_term_rate = self.base_rate + base.random.uniform(0.015, 0.025)
                market_volatility = base.random.uniform(30, 50) / 50
                market_adjustment = base.random.uniform(-0.15, -0.05)

            for i, bank in enumerate(banks):
                bank["market_volatility"] = market_volatility
                bank["loan_interest_rate"] = (
                    self.base_rate + base.random.uniform(0.01, 0.03)
                    if self.market_environment == "bull"
                    else self.base_rate + base.random.uniform(0.03, 0.05)
                )
                bank["investment_interest_rate"] = self.long_term_rate + base.random.uniform(0.005, 0.015)
                bank.setdefault("risk_appetite", 0.5)
                bank.setdefault("hurdle_rate", 0.04)
                bank.setdefault("pending_endowment", 0.0)
                bank["current_liabilities"] += 0.002 * bank["current_liabilities"]
                if not self._bank_is_flow_frozen(i):
                    bank["liquid_assets"] *= (1 + market_adjustment)
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
                    self.assign_roles_by_risk(car_cutoff=getattr(self, "car_cutoff", 0.08), lcr_cutoff=1.0)
                    if hasattr(self, "assign_roles_by_risk")
                    else self.assign_roles()
                )

            intentions = base.collect_intentions(
                banks, n, self.roles, self.reserve_buffer,
                self.base_rate, lcr_target=1.0,
                last_avg_rate=getattr(self, "last_avg_rate", None),
            )
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

            maturity_periods = int(getattr(self, "interbank_contract_maturity", 2))
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
                    self.invest_free_cash_into_projects(i, invest_frac=0.05)

            for i in range(n):
                if self._bank_is_flow_frozen(i):
                    continue
                self.allocate_borrowed_to_projects(i)
                self.update_project_book(i)

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
