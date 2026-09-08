# ===== Runtime setup (paste from line 1) =====
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"networkx backend defined more than once",
    category=RuntimeWarning,
    module=r"networkx\.utils\.backends",
)

from pathlib import Path
from output_paths import (
    CENTRALIZED_FIG_DIR,
    ARTIFACT_SCHEMA_VERSION,
    simulation_code_fingerprint,
)

OUTPUT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = OUTPUT_ROOT / "输入"
OUTPUT_DIR = OUTPUT_ROOT / "输出"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = CENTRALIZED_FIG_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)
# 最多运行 4000 步；商业银行只剩 STOP_ALIVE_THRESHOLD 家时早停，以其 collapse_step 为存活时间。
DEFAULT_SIM_T_CAP = 4000
MAX_PLOT_STEPS = 4000
STOP_ALIVE_THRESHOLD = 1
MAX_NETWORK_SNAPSHOT_STEP = 200
DEFAULT_MATCH_B = 1200.0
# OneDrive can briefly lock diagnostic files while synchronizing.  Initial
# state exports must never abort a simulation or a parameter sweep.
INITIAL_STATE_EXPORT_LOCK_WARNED = False

import matplotlib
matplotlib.use("Agg")  # batch/save-only: never block on GUI windows
import matplotlib.pyplot as plt
plt.ioff()
plt.show = lambda *args, **kwargs: None  # 禁止弹窗；图只保存到文件夹
import matplotlib.figure
#matplotlib.figure.Figure.savefig = lambda *args, **kwargs: None

import time
import numpy as np
import random
import os
import json
import csv
import torch
import pandas as pd
from openpyxl import Workbook
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data   # type: ignore
from torch_geometric.nn import GCNConv  # 新增：真正的 GNN 卷积层
from torch.utils.data import DataLoader, Dataset, random_split
from scipy.optimize import linear_sum_assignment

# 可选依赖：没装也不报错
try:
    import mplcursors  # type: ignore
except Exception:
    mplcursors = None

import networkx as nx
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass
from PIL import Image

from interbank_installment_rollover import (
    ROLLOVER_BORROW_BLOCK_ALL,
    ROLLOVER_BORROW_COUPON_CLEARED,
    ROLLOVER_BORROW_PROJECT_ONLY,
    DEFAULT_DEBT_BURDEN_KAPPA,
    DEFAULT_IBL_CAP_ASSET_LAMBDA,
    SCHEDULE_INSTALLMENT,
    ScheduleConfig,
    choose_trade_schedule,
    compute_project_investment_borrow_cap,
    effective_notional,
    filter_borrowers_for_rollover_block,
    ibl_borrowing_room,
    rollover_borrow_quantity,
    schedule_config_from_mapping,
    settle_interbank_period,
)
from bank_regulatory import (
    MAX_CAR_RATIO as DAILY_MAX_CAR_RATIO,
    MIN_CAR_RWA_LIA_FRAC as DAILY_MIN_CAR_RWA_LIA_FRAC,
    regulatory_car,
    regulatory_rwa,
)
from bank_econ_shared import (
    apply_deposit_flow,
    bank_in_cbs_active,
    car_below_threshold,
    measure_theta_grid,
    measure_w1_grid,
)
from interbank_intentions import (
    DAILY_INTERBANK_INTENTION_LCR_TARGET,
    Intention,
    _intentions_to_arrays,
    collect_intentions,
    edge_jaccard,
    exposure_network_metrics,
    undirected_edge_set,
)
from interbank_resolution import settle_absorbing_default, settle_absorbing_defaults_batch

# One period is one business day: keep policy/interbank rates in daily units.
DAILY_POLICY_RATE_FLOOR = 0.00005
DAILY_BULL_BASE_RATE = 0.00010
DAILY_BEAR_BASE_RATE = 0.00020
DAILY_POLICY_RATE_CEILING = 0.00025
DAILY_LONG_RATE_SPREAD_BULL = (0.00003, 0.00007)
DAILY_LONG_RATE_SPREAD_BEAR = (0.00004, 0.00008)
DAILY_LOAN_SPREAD_BULL = (0.00003, 0.00008)
DAILY_LOAN_SPREAD_BEAR = (0.00006, 0.00012)
DAILY_INVESTMENT_RETURN_BULL = (-0.00006, 0.00030)
DAILY_INVESTMENT_RETURN_BEAR = (-0.00032, 0.00010)
DAILY_INVESTMENT_SPREAD_INIT = (0.00003, 0.00008)
DAILY_INVESTMENT_SPREAD_STEP = (0.00002, 0.00006)
DAILY_PROJECT_SPREAD = 0.00014
# 5bp/day let ON/ON heal: CBS fell to ~0.1 and FR stayed at 1–3 banks / 800 steps.
# 8bp/day is the middle: credit visible, without the t=300–400 mass default.
DAILY_PROJECT_PD_DEFAULT = 0.0008
DAILY_PROJECT_PD_RANGE = (0.0003, 0.0016)
DAILY_PROJECT_MATURITY_DAYS = (20, 61)  # rng.integers high is exclusive; yields 20-60 steps
DAILY_PROJECT_SHOCK_MEAN_BULL = -0.00002
DAILY_PROJECT_SHOCK_STD_BULL = 0.00040
DAILY_PROJECT_SHOCK_MEAN_BEAR = -0.00012
DAILY_PROJECT_SHOCK_STD_BEAR = 0.00050
DAILY_PROJECT_REALIZED_CLIP = (-0.0015, 0.0008)
DAILY_PROJECT_RETURN_DEFAULT = 0.00008
DAILY_PROJECT_RISK_DEFAULT = 0.00016
DAILY_INTERBANK_ROLE_LCR_CUTOFF = 0.85
DAILY_INTERBANK_CONTRACT_MATURITY = 1  # OFF single_payment: next-period P(1+r); ON uses 5–20 installment
DAILY_RFQ_QUOTE_SPREAD = 0.00005
DAILY_CB_DEPOSIT_SPREAD = 0.00002
DAILY_CB_LENDING_SPREAD = 0.00005
DAILY_CB_PENALTY_SPREAD = 0.00005
DAILY_SOLVENCY_SUPPORT_SPREAD = 0.00003
DAILY_POLICY_EASING_CRISIS = 0.00005
DAILY_POLICY_EASING_DEFENSIVE = 0.000025
DAILY_FACILITY_SPREAD_CRISIS = 0.00003
DAILY_FACILITY_SPREAD_DEFENSIVE = 0.00004
DAILY_FACILITY_SPREAD_HOLD = 0.00005
DAILY_POLICY_RATE_MAX_STEP_CHANGE = 0.000025
DAILY_POLICY_RATE_CHANGE_THRESHOLD = 0.000005
DAILY_ROLLOVER_SPREAD_SHORT = 0.0
DAILY_ROLLOVER_SPREAD_LONG = 0.0
DAILY_ROLLOVER_SPREAD = 0.0
DAILY_HURDLE_RATE = 0.00012
DAILY_PENALTY_RATE_CEILING = 0.00035
DAILY_LIABILITY_GROWTH = 0.00002
DAILY_MARKET_ADJUSTMENT_BULL = (0.0002, 0.0008)
DAILY_MARKET_ADJUSTMENT_BEAR = (-0.0010, -0.0002)
# Common shock raises same-period outflow pressure (LCR / funding need).
# Do not haircut cash: that is a solvency hit and syncs DEN/CEN defaults.
DAILY_COMMON_LIQUIDITY_SHOCK_PROB = 0.035
DAILY_COMMON_LIQUIDITY_OUTFLOW_MULTIPLIER = 1.25
DAILY_COMMON_PROJECT_PD_MULTIPLIER = 2.0
DAILY_COMMON_PROJECT_PD_STRESS_DAYS = 5
# DAILY_MAX_CAR_RATIO / DAILY_MIN_CAR_RWA_LIA_FRAC imported from bank_regulatory
DAILY_MAX_LCR_RATIO = 3.00
DAILY_MIN_LCR_OUTFLOW_RATE = 0.10

@dataclass
class ProjectLoan:
    principal: float
    rate: float           # 每期利率
    maturity: int         # 期数
    age: int = 0
    pd: float = DAILY_PROJECT_PD_DEFAULT      # 每工作日违约概率
    lgd: float = 0.4      # 违约损失率
    project_id: int = 0
    # Frozen borrower risk at origination; scales PD and shock vol each period.
    origination_risk: float = 0.0


@dataclass
class Trade:
    lender_idx: int
    borrower_idx: int
    amount: float
    rate: float
    step_executed: int


@dataclass
class Contract:
    contract_id: str
    lender_idx: int
    borrower_idx: int
    principal: float
    rate: float
    created_step: int
    maturity_step: int
    schedule_type: str = "single_payment"
    remaining_principal: float = 0.0
    coupon_rate: float = 0.0
    tenor_total: int = 1
    periods_paid: int = 0
    settlement_rate: float = 0.0
    is_rollover_residual: bool = False
    consecutive_misses: int = 0
    arrears_due: float = 0.0

    def __post_init__(self) -> None:
        from interbank_installment_rollover import normalize_schedule_type, SCHEDULE_INSTALLMENT
        self.schedule_type = normalize_schedule_type(self.schedule_type)
        if self.schedule_type == SCHEDULE_INSTALLMENT and self.remaining_principal <= 0.0:
            self.remaining_principal = float(self.principal)
        if self.coupon_rate <= 0.0:
            self.coupon_rate = float(self.rate)
        if self.settlement_rate <= 0.0:
            self.settlement_rate = float(self.rate)


class ContractBook:
    _next_id: int = 0

    def __init__(self):
        self.contracts: list[Contract] = []

    def _new_id(self) -> str:
        ContractBook._next_id += 1
        return f"c{ContractBook._next_id}"

    def add_contract(self, c: Contract) -> None:
        self.contracts.append(c)

    def add_from_trade(self, t: Trade, maturity_in_periods: int = 1) -> Contract:
        c = Contract(
            contract_id=self._new_id(),
            lender_idx=t.lender_idx,
            borrower_idx=t.borrower_idx,
            principal=float(t.amount),
            rate=float(t.rate),
            created_step=int(t.step_executed),
            maturity_step=int(t.step_executed) + max(1, int(maturity_in_periods)),
            settlement_rate=float(t.rate),
        )
        self.add_contract(c)
        return c

    def add_from_trade_with_schedule(
        self,
        t: Trade,
        borrower: dict,
        cfg: ScheduleConfig,
    ) -> tuple[Contract, dict]:
        dec = choose_trade_schedule(
            principal=float(t.amount),
            settlement_rate=float(t.rate),
            borrower=borrower,
            cfg=cfg,
        )
        step = int(t.step_executed)
        installment = dec["schedule_type"] == "installment"
        if installment:
            tenor = int(dec["tenor"])
            contract = Contract(
                contract_id=self._new_id(),
                lender_idx=int(t.lender_idx),
                borrower_idx=int(t.borrower_idx),
                principal=float(t.amount),
                remaining_principal=float(t.amount),
                rate=float(dec["settlement_rate"]),
                coupon_rate=float(dec["coupon_rate"]),
                schedule_type="installment",
                tenor_total=tenor,
                periods_paid=0,
                settlement_rate=float(dec["settlement_rate"]),
                created_step=step,
                maturity_step=step + tenor,
                arrears_due=0.0,
            )
        else:
            maturity = int(dec["maturity_in_periods"])
            contract = Contract(
                contract_id=self._new_id(),
                lender_idx=int(t.lender_idx),
                borrower_idx=int(t.borrower_idx),
                principal=float(t.amount),
                rate=float(dec["settlement_rate"]),
                schedule_type="single_payment",
                settlement_rate=float(dec["settlement_rate"]),
                created_step=step,
                maturity_step=step + maturity,
                arrears_due=0.0,
            )
        self.add_contract(contract)
        return contract, dec

    def contracts_due_at(self, step: int) -> list[Contract]:
        return [c for c in self.contracts if int(c.maturity_step) <= int(step)]

    def remove_contract(self, c: Contract) -> None:
        try:
            self.contracts.remove(c)
        except ValueError:
            pass


def configure_simulation_features(
    sim,
    *,
    rollover_enabled: bool = True,
    policy_support_enabled: bool = True,
    centralized_cycle_length: int | None = None,
    relationship_lending_enabled: bool = False,
):
    """统一设置实验开关，供模型脚本和 compare 脚本复用。"""
    sim.rollover_enabled = bool(rollover_enabled)
    sim.policy_support_enabled = bool(policy_support_enabled)
    # Support OFF closes rate/reserve policy AND all injections; ON enables all three.
    sim.policy_enabled = bool(policy_support_enabled)
    sim.central_bank_support_enabled = bool(policy_support_enabled)
    sim.schedule_selection = (
        "installment" if rollover_enabled else "single_payment"
    )
    sim.rollover_mode = (
        "installment" if rollover_enabled else "off"
    )
    # ON: entire notional is installment over 5–20 periods (no partial roll ratio).
    sim.rollover_min_tenor = 5
    sim.rollover_max_tenor = 20
    sim.soft_principal_deferral = True
    sim.allow_early_full_repay = False
    sim.interbank_lgd = float(getattr(sim, "interbank_lgd", 0.4))
    sim.pending_policy_support = []
    # Support on ⇒ enable solvency/capital support (disbursed with one-period lag).
    sim.solvency_support_enabled = bool(policy_support_enabled)
    if centralized_cycle_length is not None:
        sim.centralized_cycle_length = max(1, int(centralized_cycle_length))
    if hasattr(sim, "relationship_lending_enabled"):
        sim.relationship_lending_enabled = bool(relationship_lending_enabled)
    sim.feature_config = {
        "rollover_enabled": bool(rollover_enabled),
        "policy_support_enabled": bool(policy_support_enabled),
        "centralized_cycle_length": int(
            getattr(sim, "centralized_cycle_length", 1)
        ),
        "relationship_lending_enabled": bool(relationship_lending_enabled),
    }
    return sim


def aggregate_contracts_to_exposure_matrix_at_step(book: ContractBook, n: int, current_step: int) -> np.ndarray:
    L = np.zeros((n, n), dtype=float)
    for c in book.contracts:
        if int(c.maturity_step) > int(current_step):
            P = effective_notional(c)
            L[c.lender_idx, c.borrower_idx] += P
            L[c.borrower_idx, c.lender_idx] -= P
    return L


def aggregate_contracts_to_gross_exposure_matrix_at_step(
    book: ContractBook,
    n: int,
    current_step: int,
) -> np.ndarray:
    """
    Gross outstanding creditor-to-debtor exposure matrix.

    G[lender, borrower] records remaining principal.
    Reciprocal claims are retained separately.
    """
    G = np.zeros((n, n), dtype=float)

    for c in book.contracts:
        if int(c.maturity_step) <= int(current_step):
            continue

        principal = effective_notional(c)

        if principal <= 1e-12:
            continue

        G[
            int(c.lender_idx),
            int(c.borrower_idx),
        ] += float(principal)

    np.fill_diagonal(G, 0.0)

    return G


@dataclass
class CentralBankCorridor:
    """央行利率走廊：存款/贷款便利利率；0 号银行为央行。"""

    deposit_rate: float
    lending_rate: float
    base_rate: float

    def use_deposit_facility(self, bank_idx: int, amount: float, banks: list) -> None:
        if bank_idx == 0 or amount <= 0:
            return
        if bank_idx < len(banks):
            banks[bank_idx]["liquid_assets"] = float(banks[bank_idx].get("liquid_assets", 0.0)) - amount
        if len(banks) > 0:
            banks[0]["liquid_assets"] = float(banks[0].get("liquid_assets", 0.0)) + amount

    def use_lending_facility(self, bank_idx: int, amount: float, banks: list) -> None:
        if bank_idx == 0 or amount <= 0:
            return
        if len(banks) > 0:
            banks[0]["liquid_assets"] = float(banks[0].get("liquid_assets", 0.0)) - amount
        if bank_idx < len(banks):
            banks[bank_idx]["liquid_assets"] = float(banks[bank_idx].get("liquid_assets", 0.0)) + amount


def liquidity_default_candidates(
    liabilities: np.ndarray,
    banks: list,
    n: int,
    use_core: bool = False,
) -> np.ndarray:
    """
    liabilities[i, j] >= 0:
    amount currently due from debtor i to creditor j.
    """
    e = np.zeros(n, dtype=float)

    for i in range(min(n, len(banks))):
        b = banks[i]
        e[i] = (
            b["core_capital"] + b["liquid_assets"]
            if use_core
            else b["liquid_assets"]
        )

    Lbar = np.maximum(
        np.asarray(liabilities, dtype=float),
        0.0,
    )

    np.fill_diagonal(Lbar, 0.0)

    p_bar = Lbar.sum(axis=1)
    p_bar[p_bar < 1e-12] = 0.0

    shortfall = (p_bar > 0.0) & (e < p_bar * 0.999)

    return shortfall


def run_en_clearing_and_recovery(
    liabilities: np.ndarray,
    banks: list,
    n: int,
    use_core: bool = False,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, list[int]]:
    """
    Eisenberg-Noe clearing on gross due-payment liabilities.

    liabilities[i, j] >= 0 means that debtor i currently owes
    creditor j this amount. Reciprocal obligations are retained
    separately and are not bilaterally netted before clearing.
    """
    eps = 1e-9

    Lbar = np.maximum(
        np.asarray(liabilities, dtype=float),
        0.0,
    ).copy()

    np.fill_diagonal(Lbar, 0.0)

    p_bar = Lbar.sum(axis=1)

    if p_bar.sum() <= eps:
        return np.zeros(n), []

    Pi = np.divide(
        Lbar,
        p_bar[:, None],
        out=np.zeros_like(Lbar),
        where=(p_bar[:, None] > 0.0),
    )

    e = np.zeros(n, dtype=float)

    for i in range(min(n, len(banks))):
        b = banks[i]
        e[i] = (
            b["core_capital"] + b["liquid_assets"]
            if use_core
            else b["liquid_assets"]
        )

    p = p_bar.copy()

    for _ in range(max_iter):
        p_new = np.minimum(
            p_bar,
            Pi.T @ p + e,
        )

        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            break

        p = p_new

    failed = [
        i
        for i in range(n)
        if p[i] < p_bar[i] - tol
    ]

    recv = Pi.T @ p

    for i in range(min(n, len(banks))):
        banks[i]["liquid_assets"] = float(
            max(
                0.0,
                e[i] - p[i] + recv[i],
            )
        )

    return p, failed


def total_notional_by_bank_from_book(book: ContractBook, n: int, current_step: int) -> tuple[np.ndarray, np.ndarray]:
    assets = np.zeros(n, dtype=float)
    liabilities = np.zeros(n, dtype=float)
    for c in book.contracts:
        if int(c.maturity_step) <= int(current_step):
            continue
        P = effective_notional(c)
        assets[c.lender_idx] += P
        liabilities[c.borrower_idx] += P
    return assets, liabilities


def update_bank_states_from_contract_book(
    banks: list, book: ContractBook, n: int, current_step: int
) -> None:
    assets, liabilities = total_notional_by_bank_from_book(book, n, current_step)
    for i in range(min(n, len(banks))):
        # 吸收态/冻结墓碑：同业边已关闭，强制 IB 科目为 0，勿覆盖其余冻结科目
        if banks[i].get("absorbing_default") or banks[i].get("balance_sheet_frozen"):
            banks[i]["interbank_assets"] = 0.0
            banks[i]["interbank_liabilities"] = 0.0
            continue
        banks[i]["interbank_assets"] = float(assets[i])
        banks[i]["interbank_liabilities"] = float(liabilities[i])


def _is_central_bank(b) -> bool:
    return (b.get("type") == "central") or (b.get("name") == "CentralBank")


def _bank_regulatory_rwa(b) -> float:
    """Alias: all RWA paths go through bank_regulatory.regulatory_rwa."""
    return regulatory_rwa(b)


def _bank_measure_car(b) -> float:
    """Alias: all CAR recompute paths go through bank_regulatory.regulatory_car."""
    return regulatory_car(b)


def _bank_project_amount(b) -> float:
    return float(b.get("investment", {}).get("projects", {}).get("amount", 0.0))


def _bank_total_assets(b) -> float:
    """总资产 = 现金 + 同业资产 + 项目资产（不含权益/核心资本）。"""
    return (
        float(b.get("liquid_assets", 0.0))
        + float(b.get("interbank_assets", 0.0))
        + _bank_project_amount(b)
    )


def _bank_total_liabilities(b) -> float:
    """会计总负债 = 外部负债 + 同业负债 + 展期出的长期负债（termed_out）。

    LCR 短期流出分母不含 termed_out（见 ``_bank_lcr`` / ``_safe_lcr_value``）。
    """
    return (
        float(b.get("current_liabilities", 0.0))
        + float(b.get("interbank_liabilities", 0.0))
        + float(b.get("termed_out_liabilities", 0.0))
    )


def _bank_accounting_equity(b) -> float:
    """
    会计净资产：
    Equity = Liquid + InterbankAssets + ProjectAssets
           - (ExternalLiabilities + InterbankLiabilities + TermedOut)
    同业贷款本金同时增减资产/负债，不应改变净资产。
    短期负债转为 termed_out 也不应凭空提高权益。
    """
    return _bank_total_assets(b) - _bank_total_liabilities(b)


def _systemic_risk_from_banks(
    banks,
    weights=(0.5, 0.3, 0.2),
    car_threshold=0.08,
):
    """SR components aligned with the paper:

    FR  : share of failed (inactive) non-central banks — already defaulted.
    CBS : among active commercial banks, share with CAR < θ. Failed banks are
          recorded only in FR, so CBS remains an early-warning indicator rather
          than a second copy of realized failures.
    CGR : among *all* commercial banks, Σ capital gap / Σ required capital;
          failed banks keep gap at failure (actual capital = 0, required from
          ``required_capital_at_failure``).

    Aggregate (linear three-weight):
        SR = w1·FR + w2·CBS + w3·CGR
    with default weights (0.5, 0.3, 0.2) after non-negative renormalization.
    """
    car_threshold = float(car_threshold)
    noncentral = [b for b in banks if not _is_central_bank(b)]
    n_nc = max(1, len(noncentral))

    fr = sum(1 for b in noncentral if not b.get("is_active", True)) / n_nc

    active_noncentral = [b for b in noncentral if b.get("is_active", True)]
    low_capital = sum(
        1
        for bank in active_noncentral
        if bank_in_cbs_active(bank, car_threshold, car=_bank_measure_car(bank))
    )
    cbs = low_capital / float(max(1, len(active_noncentral)))

    gap_numerator = 0.0
    required_denominator = 0.0
    for bank in noncentral:
        if not bank.get("is_active", True):
            required = float(bank.get("required_capital_at_failure", 0.0))
            # 失败银行剩余有效资本按 0 处理
            actual = 0.0
        else:
            rwa = float(_bank_regulatory_rwa(bank))
            required = car_threshold * rwa
            actual = max(0.0, float(bank.get("core_capital", 0.0)))
        gap_numerator += max(0.0, required - actual)
        required_denominator += required
    cgr = gap_numerator / (required_denominator + 1e-9)

    w = np.asarray(weights, dtype=float)
    if w.size != 3:
        w = np.asarray([0.5, 0.3, 0.2], dtype=float)
    w = np.maximum(w, 0.0)
    weight_sum = float(w.sum())
    if weight_sum <= 1e-12:
        w = np.asarray([0.5, 0.3, 0.2], dtype=float)
    else:
        w = w / weight_sum
    w1, w2, w3 = float(w[0]), float(w[1]), float(w[2])
    sr = float(np.clip(w1 * fr + w2 * cbs + w3 * cgr, 0.0, 1.0))
    return sr, float(fr), float(cbs), float(cgr)


def decentralized_systemic_risk(
    banks: list,
    book: ContractBook,
    n: int,
    current_step: int,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    car_threshold: float = 0.08,
) -> float:
    """由 ContractBook 聚合敞口后，与主循环相同口径的 SR（用于校验）。"""
    _ = aggregate_contracts_to_exposure_matrix_at_step(book, n, current_step)
    sr, _, _, _ = _systemic_risk_from_banks(
        banks, weights=weights, car_threshold=float(car_threshold)
    )
    return sr


def validate_decentralized_vs_baseline(sr_baseline: float, sr_book: float, tol: float = 0.15) -> bool:
    return abs(sr_baseline - sr_book) <= tol


# ===== Node feature order (15D, no stocks/bonds) =====
FEATURE_ORDER_15 = [
    "core_capital",               # 0
    "liquid_assets",              # 1
    "current_liabilities",        # 2
    "interbank_assets",           # 3
    "interbank_liabilities",      # 4
    "solvency_ratio",             # 5
    "is_active",                  # 6 (0/1)
    "capital_adequacy_ratio",     # 7
    "liquidity_coverage_ratio",   # 8
    "leverage_ratio",             # 9
    "market_volatility",          #10
    "loan_interest_rate",         #11
    "investment_interest_rate",   #12
    "risk_appetite",              #13
    "outflow_rate"                #14
]


def _bank_to_feature_vec_15(b, env):
    """严格按 FEATURE_ORDER_15 抽取单个银行的15维特征。"""
    vals = {
        "core_capital":             float(b.get("core_capital", 0.0)) / 10000.0,
        "liquid_assets":            float(b.get("liquid_assets", 0.0)) / 10000.0,
        "current_liabilities":      float(b.get("current_liabilities", 0.0)) / 10000.0,
        "interbank_assets":         float(b.get("interbank_assets", 0.0)) / 10000.0,
        "interbank_liabilities":    float(b.get("interbank_liabilities", 0.0)) / 10000.0,

        # 比例类保持原样（或轻微 clip）
        "solvency_ratio":           float(np.clip(b.get("solvency_ratio", 0.0), 0.0, 5.0)),
        "is_active":                float(bool(b.get("is_active", True))),
        "capital_adequacy_ratio":   float(np.clip(b.get("capital_adequacy_ratio", 0.0), 0.0, 1.5)),
        "liquidity_coverage_ratio": float(np.clip(b.get("liquidity_coverage_ratio", 0.0), 0.0, 5.0)),
        "leverage_ratio":           float(np.clip(b.get("leverage_ratio", 0.0), 0.0, 5.0)),

        "market_volatility":        float(b.get("market_volatility", 0.0)),
        "loan_interest_rate":       float(b.get("loan_interest_rate", 0.0)),
        "investment_interest_rate": float(b.get("investment_interest_rate", 0.0)),
        "risk_appetite":            float(b.get("risk_appetite", 0.0)),
        "outflow_rate":             float(b.get("outflow_rate", 0.0)),
    }

    return [vals[k] for k in FEATURE_ORDER_15]


# 保留：项目根路径（可用作备用）
BASE_DIR = Path(__file__).resolve().parent

# —— 统一输入/输出目录 —— 
MODEL_DIR  = INPUT_DIR
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "gnn_lstm_model.pth"        # legacy local path
DATA_FILE  = MODEL_DIR / "bank_contagion_data_centralized.json"  # CEN scratch cache
from interbank_matcher_shared import (
    SHARED_DATA_FILE,
    SHARED_MATCHER_PATH,
    SHARED_GNN_LSTM_PATH,
    GNN_PAIR_MATCHER_V2_PATH,
    DATA_FILE_CENTRALIZED,
    load_json_dataset,
    save_json_dataset,
    should_regenerate_dataset,
    matcher_meta,
)
DATA_FILE = DATA_FILE_CENTRALIZED
INITIAL_STATE_DIR = OUTPUT_DIR / "initial_states"
INITIAL_STATE_DIR.mkdir(parents=True, exist_ok=True)
POLICY_LOG_DIR = OUTPUT_DIR / "policy_logs"
POLICY_LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_RANDOM_SEED = 42


def set_random_seed(seed: int = DEFAULT_RANDOM_SEED) -> int:
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


set_random_seed(DEFAULT_RANDOM_SEED)

plt.style.use('ggplot')
# plt.ion()

# 进程内均值轨迹缓存：避免不同出图函数重复仿真同一参数组合
_MEAN_COMPONENTS_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


class RegulatoryAdvisor:
    def __init__(self, banks, exposure_matrix, systemic_risk):
        self.banks = banks
        self.exposure_matrix = exposure_matrix
        self.systemic_risk = systemic_risk
    
    def generate_recommendations(self):
        recommendations = []
        high_risk_banks = []
        for i, bank in enumerate(self.banks):
            if bank['is_active'] and bank['solvency_ratio'] < 1.0:
                high_risk_banks.append((i, bank['name'], bank['solvency_ratio']))
        high_risk_banks.sort(key=lambda x: x[2])
        
        if self.systemic_risk > 0.7:
            recommendations.append("系统性风险高，建议采取紧急措施：")
            for i, name, solvency in high_risk_banks[:3]:
                recommendations.append(
                    f"- 向 {name} 注入资本 {self.banks[i]['current_liabilities'] * 0.2:.2f} 以提高偿付能力"
                )
            recommendations.append("- 提高所有银行的最低资本充足率要求至 10%")
        elif self.systemic_risk > 0.4:
            recommendations.append("系统性风险中等，建议加强监控：")
            for i, name, solvency in high_risk_banks[:2]:
                recommendations.append(f"- 限制 {name} 的高风险投资，降低其风险偏好")
            recommendations.append("- 要求影子银行增加流动性储备")
        else:
            recommendations.append("系统性风险低，建议维持现状：")
            recommendations.append("- 继续监控市场波动和银行间债务")
        
        high_exposure = []
        for i in range(len(self.banks)):
            for j in range(len(self.banks)):
                if self.exposure_matrix[i, j] > 300:
                    high_exposure.append(
                        (self.banks[i]['name'], self.banks[j]['name'], self.exposure_matrix[i, j])
                    )
        if high_exposure:
            recommendations.append("- 高暴露债务关系：")
            for src, dst, amt in high_exposure[:2]:
                recommendations.append(
                    f"  - {src} 对 {dst} 的债务 {amt:.2f}，建议降低债务集中度"
                )
        
        return recommendations


def export_initial_bank_table(banks, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, bank in enumerate(banks):
        liab = bank.get("liabilities_breakdown", {})
        proj = bank.get("investment", {}).get("projects", {})
        rows.append({
            "bank_id": idx,
            "name": bank.get("name"),
            "type": bank.get("type"),
            "is_active": bank.get("is_active"),
            "core_capital": float(bank.get("core_capital", 0.0)),
            "liquid_assets": float(bank.get("liquid_assets", 0.0)),
            "current_liabilities": float(bank.get("current_liabilities", 0.0)),
            "deposits": float(liab.get("deposits", 0.0)),
            "interbank_borrowing": float(liab.get("interbank", 0.0)),
            "wholesale_funding": float(liab.get("wholesale", 0.0)),
            "interbank_assets": float(bank.get("interbank_assets", 0.0)),
            "interbank_liabilities": float(bank.get("interbank_liabilities", 0.0)),
            "project_amount": float(proj.get("amount", 0.0)),
            "solvency_ratio": float(bank.get("solvency_ratio", 0.0)),
            "capital_adequacy_ratio": float(bank.get("capital_adequacy_ratio", 0.0)),
            "liquidity_coverage_ratio": float(bank.get("liquidity_coverage_ratio", 0.0)),
            "leverage_ratio": float(bank.get("leverage_ratio", 0.0)),
            "risk_appetite": float(bank.get("risk_appetite", 0.0)),
            "market_volatility": float(bank.get("market_volatility", 0.0)),
            "loan_interest_rate": float(bank.get("loan_interest_rate", 0.0)),
            "investment_interest_rate": float(bank.get("investment_interest_rate", 0.0)),
            "outflow_rate": float(bank.get("outflow_rate", 0.0)),
        })
    csv_path = output_dir / f"{prefix}_initial_bank_data.csv"
    json_path = output_dir / f"{prefix}_initial_bank_data.json"
    try:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()) if rows else ["bank_id"],
            )
            writer.writeheader()
            writer.writerows(rows)
        json_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except PermissionError:
        global INITIAL_STATE_EXPORT_LOCK_WARNED
        if not INITIAL_STATE_EXPORT_LOCK_WARNED:
            print(
                "[warn] 初始状态诊断文件被其他程序或 OneDrive 锁定，"
                "跳过本次导出；仿真将继续运行。"
            )
            INITIAL_STATE_EXPORT_LOCK_WARNED = True


def export_policy_logs_excel(summary_rows, event_rows, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    event_df = pd.DataFrame(event_rows)
    excel_path = output_dir / f"{prefix}_central_bank_policy_log.xlsx"
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "policy_summary"
    if summary_df.empty:
        ws1.append(["step"])
    else:
        ws1.append(list(summary_df.columns))
        for row in summary_df.itertuples(index=False, name=None):
            ws1.append(list(row))
    ws2 = wb.create_sheet("policy_events")
    if event_df.empty:
        ws2.append(["step"])
    else:
        ws2.append(list(event_df.columns))
        for row in event_df.itertuples(index=False, name=None):
            ws2.append(list(row))
    summary_csv = output_dir / f"{prefix}_central_bank_policy_summary.csv"
    events_csv = output_dir / f"{prefix}_central_bank_policy_events.csv"
    try:
        wb.save(excel_path)
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        event_df.to_csv(events_csv, index=False, encoding="utf-8-sig")
    except PermissionError as e:
        print(f"[policy_log] Skip export because file is locked or not writable: {e}")


@dataclass
class CentralBankPolicyAction:
    policy_rate: float
    reserve_requirement: float
    liquidity_support_ratio: float
    facility_spread: float
    note: str = ""


class BankNetworkSimulator:
    def __init__(self, num_banks=30, max_steps=5, B=1200, sigma=0.3, free_market=False, seed: int | None = DEFAULT_RANDOM_SEED):
        self.seed = DEFAULT_RANDOM_SEED if seed is None else int(seed)
        set_random_seed(self.seed)
        ss = np.random.SeedSequence(self.seed)
        env_ss, proj_ss, match_ss = ss.spawn(3)
        self.rng_environment = np.random.default_rng(env_ss)
        self.rng_project = np.random.default_rng(proj_ss)
        self.rng_matching = np.random.default_rng(match_ss)
        # 向后兼容：self.rng 仅作 environment 别名；项目相关必须用 rng_project，撮合用 rng_matching
        self.rng = self.rng_environment
        self.num_banks = num_banks
        self.max_steps = max_steps
        self.B = B
        self.sigma = sigma
        self.free_market = free_market

        self.bank_names = ['CentralBank'] + [f'Bank{i}' for i in range(1, num_banks)]
        _all_types = ['central'] + ['commercial'] * 20 + ['shadow'] * 9
        self.bank_types = _all_types[:num_banks]
        self.colors = {'central': 'gold', 'commercial': 'lightblue', 'shadow': 'lightcoral'}
        self.simulation_history = []
        self.record_history = False
        self.market_environment = None
        self.market_volatility = 0.3
        self.base_rate = DAILY_BULL_BASE_RATE
        self.clear_max_iter = 100
        self.clear_tol = 1e-6
        self.initial_base_rate = self.base_rate
        self.long_term_rate = self.base_rate + DAILY_LONG_RATE_SPREAD_BULL[0]
        self.market_duration = 0
        self.market_duration_limit = random.randint(2, 3)
        self.prev_market_environment = None

        self.stock_price = {i: 1.0 for i in range(num_banks)}
        self.bond_price  = {i: 1.0 for i in range(num_banks)}
        self.price_sensitivity = 0.0001

        # === 网络稀疏与角色-连边控制（新）===
        self.eta = 0.15
        self.link_density = 0.50
        self.max_degree = 12
        self.central_edge_ratio = 0.15
        self.max_central_degree_per_noncentral = 1

        # —— 禁止“借入再放贷”所需的最小台账 —— 
        self.borrowed_cash  = np.zeros(self.num_banks, dtype=float)
        # Σ(成交金额 × 成交前 borrower risk)
        self.borrowed_origination_risk_sum = np.zeros(
            self.num_banks,
            dtype=float,
        )
        self.ib_asset       = np.zeros(self.num_banks, dtype=float)
        self.ib_liab        = np.zeros(self.num_banks, dtype=float)
        self.reserve_buffer = np.full(self.num_banks, 0.02, dtype=float)

        # 项目池（唯一非同业投资资产）
        self.project_book: list[list[ProjectLoan]] = [[] for _ in range(self.num_banks)]
        self.project_min_share = 0.80
        self.reserve_min_share = 0.20

        # ========= 自由市场模式覆盖 =========
        if self.free_market:
            self.reserve_buffer[:] = 0.005
            self.project_min_share = 0.50
            self.reserve_min_share = 0.50
            self.link_density = 0.80
            self.max_degree   = 12
            self.central_edge_ratio = 0.40

        # （可选）避免 A↔B 来回拆借的“反向记忆”
        self.forbid_reciprocal_history = True
        self.reciprocal_cooldown = None
        self._pair_dir  = {}
        self._pair_step = {}

        # 让 CAR 阈值变成实例属性
        self.car_cutoff = 0.08
        self.lcr_cutoff = DAILY_INTERBANK_ROLE_LCR_CUTOFF

        self.prev_exposure_matrix = None
        self.contract_book = ContractBook()
        self.interbank_contract_maturity = DAILY_INTERBANK_CONTRACT_MATURITY
        self.central_corridor: CentralBankCorridor | None = None
        self._validate_book_sr = False

    # === 统一的“安全版 CAR”计算函数 ===
    def _safe_car_value(
        self,
        core_capital: float,
        interbank_assets: float,
        projects_amt: float,
        current_liabilities: float | None = None,
    ) -> float:
        """CAR via bank_regulatory.regulatory_car / regulatory_rwa."""
        return regulatory_car(
            core_capital=float(core_capital),
            interbank_assets=float(interbank_assets),
            projects_amt=float(projects_amt),
            liabilities=(
                None if current_liabilities is None else max(0.0, float(current_liabilities))
            ),
        )

    def _safe_lcr_value(
        self,
        liquid_assets: float,
        current_liabilities: float,
        outflow_rate: float,
        interbank_liabilities: float = 0.0,
    ) -> float:
        """LCR with minimum stress outflow and display cap.

        Denominator includes external liabilities + outstanding interbank
        liabilities so overnight / rollover claims affect liquidity pressure.
        """
        eps = 1e-9
        lia = max(0.0, float(current_liabilities)) + max(0.0, float(interbank_liabilities))
        liq = max(0.0, float(liquid_assets))
        stress = lia * max(float(outflow_rate), DAILY_MIN_LCR_OUTFLOW_RATE)
        if stress <= eps:
            return 0.0 if liq <= eps else DAILY_MAX_LCR_RATIO
        return float(np.clip(liq / stress, 0.0, DAILY_MAX_LCR_RATIO))

    def _update_network_stability(self, step: int, risk: float) -> bool:
        """连续多期网络敞口和风险都几乎不变时，判定为稳定。"""
        L = np.asarray(
            getattr(self, "exposure_matrix", np.zeros((self.num_banks, self.num_banks))),
            dtype=float,
        )
        active = tuple(bool(b.get("is_active", True)) for b in getattr(self, "banks", []))

        prev_L = getattr(self, "_prev_stability_exposure_matrix", None)
        prev_risk = getattr(self, "_prev_stability_risk", None)
        prev_active = getattr(self, "_prev_stability_active", None)
        if prev_L is None or prev_risk is None or prev_active is None:
            self._prev_stability_exposure_matrix = L.copy()
            self._prev_stability_risk = float(risk)
            self._prev_stability_active = active
            self.network_stable_count = 0
            return False

        denom = max(float(np.linalg.norm(prev_L)), float(np.linalg.norm(L)), 1.0)
        exposure_change = float(np.linalg.norm(L - prev_L) / denom)
        risk_change = abs(float(risk) - float(prev_risk))
        active_changed = active != prev_active

        if (
            int(step) >= int(getattr(self, "network_stability_min_step", 20))
            and not active_changed
            and exposure_change <= float(getattr(self, "network_stability_exposure_tol", 5e-3))
            and risk_change <= float(getattr(self, "network_stability_risk_tol", 1e-3))
        ):
            self.network_stable_count = int(getattr(self, "network_stable_count", 0)) + 1
        else:
            self.network_stable_count = 0

        self._prev_stability_exposure_matrix = L.copy()
        self._prev_stability_risk = float(risk)
        self._prev_stability_active = active

        if (
            getattr(self, "network_stable_step", None) is None
            and self.network_stable_count >= int(getattr(self, "network_stability_window", 50))
        ):
            self.network_stable_step = int(step)
            print(
                f"[NETWORK STABLE] step={self.network_stable_step} "
                f"(stable_count={self.network_stable_count}, "
                f"exposure_change={exposure_change:.3e}, risk_change={risk_change:.3e})"
            )
            return True
        return getattr(self, "network_stable_step", None) is not None

    def _refresh_exposure_from_contract_book(self, step: int) -> np.ndarray:
        self.exposure_matrix = aggregate_contracts_to_exposure_matrix_at_step(
            self.contract_book,
            self.num_banks,
            int(step),
        )
        np.fill_diagonal(self.exposure_matrix, 0.0)
        return self.exposure_matrix

    def _schedule_cfg(self) -> ScheduleConfig:
        return schedule_config_from_mapping({
            "schedule_selection": getattr(self, "schedule_selection", "auto"),
            "single_payment_maturity_periods": getattr(self, "interbank_contract_maturity", 1),
            "bullet_maturity_periods": getattr(self, "interbank_contract_maturity", 1),
            "bullet_max_principal": getattr(self, "bullet_max_principal", 250.0),
            "installment_min_principal": getattr(self, "installment_min_principal", 400.0),
            "lcr_installment_cutoff": getattr(self, "lcr_installment_cutoff", 1.0),
            "rollover_min_tenor": getattr(self, "rollover_min_tenor", 5),
            "rollover_max_tenor": getattr(self, "rollover_max_tenor", 20),
            "rollover_ref_small": getattr(self, "rollover_ref_small", 50.0),
            "rollover_ref_large": getattr(self, "rollover_ref_large", 2000.0),
            "rollover_spread_short": getattr(self, "rollover_spread_short", DAILY_ROLLOVER_SPREAD_SHORT),
            "rollover_spread_long": getattr(self, "rollover_spread_long", DAILY_ROLLOVER_SPREAD_LONG),
            "rollover_mode": getattr(self, "rollover_mode", "installment"),
            "rollover_extension_periods": getattr(
                self, "rollover_extension_periods", self.interbank_contract_maturity
            ),
            "soft_principal_deferral": bool(
                getattr(self, "soft_principal_deferral", True)
            ),
            "allow_early_full_repay": bool(getattr(self, "allow_early_full_repay", False)),
            "early_repay_lcr_buffer": float(getattr(self, "early_repay_lcr_buffer", 0.85)),
        })

    def _settle_interbank_installment_period(self, step: int) -> list[int]:
        """分期/单期：全市场 due_flows 合并后一次 EN。"""
        step = int(step)
        n = self.num_banks
        corridor = getattr(self, "central_corridor", None) or CentralBankCorridor(
            deposit_rate=max(0.0, float(self.base_rate) - DAILY_CB_DEPOSIT_SPREAD),
            lending_rate=float(self.base_rate) + DAILY_CB_LENDING_SPREAD,
            base_rate=float(self.base_rate),
        )

        def _issue(i: int, amt: float, st: int) -> None:
            """Ring-fenced settlement liquidity: only cover current payment gap for viable banks."""
            bank = self.banks[i]
            if not bank.get("is_active", True):
                return
            if self._bank_equity(bank) <= 0.0:
                return
            if float(_bank_measure_car(bank)) < float(
                getattr(self, "policy_car_floor", 0.06)
            ):
                return
            need = float(amt)  # payment gap = due - LA from settle_interbank_period
            if need <= 1e-9:
                return
            la_pre = float(bank.get("liquid_assets", 0.0))
            injected = self._issue_central_bank_liquidity_support(
                i, need, st, rate=corridor.lending_rate, tenor=1, kind="settlement_backstop",
            )
            if injected > 1e-9:
                idx = int(i)
                self._settlement_support_issued[idx] = (
                    float(self._settlement_support_issued.get(idx, 0.0)) + float(injected)
                )
                self._settlement_support_la_pre.setdefault(idx, la_pre)
                self._settlement_support_due[idx] = float(la_pre + need)

        def _run_en_and_store(liabilities, banks, n, use_core=False, max_iter=100, tol=1e-6):
            p, failed = run_en_clearing_and_recovery(
                liabilities, banks, n, use_core=use_core, max_iter=max_iter, tol=tol
            )
            self._last_en_payments = np.asarray(p, dtype=float).copy()
            return p, failed

        self._settlement_support_issued = {}
        self._settlement_support_la_pre = {}
        self._settlement_support_due = {}
        self._last_en_payments = np.zeros(n, dtype=float)
        self.last_settlement_support_used = 0.0
        settle_result = settle_interbank_period(
            self.contract_book,
            self.banks,
            n,
            step,
            cfg=self._schedule_cfg(),
            liquidity_default_candidates=liquidity_default_candidates,
            run_en_clearing_and_recovery=_run_en_and_store,
            # Contemporaneous EN backstop disabled: stress is queued after settlement and injected at t+1 start.
            issue_liquidity_support=None,
            corridor_lending_rate=float(corridor.lending_rate),
            use_core=False,
            clear_max_iter=int(getattr(self, "clear_max_iter", 100)),
            clear_tol=float(getattr(self, "clear_tol", 1e-6)),
            verbose_rollover=bool(getattr(self, "verbose_rollover", True)),
            clawback_settlement_support=self._clawback_unused_settlement_support,
        )
        from interbank_installment_rollover import (
            active_installment_rollover_borrowers,
            log_rollover_borrow_policy_note,
        )

        self.rollover_coupon_due_borrowers = set(settle_result.coupon_due_borrowers)
        self.rollover_coupon_cleared_borrowers = set(settle_result.coupon_cleared_borrowers)
        self.rollover_active_borrowers = active_installment_rollover_borrowers(
            self.contract_book, step
        )
        # Active installment borrowers are passed to the borrowing policy.
        # coupon_cleared blocks only those that did not clear today's coupon;
        # it does not turn arrears into a default.
        self.rollover_blocked_borrowers = (
            set(self.rollover_active_borrowers)
            if bool(getattr(self, "rollover_enabled", False))
            else set()
        )
        failed = settle_result.failed
        self.last_en_unpaid = float(getattr(settle_result, "unpaid_amount", 0.0))
        self.last_defaulted_banks = [int(i) for i in failed if 0 < int(i) < len(self.banks)]

        # Do not mutate here; simulate_step calls _resolve_bank_default (LGD write-off).
        self._refresh_exposure_from_contract_book(step)
        return failed

    def _settle_due_interbank_contracts(self, step: int) -> list[int]:
        """到期/分期结算（兼容旧调用名）。"""
        return self._settle_interbank_installment_period(step)

    def _clawback_unused_settlement_support(
        self,
        liabilities: np.ndarray,
        payments: np.ndarray,
        step: int,
    ) -> None:
        """
        Immediately after one EN clearing phase, return unused ring-fenced
        settlement support and restore policy budgets.

        used = min(S, max(0, p_i - recv_i - LA_pre))
        where recv = Pi.T @ p is interbank receipts in this phase.
        """
        L = np.asarray(liabilities, dtype=float)
        p = np.asarray(payments, dtype=float)
        if L.ndim != 2 or p.ndim != 1:
            return
        p_bar = L.sum(axis=1)
        Pi = np.divide(
            L,
            p_bar[:, None],
            out=np.zeros_like(L),
            where=p_bar[:, None] > 0.0,
        )
        recv = Pi.T @ p

        issued = getattr(self, "_settlement_support_issued", {}) or {}
        la_pre_map = getattr(self, "_settlement_support_la_pre", {}) or {}
        for i, S in list(issued.items()):
            S = float(S)
            if S <= 1e-9 or i <= 0 or i >= len(self.banks):
                continue
            bank = self.banks[i]
            la_pre = float(la_pre_map.get(i, 0.0))
            p_i = float(p[i]) if i < len(p) else 0.0
            recv_i = float(recv[i]) if i < len(recv) else 0.0
            used = min(S, max(0.0, p_i - recv_i - la_pre))
            unused = max(0.0, S - used)
            self.last_settlement_support_used = float(
                getattr(self, "last_settlement_support_used", 0.0)
            ) + float(used)
            la_now = float(bank.get("liquid_assets", 0.0))
            unused = min(unused, la_now)
            if unused <= 1e-9:
                continue
            bank["liquid_assets"] = la_now - unused
            bank["current_liabilities"] = max(
                0.0, float(bank.get("current_liabilities", 0.0)) - unused
            )
            bank["cb_policy_balance"] = max(
                0.0, float(bank.get("cb_policy_balance", 0.0)) - unused
            )
            self._reduce_settlement_backstop_book(int(i), float(unused), int(step))
            self.cb_remaining_budget = min(
                float(getattr(self, "cb_total_budget", 0.0)),
                float(getattr(self, "cb_remaining_budget", 0.0)) + unused,
            )
            self.cb_step_budget_remaining = min(
                float(getattr(self, "cb_step_budget", 0.0)),
                float(getattr(self, "cb_step_budget_remaining", 0.0)) + unused,
            )
            if bool(getattr(self, "verbose_matching", False)):
                print(
                    f"[policy] step={int(step)} clawback unused settlement support "
                    f"bank={int(i)} unused={unused:.4f}"
                )
        self._settlement_support_issued = {}
        self._settlement_support_la_pre = {}
        self._settlement_support_due = {}

    def _reduce_settlement_backstop_book(
        self, bank_idx: int, amount: float, step: int
    ) -> None:
        """Reduce principal on settlement_backstop loans after clawback."""
        remain = float(amount)
        if remain <= 1e-9:
            return
        book = list(getattr(self, "policy_support_book", []) or [])
        for loan in reversed(book):
            if remain <= 1e-9:
                break
            if int(loan.get("bank_idx", -1)) != int(bank_idx):
                continue
            loan_kind = str(loan.get("kind", "")).removeprefix("policy_")
            if loan_kind != "settlement_backstop":
                continue
            prin = float(loan.get("principal", 0.0))
            if prin <= 1e-9:
                continue
            take = min(prin, remain)
            loan["principal"] = prin - take
            remain -= take
            getattr(self, "policy_event_log", []).append({
                "step": int(step),
                "bank_idx": int(bank_idx),
                "event": "settlement_support_clawback",
                "amount": float(take),
            })
        self.policy_support_book = [
            loan for loan in book if float(loan.get("principal", 0.0)) > 1e-9
        ]

    def initialize_network(self):
        """初始化银行、仅项目投资；同业边由撮合函数生成并做现金结算。"""
        # 市场环境 & 利率
        self.market_environment = 'bull' if random.random() < 0.6 else 'bear'
        self.prev_market_environment = self.market_environment
        self.market_duration = 0
        self.market_duration_limit = getattr(self, 'market_duration_limit', random.randint(2, 5))

        self.base_rate = DAILY_BULL_BASE_RATE if self.market_environment == 'bull' else DAILY_BEAR_BASE_RATE
        self.initial_base_rate = self.base_rate
        long_spread = DAILY_LONG_RATE_SPREAD_BULL if self.market_environment == 'bull' else DAILY_LONG_RATE_SPREAD_BEAR
        self.long_term_rate = self.base_rate + random.uniform(*long_spread)

        # —— 初始化银行列表（项目资产为唯一非同业资产）——
        self.banks = []
        self.contract_book = ContractBook()
        from bank_econ_shared import reset_screening_loss
        reset_screening_loss(self, totals=True, step=True)
        self.borrowed_cash[:] = 0.0
        self.borrowed_origination_risk_sum[:] = 0.0
        self.ib_asset[:] = 0.0
        self.ib_liab[:] = 0.0
        for i in range(self.num_banks):
            t = self.bank_types[i]

            # 环境相关参数
            if self.market_environment == 'bull':
                cap_mul, liq_mul, lia_mul = random.uniform(1.1, 1.2), random.uniform(1.1, 1.2), random.uniform(0.8, 0.9)
                inv_ret = random.uniform(*DAILY_INVESTMENT_RETURN_BULL)
                vol     = random.uniform(10, 20) / 50
                loan_rt = self.base_rate + random.uniform(*DAILY_LOAN_SPREAD_BULL)
                risk_app = random.uniform(0.7, 1.0) if t != 'central' else 0.3
            else:
                cap_mul, liq_mul, lia_mul = random.uniform(0.8, 0.9), random.uniform(0.8, 0.9), random.uniform(1.1, 1.2)
                inv_ret = random.uniform(*DAILY_INVESTMENT_RETURN_BEAR)
                vol     = random.uniform(30, 50) / 50
                loan_rt = self.base_rate + random.uniform(*DAILY_LOAN_SPREAD_BEAR)
                risk_app = random.uniform(0.0, 0.3) if t != 'central' else 0.3

            # 核心资本 / 流动性（央行更充裕）
            core = 10000.0 if i == 0 else float(random.randint(1000, 5000)) * cap_mul
            liq  = 5000.0  if i == 0 else float(random.randint(500, 2000)) * liq_mul

            # 负债结构（比例）
            if t == 'central':
                dep_ratio = random.uniform(0.10, 0.30); ib_ratio = random.uniform(0.05, 0.10); wf_ratio = random.uniform(0.00, 0.05)
            elif t == 'commercial':
                dep_ratio = random.uniform(0.40, 0.60); ib_ratio = random.uniform(0.10, 0.20); wf_ratio = random.uniform(0.10, 0.20)
            else:  # shadow
                dep_ratio = 0.0; ib_ratio = random.uniform(0.20, 0.30); wf_ratio = random.uniform(0.40, 0.50)

            # 风险偏好微调
            boost = 0.1 * risk_app
            env_mul = 1.0 if self.market_environment == 'bull' else 0.95
            dep_ratio = (dep_ratio + boost) * env_mul
            ib_ratio  = (ib_ratio  + boost) * env_mul
            wf_ratio  = (wf_ratio  + boost) * env_mul

            # 负债绝对值：用随机规模参数生成负债；权益由目标 CAR 反推
            deposits = core * dep_ratio
            wholesale_funding = core * wf_ratio
            interbank_borrowing = 0.0
            lia = deposits + wholesale_funding
            interbank_assets0 = 0.0
            liquid_target = float(liq)

            from bank_econ_shared import solve_initial_equity_for_target_car

            if inv_ret > loan_rt:
                if t == 'central':
                    invest = liquid_target * random.uniform(0.05, 0.10)
                elif t == 'shadow':
                    invest = liquid_target * random.uniform(0.20, 0.30) * (1 + risk_app)
                else:
                    invest = liquid_target * random.uniform(0.10, 0.20) * (1 + 0.5 * risk_app)
                invest *= (1.1 if self.market_environment == 'bull' else 0.7)
            else:
                invest = 0.0

            intended_core, liq, projects_amt, total_assets = solve_initial_equity_for_target_car(
                liabilities=lia,
                liquid_target=liquid_target,
                risk_appetite=float(risk_app),
                interbank_assets=interbank_assets0,
                invest=float(invest),
            )

            total_assets = liq + interbank_assets0 + projects_amt
            # ===== project_book init  =====
            self.project_book[i].clear()

            if projects_amt > 1e-8:
                from project_common_random import (
                    PROJECT_ORIGIN_INITIAL,
                    make_project_id,
                    project_creation_draws,
                )

                pid = make_project_id(
                    creation_step=-1,
                    origin=PROJECT_ORIGIN_INITIAL,
                    slot=0,
                )
                initial_project_maturity, _, _ = project_creation_draws(
                    seed=int(getattr(self, "seed", DEFAULT_RANDOM_SEED)),
                    creation_step=-1,
                    bank_id=i,
                    project_id=pid,
                    maturity_range=DAILY_PROJECT_MATURITY_DAYS,
                    pd_range=DAILY_PROJECT_PD_RANGE,
                    lgd_range=(0.30, 0.60),
                )
                initial_project_age = int((i * 3) % max(1, initial_project_maturity))
                if i == 0:
                    self.project_book[i].append(ProjectLoan(
                        principal=float(projects_amt),
                        rate=0.0,
                        maturity=initial_project_maturity,
                        age=initial_project_age,
                        pd=0.0,
                        lgd=0.0,
                        project_id=pid,
                        origination_risk=0.0,
                    ))
                else:
                    self.project_book[i].append(ProjectLoan(
                        principal=float(projects_amt),
                        rate=float(self.long_term_rate + DAILY_PROJECT_SPREAD),
                        maturity=initial_project_maturity,
                        age=initial_project_age,
                        pd=DAILY_PROJECT_PD_DEFAULT,
                        lgd=0.4,
                        project_id=pid,
                        origination_risk=0.0,
                    ))
            else:
                # projects_amt 很小就不建项目；保持为空即可
                pass

            core_cap = float(intended_core)
            init_equity = total_assets - lia
            solv_ratio = init_equity / (lia + 1e-9)
            bank = {
                "id": i,
                "name": self.bank_names[i],
                "type": t,
                "is_active": True,
                "failure_step": None,
                "failure_reason": None,
                "absorbing_default": False,
                "balance_sheet_frozen": False,
                "frozen_balance_sheet": None,
                "core_capital": core_cap,
                "liquid_assets": liq,
                "total_assets": total_assets,
                "current_liabilities": lia,
                "liabilities_breakdown": {
                    "deposits": deposits,
                    "interbank": interbank_borrowing,
                    "wholesale": wholesale_funding,
                },
                "interbank_assets": interbank_assets0,
                "interbank_liabilities": 0.0,
                "termed_out_liabilities": 0.0,
                "solvency_ratio": solv_ratio,
                "capital_ratio_history": [solv_ratio],
                "risk_appetite": risk_app,
                "market_volatility": vol,
                "proj_mu": float(inv_ret),
                "proj_sigma": float(
                    self.rng_project.uniform(0.00010, 0.00020)
                    if self.market_environment == "bull"
                    else self.rng_project.uniform(0.00014, 0.00026)
                ),
                "loan_interest_rate": loan_rt,
                "investment_interest_rate": self.long_term_rate + random.uniform(*DAILY_INVESTMENT_SPREAD_INIT),
                "outflow_rate": 0.2 if t == "central" else random.uniform(0.3, 0.5),
                "investment": {"projects": {"amount": projects_amt, "risk_weight": 1.0}},
                "capital_adequacy_ratio": self._safe_car_value(core_cap, interbank_assets0, projects_amt, lia),
                "liquidity_coverage_ratio": self._safe_lcr_value(
                    liq, lia, 0.2 if t == "central" else 0.4
                ),
                "leverage_ratio": core_cap / (liq + interbank_assets0 + 1e-9),
                "pending_endowment": 0.0,
                "hurdle_rate": DAILY_HURDLE_RATE,
                "defaulted": False,
                "chi_role": 0,
                "reservation_rate": loan_rt,
                "demand": 0.0,
                "supply": 0.0,
            }
            self.banks.append(bank)

        # —— 同业矩阵置零；不在初始化撮合，首次交易从 simulate_step(0) 开始 ——
        self.exposure_matrix = np.zeros((self.num_banks, self.num_banks), dtype=float)
        self.contract_book = ContractBook()
        self.current_step = 0
        self.roles = (
            self.assign_roles_by_risk(
                car_cutoff=getattr(self, "car_cutoff", 0.08),
                lcr_cutoff=getattr(self, "lcr_cutoff", DAILY_INTERBANK_ROLE_LCR_CUTOFF),
            )
            if hasattr(self, 'assign_roles_by_risk') else self.assign_roles()
        )

        # 初始化无同业边：同业科目保持 0，监管指标基于项目资产
        for idx, b in enumerate(self.banks):
            b['interbank_assets'] = 0.0
            b['interbank_liabilities'] = 0.0
            b['capital_adequacy_ratio'] = self._safe_car_value(
                b['core_capital'],
                0.0,
                b['investment']['projects']['amount'],
                b['current_liabilities'],
            )
            b['leverage_ratio'] = b['core_capital'] / (b['liquid_assets'] + 1e-9)
            b['total_assets'] = (
                float(b['liquid_assets'])
                + float(b.get('interbank_assets', 0.0))
                + float(b['investment']['projects']['amount'])
            )
            # Backfill initial project origination_risk after CAR/LCR exist.
            from bank_econ_shared import borrower_origination_risk_q
            q0 = 0.0 if idx == 0 else float(
                borrower_origination_risk_q(
                    b,
                    system=self,
                    bank_idx=idx,
                    car_threshold=float(getattr(self, "car_cutoff", 0.08)),
                )
            )
            for loan in self.project_book[idx]:
                loan.origination_risk = float(q0)

        np.fill_diagonal(self.exposure_matrix, 0.0)
        self.one_shot_default_done = False
        self.all_default_step = None
        self.network_stable_step = None
        self.network_stable_count = 0
        self.network_stability_window = 50
        self.network_stability_min_step = self.network_stability_window
        self.network_stability_exposure_tol = 5e-3
        self.network_stability_risk_tol = 0.01
        self._prev_stability_exposure_matrix = None
        self._prev_stability_risk = None
        self._prev_stability_active = None
        # Respect configure_simulation_features Support ON/OFF (rate+reserve+injection).
        if getattr(self, "policy_support_enabled", True) is False:
            self.policy_enabled = False
            self.central_bank_support_enabled = False
            self.solvency_support_enabled = False
        else:
            self.policy_enabled = bool(getattr(self, "policy_enabled", True))
        self.policy_rate_floor = DAILY_POLICY_RATE_FLOOR
        self.policy_rate_ceiling = DAILY_POLICY_RATE_CEILING
        self.policy_rate_decision_interval = 4
        self.policy_rate_max_step_change = DAILY_POLICY_RATE_MAX_STEP_CHANGE
        self.policy_rate_change_threshold = DAILY_POLICY_RATE_CHANGE_THRESHOLD
        self.last_policy_rate_update_step = -10**9
        self.normal_reserve_requirement = 0.005 if self.free_market else 0.02
        self.reserve_buffer[:] = self.normal_reserve_requirement
        self.policy_sr_target = 0.20
        self.policy_sr_defensive_threshold = 0.18
        self.policy_sr_crisis_threshold = 0.34
        self.last_systemic_risk = 0.0
        self.last_raw_systemic_risk = 0.0
        self.last_collapse_index = 0.0
        self.policy_lcr_target = 1.00
        self.policy_car_floor = 0.06
        self.cb_loan_tenor = 2
        self.cb_penalty_spread = DAILY_CB_PENALTY_SPREAD
        # Temporary calibration: tighter CB support so ON still helps but does not flatten SR.
        self.cb_max_support_share = 0.12
        self.cb_broad_support_share = 0.02
        self.cb_total_budget = 12000.0
        self.cb_remaining_budget = self.cb_total_budget
        self.cb_step_budget = 800.0
        self.cb_step_budget_remaining = self.cb_step_budget
        self.cb_total_injected = 0.0
        # 尊重 configure_simulation_features 的 policy_support 开关
        if not hasattr(self, "solvency_support_enabled"):
            self.solvency_support_enabled = True
        if getattr(self, "policy_support_enabled", True) is False:
            self.solvency_support_enabled = False
            self.central_bank_support_enabled = False
        else:
            self.solvency_support_enabled = True
        self.solvency_support_car_trigger = 0.075
        self.solvency_support_car_floor = 0.03
        # Capital support is preventive: it may recapitalize a viable bank but
        # must not resurrect a bank whose accounting equity is already nonpositive.
        self.solvency_support_equity_floor = 0.0
        self.solvency_support_spread = DAILY_SOLVENCY_SUPPORT_SPREAD
        self.solvency_support_max_share = 0.08
        self.solvency_support_tenor = 4
        self.solvency_support_budget = 6000.0
        self.solvency_support_remaining_budget = self.solvency_support_budget
        self.solvency_support_step_budget = 400.0
        self.solvency_support_step_remaining = self.solvency_support_step_budget
        self.policy_support_book = []
        self.policy_support_book_outstanding_limit = self.cb_total_budget
        self.policy_support_cooldown_steps = 1
        self.policy_support_last_step_by_bank_kind: dict[tuple[int, str], int] = {}
        self.policy_history = []
        self.policy_event_log = []
        self.export_policy_logs = getattr(self, "export_policy_logs", True)
        self.last_policy_note = "policy_init"
        self.initial_state_export_prefix = "centralized_central_policy"
        self.rollover_blocked_borrowers: set[int] = set()
        self.rollover_active_borrowers: set[int] = set()
        self.rollover_borrow_policy = ROLLOVER_BORROW_COUPON_CLEARED
        self.rollover_coupon_cleared_borrowers: set[int] = set()
        self.rollover_coupon_due_borrowers: set[int] = set()
        # 保留 configure_simulation_features 已写入的调度开关，勿硬编码覆盖
        self.rollover_mode = str(
            getattr(
                self,
                "rollover_mode",
                "installment"
                if getattr(self, "rollover_enabled", True)
                else "off",
            )
        )
        self.schedule_selection = str(
            getattr(
                self,
                "schedule_selection",
                "installment"
                if getattr(self, "rollover_enabled", True)
                else "single_payment",
            )
        )
        self.bullet_max_principal = 150.0
        self.installment_min_principal = 200.0
        self.lcr_installment_cutoff = 1.0
        self.rollover_spread_short = DAILY_ROLLOVER_SPREAD_SHORT
        self.rollover_spread_long = DAILY_ROLLOVER_SPREAD_LONG
        self.rollover_spread = DAILY_ROLLOVER_SPREAD
        self.rollover_extension_periods = DAILY_INTERBANK_CONTRACT_MATURITY
        self.rollover_min_tenor = int(getattr(self, "rollover_min_tenor", 5))
        self.rollover_max_tenor = int(getattr(self, "rollover_max_tenor", 20))
        self.soft_principal_deferral = bool(
            getattr(self, "soft_principal_deferral", True)
        )
        self.allow_early_full_repay = bool(getattr(self, "allow_early_full_repay", False))
        self.early_repay_lcr_buffer = float(getattr(self, "early_repay_lcr_buffer", 0.85))
        self.rollover_ref_small = 50.0
        self.rollover_ref_large = 2000.0
        self.interbank_contract_maturity = DAILY_INTERBANK_CONTRACT_MATURITY
        self.verbose_matching = getattr(self, "verbose_matching", True)
        self.verbose_rollover = getattr(self, "verbose_rollover", True)
        self.trade_schedule_log: list[dict] = []
        self.last_match_stats: dict = {}
        self.last_en_unpaid: float = 0.0
        self.last_interbank_writeoff: float = 0.0
        self.cumulative_interbank_writeoff: float = 0.0
        self.last_estate_transfer_discount: float = 0.0
        self.cumulative_estate_transfer_discount: float = 0.0
        # 中心化批量拍卖：默认每日撮合（与 RFQ 同频）；L=15 仅作稳健性实验
        self.centralized_cycle_length = int(
            getattr(self, "centralized_cycle_length", 1)
        )
        self.central_order_book: dict[int, dict] = {}
        self.last_was_auction_day = False
        # Opportunity-borrow discount φ and bank-level IBL capacity
        self.debt_burden_kappa = float(
            getattr(self, "debt_burden_kappa", DEFAULT_DEBT_BURDEN_KAPPA)
        )
        self.ibl_cap_asset_lambda = float(
            getattr(self, "ibl_cap_asset_lambda", DEFAULT_IBL_CAP_ASSET_LAMBDA)
        )
        # Pre-trade liquidity window off: liquidity support is EN settlement backstop only
        self.policy_pre_trade_liquidity_window = False
        self._settlement_support_issued: dict[int, float] = {}
        self._settlement_support_la_pre: dict[int, float] = {}
        self._settlement_support_due: dict[int, float] = {}
        self.last_defaulted_banks: list[int] = []
        self.pending_policy_support: list[dict] = []
        self.interbank_lgd = float(getattr(self, "interbank_lgd", 0.4))
        self.step_metrics_history: list[dict] = []
        self.interbank_lcr_target = float(
            getattr(self, "interbank_lcr_target", DAILY_INTERBANK_INTENTION_LCR_TARGET)
        )
        self.central_corridor = CentralBankCorridor(
            deposit_rate=max(0.0, float(self.base_rate) - DAILY_CB_DEPOSIT_SPREAD),
            lending_rate=float(self.base_rate) + DAILY_CB_LENDING_SPREAD,
            base_rate=float(self.base_rate),
        )
        export_initial_bank_table(
            self.banks,
            INITIAL_STATE_DIR,
            self.initial_state_export_prefix,
        )
        if hasattr(self, "calculate_systemic_risk"):
            self._record_systemic_risk(self.calculate_systemic_risk())

    def adjust_base_rate(self):
        """
        根据市场波动与活跃度微调基准利率。
        方向：波动↑ -> 降息；活跃度↑ -> 偏紧（小幅加息）。
        Support ON（policy_enabled）时由央行政策周期独占利率，本函数不覆盖。
        """
        if getattr(self, "policy_enabled", False):
            return
        if hasattr(self, "banks") and self.banks:
            vols = [float(b.get("market_volatility", 0.5)) for b in self.banks]
            active_ratio = (
                sum(1 for b in self.banks if b.get("is_active", True)) / max(1, len(self.banks))
            )
        else:
            vols = []
            active_ratio = 0.5

        avg_vol = float(np.mean(vols)) if vols else 0.5

        k_vol, k_act = 0.5, 0.2
        delta = -k_vol * (avg_vol - 0.5) + k_act * (active_ratio - 0.5)

        base0 = float(getattr(self, "base_rate", getattr(self, "initial_base_rate", DAILY_BULL_BASE_RATE)))
        delta *= DAILY_POLICY_RATE_MAX_STEP_CHANGE
        self.base_rate = float(np.clip(base0 + delta, DAILY_POLICY_RATE_FLOOR, DAILY_POLICY_RATE_CEILING))

    def _bank_lcr(self, bank) -> float:
        return self._safe_lcr_value(
            float(bank.get("liquid_assets", 0.0)),
            float(bank.get("current_liabilities", 0.0)),
            float(bank.get("outflow_rate", 0.4)),
            interbank_liabilities=float(bank.get("interbank_liabilities", 0.0)),
        )

    def _bank_equity(self, bank) -> float:
        return _bank_accounting_equity(bank)

    def _refresh_metrics_after_settle(self, step: int) -> None:
        """Rebuild IB stocks from ContractBook and refresh CAR/LCR before matching."""
        n = self.num_banks
        banks = self.banks
        book = self.contract_book
        update_bank_states_from_contract_book(banks, book, n, int(step))
        if hasattr(self, "_refresh_exposure_from_contract_book"):
            self._refresh_exposure_from_contract_book(int(step))
        elif hasattr(self, "exposure_matrix"):
            self.exposure_matrix = aggregate_contracts_to_exposure_matrix_at_step(
                book, n, int(step)
            )
            np.fill_diagonal(self.exposure_matrix, 0.0)
        em = getattr(self, "exposure_matrix", None)
        if em is not None:
            for i, b in enumerate(banks):
                if b.get("absorbing_default") or b.get("balance_sheet_frozen"):
                    em[i, :] = 0.0
                    em[:, i] = 0.0
        for b in banks:
            if b.get("absorbing_default") or b.get("balance_sheet_frozen"):
                continue
            equity = self._bank_equity(b)
            b["core_capital"] = max(0.0, float(equity))
            tot_lia = _bank_total_liabilities(b)
            b["solvency_ratio"] = float(equity) / (tot_lia + 1e-9)
            proj_amt = b["investment"]["projects"]["amount"]
            b["capital_adequacy_ratio"] = self._safe_car_value(
                b["core_capital"],
                b["interbank_assets"],
                proj_amt,
                tot_lia,
            )
            b["liquidity_coverage_ratio"] = self._safe_lcr_value(
                b["liquid_assets"],
                b["current_liabilities"],
                b["outflow_rate"],
                interbank_liabilities=float(b.get("interbank_liabilities", 0.0)),
            )
            b["leverage_ratio"] = b["core_capital"] / (
                b["liquid_assets"] + b["interbank_assets"] + 1e-9
            )
            b["total_assets"] = _bank_total_assets(b)

    def _is_liquidity_support_target(self, bank) -> bool:
        if not bank.get("is_active", True):
            return False
        lia = (
            float(bank.get("current_liabilities", 0.0))
            + float(bank.get("interbank_liabilities", 0.0))
        )
        liq = float(bank.get("liquid_assets", 0.0))
        outflow = lia * float(bank.get("outflow_rate", 0.4))
        required_liq = float(getattr(self, "policy_lcr_target", 1.0)) * outflow
        if required_liq - liq <= 1e-9 and self._bank_lcr(bank) >= self.policy_lcr_target:
            return False
        if self._bank_equity(bank) <= 0.0:
            return False
        if regulatory_car(bank) < self.policy_car_floor:
            return False
        return True

    def _is_capital_support_target(self, bank) -> bool:
        """Strictly positive-equity bank below the preventive CAR trigger."""
        if not bank.get("is_active", True):
            return False
        if bank.get("absorbing_default") or bank.get("balance_sheet_frozen"):
            return False
        equity = float(self._bank_equity(bank))
        assets = max(float(_bank_total_assets(bank)), 1e-9)
        equity_floor = float(
            getattr(self, "solvency_support_equity_floor", 0.0)
        ) * assets
        if equity <= max(0.0, equity_floor) + 1e-12:
            return False
        return car_below_threshold(
            regulatory_car(bank),
            float(getattr(self, "solvency_support_car_trigger", 0.075)),
        )

    def _run_central_bank_policy_cycle(self, step: int) -> None:
        policy_obs = self._observe_central_bank_conditions()
        policy_action = self._decide_central_bank_policy(policy_obs)
        self._apply_central_bank_policy(policy_action, int(step))

    def _record_systemic_risk(self, risk: float) -> None:
        """保存 raw SR（政策用）与单调 collapse_index（展示/崩溃轨迹用）。"""
        raw = float(np.clip(float(risk), 0.0, 1.0))
        self.last_raw_systemic_risk = raw
        prev = float(getattr(self, "last_collapse_index", 0.0))
        self.last_collapse_index = float(max(prev, raw))
        # Policy observes raw SR (may fall after high-stress banks exit).
        self.last_systemic_risk = raw

    def _observe_central_bank_conditions(self):
        """仅使用上一阶段已计算的 SR，不再重复构造 stress/aux 评分。"""
        sr = float(getattr(self, "last_systemic_risk", 0.0))
        return {"systemic_risk": sr}

    def _decide_central_bank_policy(self, obs) -> CentralBankPolicyAction:
        """
        三档政策：仅依据上一阶段 SR。
        - crisis_easing: SR >= policy_sr_crisis_threshold（默认 0.34）
        - defensive_easing: SR >= policy_sr_defensive_threshold（默认 0.18）
        - hold: 其余
        """
        base_target = float(getattr(self, "base_rate", DAILY_BULL_BASE_RATE))
        reserve_target = float(getattr(self, "normal_reserve_requirement", 0.02))
        sr = float(obs.get("systemic_risk", 0.0))
        sr_crisis = float(getattr(self, "policy_sr_crisis_threshold", 0.34))
        sr_defensive = float(getattr(self, "policy_sr_defensive_threshold", 0.18))

        if sr >= sr_crisis:
            return CentralBankPolicyAction(
                policy_rate=max(self.policy_rate_floor, base_target - DAILY_POLICY_EASING_CRISIS),
                reserve_requirement=max(0.005, reserve_target - 0.010),
                liquidity_support_ratio=0.20,
                facility_spread=DAILY_FACILITY_SPREAD_CRISIS,
                note="crisis_easing",
            )
        if sr >= sr_defensive:
            return CentralBankPolicyAction(
                policy_rate=max(self.policy_rate_floor, base_target - DAILY_POLICY_EASING_DEFENSIVE),
                reserve_requirement=max(0.0075, reserve_target - 0.005),
                liquidity_support_ratio=0.12,
                facility_spread=DAILY_FACILITY_SPREAD_DEFENSIVE,
                note="defensive_easing",
            )
        return CentralBankPolicyAction(
            policy_rate=float(np.clip(base_target, self.policy_rate_floor, self.policy_rate_ceiling)),
            reserve_requirement=reserve_target,
            liquidity_support_ratio=0.04,
            facility_spread=DAILY_FACILITY_SPREAD_HOLD,
            note="hold",
        )

    def _reset_policy_step_budget(self) -> None:
        self.cb_step_budget_remaining = min(
            float(getattr(self, "cb_step_budget", 0.0)),
            float(getattr(self, "cb_remaining_budget", 0.0)),
        )
        self.solvency_support_step_remaining = min(
            float(getattr(self, "solvency_support_step_budget", 0.0)),
            float(getattr(self, "solvency_support_remaining_budget", 0.0)),
        )

    def _settle_central_bank_loans(self, step: int) -> None:
        if not getattr(self, "policy_support_book", None):
            return
        open_loans = []
        for loan in self.policy_support_book:
            if int(loan["maturity_step"]) > int(step):
                open_loans.append(loan)
                continue
            bank_idx = int(loan["bank_idx"])
            if bank_idx <= 0 or bank_idx >= len(self.banks):
                continue
            principal = float(loan["principal"])
            rate = float(loan["rate"])
            due = principal * (1.0 + rate)
            bank = self.banks[bank_idx]
            current_lia = float(bank.get("current_liabilities", 0.0))
            interbank_lia = float(bank.get("interbank_liabilities", 0.0))
            total_lia = current_lia + interbank_lia
            outflow_rate = float(bank.get("outflow_rate", 0.4))
            buffer = (
                float(getattr(self, "policy_lcr_target", 1.0))
                * outflow_rate
                * total_lia
            )
            repayable_cash = max(0.0, float(bank.get("liquid_assets", 0.0)) - buffer)
            payment = min(repayable_cash, due)
            bank["liquid_assets"] = float(bank.get("liquid_assets", 0.0)) - payment
            principal_repaid = min(principal, payment * principal / (due + 1e-9))
            bank["current_liabilities"] = max(0.0, float(bank.get("current_liabilities", 0.0)) - principal_repaid)
            balance_key = loan.get("balance_key", "cb_policy_balance")
            bank[balance_key] = max(0.0, float(bank.get(balance_key, 0.0)) - principal_repaid)
            if loan.get("capital_like", False):
                bank["core_capital"] = max(0.0, float(bank.get("core_capital", 0.0)) - principal_repaid)
                bank["policy_capital_buffer"] = max(0.0, float(bank.get("policy_capital_buffer", 0.0)) - principal_repaid)
            budget_bucket = loan.get("budget_bucket", "liquidity")
            if budget_bucket == "solvency":
                self.solvency_support_remaining_budget = min(
                    self.solvency_support_budget,
                    float(self.solvency_support_remaining_budget) + principal_repaid,
                )
            else:
                self.cb_remaining_budget = min(
                    self.cb_total_budget,
                    float(self.cb_remaining_budget) + principal_repaid,
                )
            if payment + 1e-9 < due and bank.get("is_active", True):
                rolled = max(0.0, principal - principal_repaid)
                open_loans.append({
                    "bank_idx": bank_idx,
                    "principal": rolled,
                    "rate": min(rate + DAILY_POLICY_EASING_DEFENSIVE, DAILY_PENALTY_RATE_CEILING),
                    "created_step": int(step),
                    "maturity_step": int(step) + 1,
                    "kind": "policy_rollover",
                    "budget_bucket": budget_bucket,
                    "balance_key": balance_key,
                    "capital_like": bool(loan.get("capital_like", False)),
                })
            self.policy_event_log.append({
                "step": int(step),
                "event_type": "repayment",
                "bank_idx": bank_idx,
                "bank_name": bank.get("name", f"Bank{bank_idx}"),
                "policy_kind": loan.get("kind", ""),
                "budget_bucket": budget_bucket,
                "amount_principal_repaid": principal_repaid,
                "cash_payment": payment,
                "rate": rate,
                "remaining_liquidity_budget": float(self.cb_remaining_budget),
                "remaining_solvency_budget": float(self.solvency_support_remaining_budget),
            })
        self.policy_support_book = open_loans

    def _issue_policy_support(
        self,
        bank_idx: int,
        amount: float,
        step: int,
        rate: float,
        tenor: int,
        kind: str,
        *,
        per_bank_cap_share: float,
        budget_bucket: str,
        balance_key: str | None,
        support_type: str = "loan",
    ) -> float:
        if not getattr(self, "central_bank_support_enabled", True):
            return 0.0
        if bank_idx <= 0 or bank_idx >= len(self.banks):
            return 0.0
        amount = float(amount)
        if amount <= 1e-9:
            return 0.0
        bank = self.banks[bank_idx]
        if not bank.get("is_active", True):
            return 0.0
        is_settlement_backstop = (
            str(kind).removeprefix("policy_") == "settlement_backstop"
        )
        if not is_settlement_backstop:
            support_key = (int(bank_idx), str(budget_bucket))
            last_support_step = self.policy_support_last_step_by_bank_kind.get(support_key)
            if last_support_step is not None:
                cooldown = int(getattr(self, "policy_support_cooldown_steps", 0))
                if int(step) - int(last_support_step) < cooldown:
                    return 0.0
        total_lia = (
            float(bank.get("current_liabilities", 0.0))
            + float(bank.get("interbank_liabilities", 0.0))
        )
        per_bank_cap = per_bank_cap_share * total_lia
        if balance_key:
            room = max(0.0, per_bank_cap - float(bank.get(balance_key, 0.0)))
        else:
            room = max(0.0, per_bank_cap)
        if budget_bucket == "solvency":
            budget_room = min(
                float(getattr(self, "solvency_support_step_remaining", 0.0)),
                float(getattr(self, "solvency_support_remaining_budget", 0.0)),
            )
        else:
            budget_room = min(
                float(getattr(self, "cb_step_budget_remaining", 0.0)),
                float(getattr(self, "cb_remaining_budget", 0.0)),
            )
        support_type = str(support_type).lower()
        if support_type == "loan":
            outstanding = sum(float(loan.get("principal", 0.0)) for loan in self.policy_support_book)
            book_room = max(0.0, float(getattr(self, "policy_support_book_outstanding_limit", 0.0)) - outstanding)
            budget_room = min(budget_room, book_room)
        amount = min(amount, room, budget_room)
        if amount <= 1e-9:
            return 0.0
        bank["liquid_assets"] = float(bank.get("liquid_assets", 0.0)) + amount
        if support_type == "loan":
            bank["current_liabilities"] = float(bank.get("current_liabilities", 0.0)) + amount
            if balance_key:
                bank[balance_key] = float(bank.get(balance_key, 0.0)) + amount
        elif support_type == "capital":
            bank["core_capital"] = float(bank.get("core_capital", 0.0)) + amount
            bank["policy_capital_buffer"] = float(bank.get("policy_capital_buffer", 0.0)) + amount
        else:
            return 0.0
        if budget_bucket == "solvency":
            self.solvency_support_remaining_budget = max(0.0, float(self.solvency_support_remaining_budget) - amount)
            self.solvency_support_step_remaining = max(0.0, float(self.solvency_support_step_remaining) - amount)
        else:
            self.cb_remaining_budget = max(0.0, float(self.cb_remaining_budget) - amount)
            self.cb_step_budget_remaining = max(0.0, float(self.cb_step_budget_remaining) - amount)
        if not is_settlement_backstop:
            support_key = (int(bank_idx), str(budget_bucket))
            self.policy_support_last_step_by_bank_kind[support_key] = int(step)
        self.cb_total_injected = float(getattr(self, "cb_total_injected", 0.0)) + amount
        if support_type == "loan":
            self.policy_support_book.append({
                "bank_idx": bank_idx,
                "principal": amount,
                "rate": float(rate),
                "created_step": int(step),
                "maturity_step": int(step) + max(1, int(tenor)),
                "kind": f"policy_{kind}",
                "budget_bucket": budget_bucket,
                "balance_key": balance_key or "cb_policy_balance",
                "capital_like": False,
            })
        self.policy_event_log.append({
            "step": int(step),
            "event_type": "issuance",
            "bank_idx": bank_idx,
            "bank_name": bank.get("name", f"Bank{bank_idx}"),
            "policy_kind": f"policy_{kind}",
            "budget_bucket": budget_bucket,
            "support_type": support_type,
            "amount": amount,
            "rate": float(rate),
            "tenor": int(max(1, int(tenor))),
            "capital_like": bool(support_type == "capital"),
            "remaining_liquidity_budget": float(self.cb_remaining_budget),
            "remaining_solvency_budget": float(self.solvency_support_remaining_budget),
        })
        return amount

    def _issue_central_bank_liquidity_support(self, bank_idx: int, amount: float, step: int, rate: float, tenor: int = 2, kind: str = "slf") -> float:
        return self._issue_policy_support(
            bank_idx,
            amount,
            step,
            rate,
            tenor,
            kind,
            per_bank_cap_share=self.cb_max_support_share,
            budget_bucket="liquidity",
            balance_key="cb_policy_balance",
            support_type="loan",
        )

    def _issue_central_bank_solvency_support(self, bank_idx: int, amount: float, step: int, rate: float, tenor: int = 4, kind: str = "capital_support") -> float:
        return self._issue_policy_support(
            bank_idx,
            amount,
            step,
            rate,
            tenor,
            kind,
            per_bank_cap_share=self.solvency_support_max_share,
            budget_bucket="solvency",
            balance_key=None,
            support_type="capital",
        )

    def _apply_central_bank_policy(self, action: CentralBankPolicyAction, step: int) -> None:
        """按 action 更新走廊与准备金；流动性/资本支持在合格银行集合上依规则排序后按步预算配给。"""
        if not getattr(self, "policy_enabled", True):
            return
        desired_rate = float(np.clip(action.policy_rate, self.policy_rate_floor, self.policy_rate_ceiling))
        current_rate = float(getattr(self, "base_rate", desired_rate))
        interval = max(1, int(getattr(self, "policy_rate_decision_interval", 1)))
        should_reprice = (int(step) - int(getattr(self, "last_policy_rate_update_step", -10**9))) >= interval
        delta = desired_rate - current_rate
        if should_reprice and abs(delta) >= float(getattr(self, "policy_rate_change_threshold", 0.0)):
            cap = max(0.0, float(getattr(self, "policy_rate_max_step_change", 1.0)))
            move = float(np.clip(delta, -cap, cap))
            current_rate = float(np.clip(current_rate + move, self.policy_rate_floor, self.policy_rate_ceiling))
            self.last_policy_rate_update_step = int(step)
        self.base_rate = current_rate
        self.long_term_rate = max(self.base_rate + DAILY_LONG_RATE_SPREAD_BULL[0], self.long_term_rate)
        reserve_target = float(np.clip(action.reserve_requirement, 0.0, self.normal_reserve_requirement))
        self.reserve_buffer[:] = reserve_target
        if len(self.reserve_buffer) > 0:
            self.reserve_buffer[0] = 0.0
        self.central_corridor = CentralBankCorridor(
            deposit_rate=max(0.0, self.base_rate - DAILY_CB_DEPOSIT_SPREAD),
            lending_rate=self.base_rate + float(action.facility_spread),
            base_rate=self.base_rate,
        )

        support_total = 0.0
        supported_banks = 0
        # Solvency support is applied same-period before EN clearing.
        # Liquidity support for payment gaps is provided inside EN settlement only.
        facility_rate = self.base_rate + float(action.facility_spread)
        if bool(getattr(self, "policy_pre_trade_liquidity_window", False)):
            raise RuntimeError(
                "policy_pre_trade_liquidity_window is forbidden; "
                "liquidity support must disburse next period only."
            )

        self.last_policy_action = action
        self.last_policy_note = action.note
        self.policy_history.append({
            "step": int(step),
            "systemic_risk": float(getattr(self, "last_systemic_risk", 0.0)),
            "policy_rate": self.base_rate,
            "reserve_requirement": reserve_target,
            "support_total": support_total,
            "supported_banks": supported_banks,
            "solvency_support_total": 0.0,
            "solvency_supported_banks": 0,
            "disbursed_liquidity_support": 0.0,
            "disbursed_solvency_support": 0.0,
            "remaining_budget": float(self.cb_remaining_budget),
            "step_budget_remaining": float(self.cb_step_budget_remaining),
            "solvency_remaining_budget": float(self.solvency_support_remaining_budget),
            "solvency_step_remaining": float(self.solvency_support_step_remaining),
            "note": action.note,
        })
        # Excel export deferred to end of simulate_step.


    def _disburse_pending_policy_support(self, step: int) -> dict:
        """Disburse liquidity support queued after t-1 EN settlement (t start)."""
        out = {"liquidity": 0.0, "solvency": 0.0, "n": 0}
        queue = list(getattr(self, "pending_policy_support", []) or [])
        self.pending_policy_support = []
        if not getattr(self, "policy_support_enabled", True):
            return out
        if not getattr(self, "central_bank_support_enabled", True):
            return out
        for item in queue:
            bank_idx = int(item.get("bank_idx", -1))
            amount = float(item.get("amount", 0.0))
            if bank_idx <= 0 or amount <= 1e-9:
                continue
            if bank_idx >= len(self.banks):
                continue
            bank = self.banks[bank_idx]
            if not bank.get("is_active", True):
                continue
            kind = str(item.get("kind", "liquidity"))
            # Capital support is same-period pre-EN; ignore legacy deferred solvency items.
            if kind == "solvency":
                continue
            rate = float(item.get("rate", getattr(self, "base_rate", 0.0)))
            injected = self._issue_central_bank_liquidity_support(
                bank_idx,
                amount,
                int(step),
                rate=rate,
                tenor=int(item.get("tenor", getattr(self, "cb_loan_tenor", 2))),
                kind="liquidity_deferred",
            )
            out["liquidity"] += float(injected)
            if injected > 1e-9:
                out["n"] += 1
        return out

    def _queue_policy_support_from_stress(self, step: int) -> dict:
        """
        After EN settlement: queue *liquidity* support for t+1 only.
        Capital support is applied once after projects, before equity cascade.
        """
        queued = {"liquidity": 0.0, "solvency": 0.0, "n_liq": 0, "n_sol": 0}
        if not getattr(self, "policy_support_enabled", True):
            return queued
        if not getattr(self, "policy_enabled", True):
            return queued

        pending = list(getattr(self, "pending_policy_support", []) or [])
        pending = [x for x in pending if int(x.get("queued_step", -1)) != int(step)]
        pending = [x for x in pending if str(x.get("kind", "liquidity")) != "solvency"]

        facility_spread = DAILY_FACILITY_SPREAD_HOLD
        action = getattr(self, "last_policy_action", None)
        if action is not None:
            facility_spread = float(getattr(action, "facility_spread", facility_spread))
        facility_rate = float(getattr(self, "base_rate", 0.0)) + float(facility_spread)
        reserve_target = float(getattr(self, "normal_reserve_requirement", 0.02))
        if len(getattr(self, "reserve_buffer", [])) > 1:
            reserve_target = float(self.reserve_buffer[1])

        for i in range(1, len(self.banks)):
            bank = self.banks[i]
            if not self._is_liquidity_support_target(bank):
                continue
            if self._bank_equity(bank) <= 0.0:
                continue
            current_lia = float(bank.get("current_liabilities", 0.0))
            interbank_lia = float(bank.get("interbank_liabilities", 0.0))
            total_lia = current_lia + interbank_lia
            liq = float(bank.get("liquid_assets", 0.0))
            outflow_rate = float(bank.get("outflow_rate", 0.4))
            reserve_need = reserve_target * current_lia
            stress_outflow = total_lia * outflow_rate
            required_liq = max(reserve_need, stress_outflow)
            gap = max(0.0, required_liq - liq)
            if gap <= 1e-9:
                continue
            liq_ratio = float(getattr(action, "liquidity_support_ratio", 0.04)) if action is not None else 0.04
            cap = min(float(getattr(self, "cb_max_support_share", 0.25)), liq_ratio) * max(total_lia, 1e-9)
            amount = min(gap * 1.05, cap)
            if amount <= 1e-9:
                continue
            pending.append({
                "bank_idx": int(i),
                "amount": float(amount),
                "kind": "liquidity",
                "rate": float(facility_rate),
                "tenor": int(getattr(self, "cb_loan_tenor", 2)),
                "queued_step": int(step),
            })
            queued["liquidity"] += float(amount)
            queued["n_liq"] += 1

        self.pending_policy_support = pending
        self.last_queued_support = dict(queued)
        if self.policy_history:
            self.policy_history[-1]["queued_liquidity_support"] = float(queued["liquidity"])
            self.policy_history[-1]["queued_solvency_support"] = 0.0
        return queued


    def _resolve_bank_defaults_batch(
        self, failed_indices, step: int, reason: str = ""
    ) -> list[int]:
        """Two-phase batch absorbing default on a shared claim snapshot."""
        from interbank_resolution import settle_absorbing_defaults_batch

        idxs = sorted(
            {
                int(i)
                for i in (failed_indices or [])
                if 0 < int(i) < len(self.banks)
                and not self.banks[int(i)].get("absorbing_default")
                and not self.banks[int(i)].get("balance_sheet_frozen")
            }
        )
        if not idxs:
            return []

        # Freeze-time capital gap snapshot for CGR (before estate clearing zeros books).
        car_cut = float(getattr(self, "car_cutoff", 0.08))
        for bank_idx in idxs:
            bank = self.banks[bank_idx]
            failure_rwa = float(regulatory_rwa(bank))
            bank["rwa_at_failure"] = failure_rwa
            bank["required_capital_at_failure"] = car_cut * failure_rwa

        lgd = float(np.clip(getattr(self, "interbank_lgd", 0.4), 0.0, 1.0))
        book = getattr(self, "contract_book", None)

        def _make_estate_contract(**kwargs):
            cid = book._new_id() if book is not None else f"estate_{step}_{kwargs.get('borrower_idx', 0)}"
            return Contract(contract_id=cid, **kwargs)

        summaries = settle_absorbing_defaults_batch(
            banks=self.banks,
            book=book,
            exposure_matrix=getattr(self, "exposure_matrix", None),
            failed_indices=idxs,
            step=int(step),
            reason=str(reason),
            lgd=lgd,
            make_contract=_make_estate_contract,
            effective_notional=effective_notional,
        )
        by_idx = {int(s["bank_idx"]): s for s in summaries}
        # Primary metric: debtor-default creditor losses only (not estate transfer discount).
        step_writeoff = float(sum(
            float(s.get("creditor_writeoff", s.get("writeoff", 0.0)) or 0.0)
            for s in summaries
        ))
        step_transfer_discount = float(sum(
            float(s.get("estate_transfer_discount", 0.0) or 0.0)
            for s in summaries
        ))
        self.last_interbank_writeoff = float(
            getattr(self, "last_interbank_writeoff", 0.0) + step_writeoff
        )
        self.cumulative_interbank_writeoff = float(
            getattr(self, "cumulative_interbank_writeoff", 0.0) + step_writeoff
        )
        from bank_econ_shared import note_screening_loss
        note_screening_loss(self, "interbank_writeoff_loss", step_writeoff)
        self.last_estate_transfer_discount = float(
            getattr(self, "last_estate_transfer_discount", 0.0) + step_transfer_discount
        )
        self.cumulative_estate_transfer_discount = float(
            getattr(self, "cumulative_estate_transfer_discount", 0.0) + step_transfer_discount
        )

        failed_set = set(idxs)
        self.pending_policy_support = [
            x for x in (getattr(self, "pending_policy_support", []) or [])
            if int(x.get("bank_idx", -1)) not in failed_set
        ]
        self.policy_support_book = [
            loan
            for loan in (getattr(self, "policy_support_book", []) or [])
            if int(loan.get("bank_idx", -1)) not in failed_set
        ]

        n = int(getattr(self, "num_banks", len(self.banks)))
        if book is not None:
            update_bank_states_from_contract_book(self.banks, book, n, int(step))

        for bank_idx in idxs:
            bank = self.banks[bank_idx]
            summary = by_idx.get(bank_idx) or {
                "recovery": 0.0,
                "writeoff": 0.0,
                "estate_claims_transferred": 0.0,
                "transfer_consideration": 0.0,
                "contracts_removed": 0,
                "affected_counterparties": [],
            }
            bank["interbank_assets"] = 0.0
            bank["interbank_liabilities"] = 0.0
            equity = self._bank_equity(bank)
            bank["core_capital"] = max(0.0, float(equity))
            tot_lia = _bank_total_liabilities(bank)
            bank["solvency_ratio"] = float(equity) / (tot_lia + 1e-9)
            proj_amt = float(bank.get("investment", {}).get("projects", {}).get("amount", 0.0))
            bank["capital_adequacy_ratio"] = self._safe_car_value(
                bank["core_capital"], 0.0, proj_amt, tot_lia
            )
            bank["liquidity_coverage_ratio"] = self._safe_lcr_value(
                bank["liquid_assets"],
                bank["current_liabilities"],
                bank.get("outflow_rate", 0.4),
                interbank_liabilities=0.0,
            )
            bank["leverage_ratio"] = bank["core_capital"] / (
                float(bank.get("liquid_assets", 0.0)) + 1e-9
            )
            bank["total_assets"] = _bank_total_assets(bank)

            recovery_total = float(summary.get("recovery", 0.0))
            writeoff_total = float(summary.get("writeoff", 0.0))
            bank["is_active"] = False
            bank["failure_step"] = int(step)
            bank["failure_reason"] = str(reason)
            from bank_econ_shared import note_first_failure
            note_first_failure(self, bank_idx, step, reason)
            bank["absorbing_default"] = True
            bank["balance_sheet_frozen"] = True
            bank["defaulted"] = True
            bank["default_recovery"] = recovery_total
            bank["default_writeoff"] = writeoff_total
            bank["frozen_balance_sheet"] = {
                "liquid_assets": float(bank.get("liquid_assets", 0.0)),
                "core_capital": float(bank.get("core_capital", 0.0)),
                "current_liabilities": float(bank.get("current_liabilities", 0.0)),
                "interbank_assets": 0.0,
                "interbank_liabilities": 0.0,
                "projects_amount": proj_amt,
                "solvency_ratio": float(bank.get("solvency_ratio", 0.0)),
                "capital_adequacy_ratio": float(bank.get("capital_adequacy_ratio", 0.0)),
                "liquidity_coverage_ratio": float(bank.get("liquidity_coverage_ratio", 0.0)),
                "leverage_ratio": float(bank.get("leverage_ratio", 0.0)),
                "total_assets": float(bank.get("total_assets", 0.0)),
                "recovery": recovery_total,
                "writeoff": writeoff_total,
                "estate_claims_transferred": float(summary.get("estate_claims_transferred", 0.0)),
                "transfer_consideration": float(summary.get("transfer_consideration", 0.0)),
            }
            self.policy_event_log.append({
                "step": int(step),
                "event_type": "default_resolve",
                "bank_idx": bank_idx,
                "bank_name": bank.get("name", f"Bank{bank_idx}"),
                "reason": str(reason),
                "lgd": float(lgd),
                "recovery": recovery_total,
                "writeoff": writeoff_total,
                "estate_claims_transferred": float(summary.get("estate_claims_transferred", 0.0)),
                "transfer_consideration": float(summary.get("transfer_consideration", 0.0)),
                "contracts_removed": int(summary.get("contracts_removed", 0)),
                "affected_counterparties": ",".join(
                    str(int(x)) for x in summary.get("affected_counterparties", [])
                ),
            })
        return idxs

    def _resolve_bank_default(self, bank_idx: int, step: int, reason: str = "") -> None:
        """Single-bank wrapper around two-phase batch resolution."""
        self._resolve_bank_defaults_batch([int(bank_idx)], step, reason=reason)

    def _apply_solvency_support(self, step: int) -> dict:
        """
        Same-period capital injection *before* EN clearing.
        Increases core_capital and liquid_assets; does not add ordinary liabilities.
        """
        out = {"solvency_support_total": 0.0, "solvency_supported_banks": 0}
        if not getattr(self, "policy_enabled", True):
            return out
        if not getattr(self, "solvency_support_enabled", False):
            return out
        if not getattr(self, "policy_support_enabled", True):
            return out

        support_rate = self.base_rate + float(self.solvency_support_spread)
        solvency_support_total = 0.0
        solvency_supported_banks = 0
        for i in range(1, len(self.banks)):
            bank = self.banks[i]
            if not self._is_capital_support_target(bank):
                continue
            current_lia = float(bank.get("current_liabilities", 0.0))
            interbank_lia = float(bank.get("interbank_liabilities", 0.0))
            total_lia = current_lia + interbank_lia
            rwa = _bank_regulatory_rwa(bank)
            # Injection raises liquid assets and accounting equity one-for-one.
            equity = float(self._bank_equity(bank))
            capital_gap = max(
                0.0,
                float(self.solvency_support_car_trigger) * rwa - equity,
            )
            if capital_gap <= 1e-9:
                continue
            per_bank_cap = float(self.solvency_support_max_share) * max(total_lia, 1e-9)
            amount = min(
                capital_gap,
                per_bank_cap,
                float(getattr(self, "solvency_support_step_remaining", 0.0)),
                float(getattr(self, "solvency_support_remaining_budget", 0.0)),
            )
            if amount <= 1e-9:
                continue
            injected = self._issue_central_bank_solvency_support(
                i,
                amount,
                step,
                rate=support_rate,
                tenor=1,
                kind="capital_subsidy",
            )
            if injected > 0.0:
                bank["risk_appetite"] = float(bank.get("risk_appetite", 0.5)) * 0.9
                solvency_support_total += injected
                solvency_supported_banks += 1

        out["solvency_support_total"] = float(solvency_support_total)
        out["solvency_supported_banks"] = int(solvency_supported_banks)
        if self.policy_history:
            self.policy_history[-1]["solvency_support_total"] = float(solvency_support_total)
            self.policy_history[-1]["solvency_supported_banks"] = int(solvency_supported_banks)
            self.policy_history[-1]["solvency_remaining_budget"] = float(
                self.solvency_support_remaining_budget
            )
            self.policy_history[-1]["solvency_step_remaining"] = float(
                self.solvency_support_step_remaining
            )
        return out

    def _refresh_regulatory_metrics(self, step: int) -> None:
        """Recompute IB stocks / CAR / LCR / equity proxies from the contract book."""
        banks = self.banks
        n = self.num_banks
        book = self.contract_book
        update_bank_states_from_contract_book(banks, book, n, int(step))
        for b in banks:
            if b.get("absorbing_default") or b.get("balance_sheet_frozen"):
                continue
            equity = self._bank_equity(b)
            b["core_capital"] = max(0.0, float(equity))
            tot_lia = _bank_total_liabilities(b)
            cap_ratio = float(equity) / (tot_lia + 1e-9)
            b["solvency_ratio"] = cap_ratio
            proj_amt = b["investment"]["projects"]["amount"]
            b["capital_adequacy_ratio"] = self._safe_car_value(
                b["core_capital"], b["interbank_assets"], proj_amt, tot_lia
            )
            b["liquidity_coverage_ratio"] = self._safe_lcr_value(
                b["liquid_assets"],
                b["current_liabilities"],
                b["outflow_rate"],
                interbank_liabilities=float(b.get("interbank_liabilities", 0.0)),
            )
            b["leverage_ratio"] = b["core_capital"] / (
                b["liquid_assets"] + b["interbank_assets"] + 1e-9
            )
            b["total_assets"] = _bank_total_assets(b)

    def _classify_equity_defaults(self) -> list[int]:
        """Identify negative-equity banks; actual write-off is done by _resolve_bank_default."""
        equity_defaults: list[int] = []
        n = self.num_banks
        for i in range(n):
            if self.bank_types[i] == "central":
                continue
            b = self.banks[i]
            if not b.get("is_active", True):
                continue
            equity = self._bank_equity(b)
            if equity < 0:
                equity_defaults.append(int(i))
        return equity_defaults

    def _cascade_negative_equity_defaults(
        self, step: int, *, reason: str = "negative_equity"
    ) -> list[int]:
        """Refresh → classify → resolve until no new negative-equity failures."""
        resolved: list[int] = []
        max_rounds = max(1, int(getattr(self, "num_banks", 1)))
        for _ in range(max_rounds):
            self._refresh_regulatory_metrics(int(step))
            newly = self._classify_equity_defaults()
            if not newly:
                break
            self._resolve_bank_defaults_batch(newly, int(step), reason=reason)
            resolved.extend(int(i) for i in newly)
        return sorted(set(resolved))

    def solve_clearing(self, L: np.ndarray, e: np.ndarray) -> np.ndarray:
        """
        Eisenberg–Noe 清算：给定净头寸矩阵 L（可正可负，主对角为0）与外生 endowment e，
        返回清算支付向量 p。
        """
        n = L.shape[0]
        Lbar = np.maximum(-L, 0.0)
        np.fill_diagonal(Lbar, 0.0)
        p_bar = Lbar.sum(axis=1)
        Pi = np.divide(Lbar, p_bar[:, None], out=np.zeros_like(Lbar), where=(p_bar[:, None] > 0))

        p = p_bar.copy()
        for _ in range(self.clear_max_iter):
            p_new = np.minimum(p_bar, Pi.T @ p + e)
            if np.max(np.abs(p_new - p)) < self.clear_tol:
                p = p_new
                break
            p = p_new
        return p

    def calculate_losses(self, i: int, failed: list) -> float:
        """
        只计算银行 i 因对失败方的同业暴露造成的损失。
        项目违约已在 update_project_book() 入账，这里不重复。
        """
        caps = np.array([b['core_capital'] for b in self.banks], dtype=float)
        liqs = np.array([b['liquid_assets'] for b in self.banks], dtype=float)
        L = self.exposure_matrix.astype(float)
        e = caps + liqs

        p = self.solve_clearing(L, e)

        Lbar  = np.maximum(-L, 0.0)
        p_bar = Lbar.sum(axis=1) + 1e-9

        interbank_loss = 0.0
        for j in failed:
            claim_ij  = Lbar[j, i]
            recovered = p[j] * (claim_ij / p_bar[j])
            interbank_loss += max(0.0, claim_ij - recovered)

        if failed:
            self.banks[i]['current_liabilities'] *= (1 + 0.02 * len(failed))

        return float(interbank_loss)
    
    def _lender_supply_amount(self, i, LCR_TARGET=1.0, ALPHA_STRESS_LENDER=1.0):
        b = self.banks[i]
        liq = float(b.get("liquid_assets", 0.0))
        lia = float(b.get("current_liabilities", 0.0))
        req = float(self.reserve_buffer[i] * lia)

        outflow_target = lia * float(b.get("outflow_rate", 0.2))
        target_liq = max(req, LCR_TARGET * ALPHA_STRESS_LENDER * outflow_target)

        avail = max(0.0, liq - target_liq)
        phi = float(b.get("risk_appetite", 0.5))
        avail *= (0.6 + 0.4 * phi)
        return float(avail)


    def assign_roles_by_risk(self, car_cutoff: float = 0.08, lcr_cutoff: float = DAILY_INTERBANK_ROLE_LCR_CUTOFF):
        """
        根据风险指标给银行分配角色：
        +1 = lender, -1 = borrower, 0 = central/不参与撮合
        """
        n = self.num_banks
        roles = np.ones(n, dtype=int)

        # 0号固定为央行（不参与 lender/borrower）
        roles[0] = 0

        for i, b in enumerate(self.banks):
            if i == 0:
                continue

            # 不活跃：不参与撮合（否则会制造假 borrower/lender）
            if not b.get("is_active", True):
                roles[i] = 0
                continue

            car = regulatory_car(b)
            lcr  = float(b.get("liquidity_coverage_ratio", 1.0))
            solv = float(b.get("solvency_ratio", 1.0))

            need = 0.0

            # CAR 低：不适合放贷
            car_low = car_below_threshold(car, car_cutoff)

            if lcr < lcr_cutoff:
                need += 0.5
            if solv < 1.0:
                need += 0.5

            # ===== 流动性缺口判断 =====
            liq = float(b.get("liquid_assets", 0.0))
            lia = float(b.get("current_liabilities", 0.0))
            req = float(self.reserve_buffer[i] * lia)
            outflow_target = lia * float(b.get("outflow_rate", 0.2))
            target_liq = max(req, outflow_target)

            if liq < target_liq:
                need += 1.0

            if car_low:
                # CAR below policy threshold: cannot be lender
                roles[i] = -1 if (lcr < lcr_cutoff or solv < 1.0 or liq < target_liq) else 0
            elif need >= 0.55:
                roles[i] = -1
            else:
                roles[i] = +1


        # 避免全是 lender 或全是 borrower（排除央行）
        caps = np.array([float(b.get("core_capital", 0.0)) for b in self.banks], dtype=float)
        idxs = np.array(
            [i for i in range(1, n) if self.banks[i].get("is_active", True)],
            dtype=int
        )
        if idxs.size == 0:
            return roles  # 全死了/只剩央行

        k = max(1, idxs.size // 4)

        if np.all(roles[idxs] == +1):
            weakest = idxs[np.argsort(caps[idxs])[:k]]
            roles[weakest] = -1
        elif np.all(roles[idxs] == -1):
            strongest = idxs[np.argsort(caps[idxs])[-k:]]
            for r in strongest:
                if not car_below_threshold(regulatory_car(self.banks[r]), car_cutoff):
                    roles[r] = +1


        # ===== PATCH 1: 供给侧兜底（使用“真实供给公式”，与撮合一致）=====
        avail_liq = np.full(n, -np.inf, dtype=float)
        for i in range(1, n):
            if not self.banks[i].get("is_active", True):
                continue
            avail_liq[i] = self._lender_supply_amount(
                i, LCR_TARGET=lcr_cutoff, ALPHA_STRESS_LENDER=1.0
            )

        lenders_now = np.where(roles == +1)[0]
        total_avail = float(np.maximum(avail_liq[lenders_now], 0.0).sum()) if lenders_now.size else 0.0

        if lenders_now.size < max(2, n // 10) or total_avail < 1e-6:
            K_force = max(2, n // 6)
            richest = np.argsort(avail_liq)[-K_force:]
            for r in richest:
                if r == 0:
                    continue
                if not np.isfinite(avail_liq[r]) or avail_liq[r] <= 1e-8:
                    continue
                if car_below_threshold(regulatory_car(self.banks[r]), car_cutoff):
                    continue
                roles[r] = +1

        # 再保险：央行永远是 0
        roles[0] = 0
        return roles



    def assign_roles(self, lender_pct: float = 0.45, borrower_pct: float = 0.45):
        """
        基于当前状态给每家银行分配当期“角色”：
        +1 = lender, -1 = borrower, 0 = central.
        """
        n = self.num_banks
        roles = np.zeros(n, dtype=int)

        solv = np.array(
            [(b['core_capital'] + b['liquid_assets']) / (b['current_liabilities'] + 1e-9)
             for b in self.banks], dtype=float
        )
        lcr_fallback = np.array([
            b.get('liquidity_coverage_ratio',
                  b['liquid_assets'] / (b.get('current_liabilities', 0.0) * b.get('outflow_rate', 0.4) + 1e-9))
            for b in self.banks
        ], dtype=float)
        score = 0.7 * solv + 0.3 * lcr_fallback

        order = np.argsort(score)
        pool  = [i for i in order if i != 0]

        forced_borrowers = [i for i in pool if getattr(self, 'borrowed_cash', np.zeros(n))[i] > 1e-8]

        k_b_target = max(1, int(np.floor(borrower_pct * len(pool))))
        k_l_target = max(1, int(np.floor(lender_pct  * len(pool))))

        borrowers = list(forced_borrowers)

        remaining = [i for i in pool if i not in borrowers]
        need_b = max(0, k_b_target - len(borrowers))
        borrowers += remaining[:need_b]
        remaining = remaining[need_b:]

        lenders = remaining[-k_l_target:] if len(remaining) >= k_l_target else remaining

        roles[borrowers] = -1
        roles[lenders]   = +1

        self.roles = roles
        return roles

    def assign_roles_balanced(self, frac_lenders: float = 0.5):
        """
        自由市场模式用的"流动性平衡型"角色分配。
        """
        n = self.num_banks
        roles = np.zeros(n, dtype=int)
        roles[0] = 0

        idxs = [i for i in range(1, n)]
        avail = []
        for i in idxs:
            liq = float(self.banks[i]['liquid_assets'])
            req = float(self.reserve_buffer[i] * self.banks[i]['current_liabilities'])
            avail.append(liq - req)
        avail = np.asarray(avail, dtype=float)

        order = np.argsort(avail)
        k_lenders = max(1, int(round(frac_lenders * len(idxs))))

        lenders_idx   = [idxs[i] for i in order[-k_lenders:]]
        borrowers_idx = [idxs[i] for i in order[:-k_lenders]]

        for i in lenders_idx:
            roles[i] = +1
        for i in borrowers_idx:
            roles[i] = -1

        return roles

    def _filter_intentions_for_matching(self, intentions):
        return intentions

    def _adjust_market_liquidity_shock(self, bank_idx: int, market_adjustment: float) -> float:
        return float(market_adjustment)

    def _lender_invest_frac(self, bank_idx: int, base_frac: float = 0.05) -> float:
        return float(base_frac)

    def _post_project_phase_hook(self, step: int) -> None:
        return None

    def _post_regulatory_refresh_hook(self, step: int) -> None:
        """After CAR/LCR refresh — danger-zone state & LCR repair."""
        return None

    def _history_extra_fields(self) -> dict:
        return {}

    def simulate_step(self, step):
        """
        单期顺序：
        0) 上期流动性救助到账（央行贷款）
        1) 据 t-1 SR 调利率/准备金（仅 Support ON）
        2) 偿还到期央行支持贷款
        3) 全部到期同业债务并入一个 EN 矩阵，只清算一次，再处理未付
           （OFF：清算前全部到期；清算中结算；任何未付→当期违约）
           （ON：当期分期+历史未付；未付并入 arrears；逾期次数仅统计）
        4) 刷新账面，仅排队下期流动性支持
        5) 市场/项目/冲击
        6) 每日撮合（CEN：全局最高成交利率 + all-or-nothing）
        7) 再刷新监管指标
        8) 最终违约判定（LGD 核销）
        9) 计算 SR_t 供下一期政策使用
        """
        try:
            self.current_step = int(step)
            self.last_interbank_writeoff = 0.0
            self.last_estate_transfer_discount = 0.0
            from bank_econ_shared import reset_screening_loss
            reset_screening_loss(self, totals=False, step=True)
            if getattr(self, "exposure_matrix", None) is not None:
                self.prev_exposure_matrix = self.exposure_matrix.copy()
            self._reset_policy_step_budget()
            # 0) disburse liquidity support queued after t-1 settlement
            disbursed = self._disburse_pending_policy_support(step)
            # 1) policy rates/reserves from last SR (gated by policy_enabled / Support ON)
            self._run_central_bank_policy_cycle(step)
            if self.policy_history:
                self.policy_history[-1]["disbursed_liquidity_support"] = float(disbursed.get("liquidity", 0.0))
                self.policy_history[-1]["disbursed_solvency_support"] = float(disbursed.get("solvency", 0.0))
            # 2) repay prior CB support
            self._settle_central_bank_loans(step)

            # 3) all due interbank claims → one EN matrix → one clearing (no pre-pay)
            # Capital support is deferred until after projects (not pre-EN).
            self._refresh_regulatory_metrics(step)
            failed = self._settle_due_interbank_contracts(step)
            self._resolve_bank_defaults_batch(failed, step, reason="en_settlement")
            contagion_en = self._cascade_negative_equity_defaults(
                step, reason="contagion_after_en"
            )
            # 4) refresh after settlement; queue liquidity support for t+1
            self._refresh_metrics_after_settle(step)
            self._queue_policy_support_from_stress(step)

            # 5) market environment, bank-level rates, shocks
            # When Support ON, CB policy already set base_rate; do not reset from initial_base_rate.
            if hasattr(self, "adjust_base_rate") and not getattr(self, "policy_enabled", False):
                self.adjust_base_rate()

            for _b in self.banks:
                if not _b.get("is_active", True):
                    _b.pop("pending_endowment", None)
                    continue
                if 'pending_endowment' in _b:
                    _b['liquid_assets'] += _b.pop('pending_endowment')

            self.market_duration += 1
            if self.market_duration >= self.market_duration_limit:
                self.prev_market_environment = self.market_environment
                self.market_environment = 'bull' if self.rng_environment.random() < 0.6 else 'bear'
                self.market_duration = 0
                self.market_duration_limit = int(self.rng_environment.integers(2, (5) + 1))

            n_banks = int(self.num_banks)
            if self.market_environment == 'bull':
                self.base_rate = max(DAILY_POLICY_RATE_FLOOR, self.base_rate + self.rng_environment.uniform(-0.00001, 0.00001))
                self.long_term_rate = self.base_rate + self.rng_environment.uniform(*DAILY_LONG_RATE_SPREAD_BULL)
                market_volatility = self.rng_environment.uniform(10, 20) / 50
                market_adjustment = self.rng_environment.uniform(*DAILY_MARKET_ADJUSTMENT_BULL)
                loan_lo, loan_hi = DAILY_LOAN_SPREAD_BULL
                out_lo, out_hi = 0.3, 0.5
            else:
                self.base_rate = min(DAILY_POLICY_RATE_CEILING, self.base_rate + self.rng_environment.uniform(0.0, 0.000015))
                self.long_term_rate = self.base_rate + self.rng_environment.uniform(*DAILY_LONG_RATE_SPREAD_BEAR)
                market_volatility = self.rng_environment.uniform(30, 50) / 50
                market_adjustment = self.rng_environment.uniform(*DAILY_MARKET_ADJUSTMENT_BEAR)
                loan_lo, loan_hi = DAILY_LOAN_SPREAD_BEAR
                out_lo, out_hi = 0.4, 0.6

            # Fixed N-slot draws: failed banks keep their RNG slot but skip application
            loan_spreads = self.rng_environment.uniform(loan_lo, loan_hi, size=n_banks)
            inv_spreads = self.rng_environment.uniform(
                DAILY_INVESTMENT_SPREAD_STEP[0], DAILY_INVESTMENT_SPREAD_STEP[1], size=n_banks
            )
            outflow_draws = self.rng_environment.uniform(out_lo, out_hi, size=n_banks)

            for i, bank in enumerate(self.banks):
                if not bank.get("is_active", True):
                    continue
                bank['market_volatility'] = market_volatility
                bank['loan_interest_rate'] = self.base_rate + float(loan_spreads[i])
                bank['investment_interest_rate'] = self.long_term_rate + float(inv_spreads[i])

                bank.setdefault('risk_appetite', 0.5)
                bank.setdefault('hurdle_rate', DAILY_HURDLE_RATE)
                bank.setdefault('pending_endowment', 0.0)

                apply_deposit_flow(bank, DAILY_LIABILITY_GROWTH)
                adj = self._adjust_market_liquidity_shock(i, market_adjustment)
                apply_deposit_flow(bank, adj)

                bank['outflow_rate'] = (
                    0.2 if bank['type'] == 'central' else float(outflow_draws[i])
                )

                if bank['liquid_assets'] < bank['current_liabilities'] * bank['outflow_rate']:
                    bank['risk_appetite'] *= 0.9

            rem = int(getattr(self, "project_pd_stress_remaining", 0) or 0)
            if rem > 0:
                self.project_pd_stress_remaining = rem - 1
                self.project_pd_stress_multiplier = DAILY_COMMON_PROJECT_PD_MULTIPLIER
            else:
                self.project_pd_stress_multiplier = 1.0
            self.last_common_liquidity_shock = False
            if self.rng_environment.random() < DAILY_COMMON_LIQUIDITY_SHOCK_PROB:
                from bank_econ_shared import apply_common_liquidity_run
                apply_common_liquidity_run(
                    self.banks,
                    multiplier=DAILY_COMMON_LIQUIDITY_OUTFLOW_MULTIPLIER,
                )
                # Fixed N-slot draws: failed banks keep their RNG slot.
                sensitivities = self.rng_environment.uniform(0.8, 1.2, size=n_banks)
                for i, bank in enumerate(self.banks):
                    if not bank.get("is_active", True):
                        continue
                    bank["outflow_rate"] = min(
                        0.95,
                        float(bank.get("outflow_rate", 0.4))
                        * DAILY_COMMON_LIQUIDITY_OUTFLOW_MULTIPLIER
                        * float(sensitivities[i]),
                    )
                self.project_pd_stress_multiplier = DAILY_COMMON_PROJECT_PD_MULTIPLIER
                self.project_pd_stress_remaining = int(DAILY_COMMON_PROJECT_PD_STRESS_DAYS)
                self.last_common_liquidity_shock = True

            if self.market_environment == 'bear' and self.rng_environment.random() < 0.1:
                for bank in self.banks:
                    if not bank.get("is_active", True):
                        continue
                    # Flight-to-deposit inflow: ΔA=ΔL, so it cannot lift CAR
                    # by gifting cash into core capital.
                    apply_deposit_flow(bank, 0.01)

            if (not getattr(self, "one_shot_default_done", False)) and (int(step) == 0):
                if self.rng_environment.random() < 0.03:
                    fail_bank = int(self.rng_environment.integers(1, (self.num_banks - 1) + 1))
                    self._resolve_bank_default(fail_bank, step, reason="one_shot")
                self.one_shot_default_done = True

            # Refresh accounting after shocks/liability growth before matching.
            self._refresh_regulatory_metrics(step)

            # 6) daily matching
            if getattr(self, "free_market", False):
                self.roles = self.assign_roles_balanced(frac_lenders=0.5)
            else:
                if hasattr(self, 'assign_roles_by_risk'):
                    self.roles = self.assign_roles_by_risk(
                        car_cutoff=getattr(self, "car_cutoff", 0.08),
                        lcr_cutoff=getattr(self, "lcr_cutoff", DAILY_INTERBANK_ROLE_LCR_CUTOFF),
                    )
                else:
                    self.roles = self.assign_roles()

            self._sparse_bipartite_update(self.roles)
            for i in range(self.num_banks):
                if self.roles[i] == +1 and self.banks[i].get("is_active", True):
                    frac = self._lender_invest_frac(i, 0.05)
                    if frac > 1e-6:
                        self.invest_free_cash_into_projects(i, invest_frac=frac)

            for i in range(self.num_banks):
                # Always tick project RNG slots (incl. failed banks) so paired seeds stay aligned
                self.allocate_borrowed_to_projects(i)
                self.update_project_book(i)
            self._post_project_phase_hook(step)

            # 7) refresh after trading
            self._refresh_regulatory_metrics(step)
            post_project_solvency = self._apply_solvency_support(step)
            if float(post_project_solvency.get("solvency_support_total", 0.0)) > 0.0:
                self._refresh_regulatory_metrics(step)
            if self.policy_history:
                self.policy_history[-1]["post_project_solvency_support"] = float(
                    post_project_solvency.get("solvency_support_total", 0.0)
                )
            for b in self.banks:
                if b.get("absorbing_default") or b.get("balance_sheet_frozen"):
                    continue
                b['capital_ratio_history'].append(float(b.get("solvency_ratio", 0.0)))
            self._post_regulatory_refresh_hook(step)

            # 8) final default classification with immediate LGD write-off (fixed point)
            equity_defaults = self._cascade_negative_equity_defaults(
                step, reason="negative_equity"
            )
            en_defs = list(getattr(self, "last_defaulted_banks", []) or [])
            self.last_defaulted_banks = sorted(
                set(en_defs + list(contagion_en) + list(equity_defaults))
            )

            # 9) SR_t for next-period policy
            n_b = self.num_banks
            risk = self.calculate_systemic_risk() if hasattr(self, "calculate_systemic_risk") else 0.0
            self._record_systemic_risk(risk)

            if getattr(self, "_validate_book_sr", False):
                sr_alt = decentralized_systemic_risk(
                    self.banks,
                    self.contract_book,
                    n_b,
                    int(step),
                )
                if not validate_decentralized_vs_baseline(float(risk), float(sr_alt), tol=0.15):
                    print(
                        f"[validate SR] step={step} calculate_systemic_risk={float(risk):.4f} "
                        f"book_formula={float(sr_alt):.4f}"
                    )

            if getattr(self, "record_history", False):
                self.simulation_history.append({
                    'step': step,
                    'systemic_risk': risk,
                    'raw_systemic_risk': float(getattr(self, "last_raw_systemic_risk", risk)),
                    'collapse_index': float(getattr(self, "last_collapse_index", risk)),
                    'policy_note': getattr(self, "last_policy_note", ""),
                    'exposure_matrix': self.exposure_matrix.copy(),
                    'bank_states': [deepcopy(b) for b in self.banks],
                    **self._history_extra_fields(),
                })
            if self.all_default_step is None:
                alive_noncentral = [
                    k for k in range(1, self.num_banks)
                    if self.banks[k].get("is_active", True)
                ]
                if len(alive_noncentral) == 0:
                    self.all_default_step = int(step)
                    print(f"[ALL DEFAULT] step={self.all_default_step} (all non-central banks defaulted)")
            self._update_network_stability(step, risk)
            self.maybe_save_network_snapshot(step, risk, tag="centralized", edge_quantile=0.0)
            if getattr(self, "export_policy_logs", True) and self.policy_history:
                export_policy_logs_excel(
                    self.policy_history,
                    self.policy_event_log,
                    POLICY_LOG_DIR,
                    self.initial_state_export_prefix,
                )
            return risk

        except Exception as e:
            print(f"Error in simulate_step: {e}")
            raise

    def _deterministic_rate_based_matching(
        self,
        lenders,
        borrowers,
        supply,
        demand,
        B,
        deg_init=None,
        r_min=None,
        r_max=None,
    ):
        """
        全局最高成交利率优先的集中式撮合（无 GNN / 无局部询价）。

        feasible:
            r_min[i] <= r_max[j]

        expected deal rate:
            r_ij = (r_min[i] + r_max[j]) / 2

        按借款人最高可成交利率依次处理：
            try_full_fill → commit；否则 rollback 后继续下一家。
        保留 all-or-nothing：不满额不成交，且不占用供给。
        """
        n = self.num_banks
        eps = 1e-8

        supply = np.asarray(supply, dtype=float).copy()
        demand = np.asarray(demand, dtype=float).copy()

        if deg_init is None:
            deg = np.zeros(n, dtype=int)
        else:
            deg = np.asarray(deg_init, dtype=int).copy()

        r_min = {} if r_min is None else r_min
        r_max = {} if r_max is None else r_max

        # 每个 borrower 的可行边：(deal_rate, lender_pos, lender_id)
        pairs_by_b = {bj: [] for bj in range(len(borrowers))}
        best_rate = {bj: -np.inf for bj in range(len(borrowers))}

        for li, i in enumerate(lenders):
            lender_min = float(
                r_min.get(
                    i,
                    self.banks[i].get(
                        "loan_interest_rate",
                        DAILY_BULL_BASE_RATE,
                    ),
                )
            )
            for bj, j in enumerate(borrowers):
                if i == j:
                    continue
                borrower_max = float(
                    r_max.get(
                        j,
                        self.banks[j].get(
                            "loan_interest_rate",
                            DAILY_BULL_BASE_RATE,
                        ) + DAILY_RFQ_QUOTE_SPREAD,
                    )
                )
                if lender_min > borrower_max + eps:
                    continue
                deal_rate = 0.5 * (lender_min + borrower_max)
                pairs_by_b[bj].append((deal_rate, li, i))
                if deal_rate > best_rate[bj]:
                    best_rate[bj] = deal_rate

        # 按借款人最高可成交利率全局排序；逐个 try-fill / commit / rollback
        borrower_order = sorted(
            [bj for bj in range(len(borrowers)) if float(demand[bj]) > eps],
            key=lambda bj: best_rate.get(bj, -np.inf),
            reverse=True,
        )

        plan = []
        for bj in borrower_order:
            j = borrowers[bj]
            need = float(demand[bj])
            if need <= eps:
                continue
            if deg[j] >= self.max_degree:
                continue

            candidates = sorted(
                pairs_by_b.get(bj, []),
                key=lambda x: x[0],
                reverse=True,
            )
            remaining = need
            tmp_alloc = []

            for deal_rate, li, i in candidates:
                if supply[li] <= eps:
                    continue
                if deg[i] >= self.max_degree:
                    continue
                if deg[j] + len(tmp_alloc) >= self.max_degree:
                    break

                amount = min(float(B), float(supply[li]), remaining)
                room = ibl_borrowing_room(
                    self.banks[j],
                    asset_lambda=float(
                        getattr(
                            self,
                            "ibl_cap_asset_lambda",
                            DEFAULT_IBL_CAP_ASSET_LAMBDA,
                        )
                    ),
                    extra_new_borrowing=need - remaining,
                )
                amount = min(amount, room)
                if amount <= eps:
                    continue

                tmp_alloc.append((i, j, float(amount), li))
                remaining -= amount
                if remaining <= eps:
                    break

            # all-or-nothing：满额才 commit；否则整单 rollback（不占用 supply）
            if remaining <= eps and tmp_alloc:
                for i, j2, amount, li in tmp_alloc:
                    plan.append((i, j2, float(amount)))
                    supply[li] -= amount
                    deg[i] += 1
                deg[j] += len(tmp_alloc)
                demand[bj] = 0.0
            # else: rollback — tmp_alloc discarded, supply untouched

        return plan


    def _is_auction_day(self, step: int | None = None) -> bool:
        """固定周期批量拍卖日：每 centralized_cycle_length 期成交一次。"""
        L = max(1, int(getattr(self, "centralized_cycle_length", 1)))
        s = int(self.current_step if step is None else step)
        return ((s + 1) % L) == 0

    def _queue_central_orders_from_intentions(self, intentions) -> None:
        """每家银行只保留最新订单（覆盖，不跨日累加数量）。"""
        book = getattr(self, "central_order_book", None)
        if not isinstance(book, dict):
            self.central_order_book = {}
            book = self.central_order_book
        step = int(getattr(self, "current_step", 0))
        for it in intentions:
            i = int(it.bank_idx)
            book[i] = {
                "bank_idx": i,
                "role": str(it.role).lower(),
                "quantity": float(it.quantity),
                "reserve_bid": float(getattr(it, "reserve_bid", 0.0)),
                "reserve_ask": float(getattr(it, "reserve_ask", 0.0)),
                "step_submitted": step,
            }

    def _reconfirm_central_order_quantities(self, intentions) -> dict[int, dict]:
        """
        拍卖日按当前状态重新确认可执行数量：
        executable = min(最新申报量, 当前可用供给/需求)。
        """
        eps = 1e-9
        today = {int(it.bank_idx): it for it in intentions}
        confirmed: dict[int, dict] = {}
        book = getattr(self, "central_order_book", {}) or {}
        if not isinstance(book, dict):
            # 兼容旧 list 结构
            tmp = {}
            for o in book:
                tmp[int(o["bank_idx"])] = o
            book = tmp
        for i, o in book.items():
            i = int(i)
            if i <= 0 or i >= len(self.banks):
                continue
            if not self.banks[i].get("is_active", True):
                continue
            role = str(o.get("role", "")).lower()
            ordered_q = max(0.0, float(o.get("quantity", 0.0)))
            if ordered_q <= eps:
                continue
            it = today.get(i)
            if it is not None and str(it.role).lower() == role:
                current_q = max(0.0, float(it.quantity))
                bid = float(getattr(it, "reserve_bid", o.get("reserve_bid", self.base_rate)))
                ask = float(getattr(it, "reserve_ask", o.get("reserve_ask", self.base_rate)))
            elif role == "lender":
                current_q = max(0.0, float(self._lender_supply_amount(i)))
                bid = float(o.get("reserve_bid", self.base_rate))
                ask = float(o.get("reserve_ask", self.base_rate))
            else:
                # 角色已变或当日无对应意向：不可再按旧需求成交
                current_q = 0.0
                bid = float(o.get("reserve_bid", self.base_rate))
                ask = float(o.get("reserve_ask", self.base_rate))
            q = min(ordered_q, current_q)
            if q <= eps:
                continue
            confirmed[i] = {
                "bank_idx": i,
                "role": role,
                "quantity": float(q),
                "reserve_bid": float(bid),
                "reserve_ask": float(ask),
                "step_submitted": int(o.get("step_submitted", getattr(self, "current_step", 0))),
            }
        return confirmed

    def _aggregate_central_order_book(self, confirmed: dict[int, dict] | None = None):
        """把已确认的最新订单转为撮合输入（每家银行至多一侧）。"""
        orders = confirmed if confirmed is not None else getattr(self, "central_order_book", {}) or {}
        if not isinstance(orders, dict):
            orders = {int(o["bank_idx"]): o for o in orders}
        lenders, borrowers = [], []
        supply, demand = [], []
        r_min, r_max = {}, {}
        for i in sorted(orders):
            o = orders[i]
            q = float(o.get("quantity", 0.0))
            if q <= 1e-12:
                continue
            role = str(o.get("role", "")).lower()
            if role == "lender":
                lenders.append(int(i))
                supply.append(q)
                r_min[int(i)] = float(o.get("reserve_bid", self.base_rate))
            else:
                borrowers.append(int(i))
                demand.append(q)
                r_max[int(i)] = float(o.get("reserve_ask", self.base_rate))
        return lenders, borrowers, supply, demand, r_min, r_max

def _sparse_bipartite_update(self, roles: np.ndarray) -> None:
    """
    集中式撮合：每日覆盖写入最新订单；仅在拍卖日按当前状态确认数量后批量成交。
    与 RFQ 共用 collect_intentions()，仅替换 matching 节奏与对象。
    """
    B_base = float(getattr(self, "B", DEFAULT_MATCH_B))
    n = self.num_banks
    eps = 1e-8
    step = int(getattr(self, "current_step", 0))
    self._refresh_exposure_from_contract_book(step)
    deg_init = np.count_nonzero(np.abs(self.exposure_matrix) > eps, axis=1).astype(int)

    intentions = collect_intentions(
        self.banks, n, roles, self.reserve_buffer,
        float(self.base_rate),
        lcr_target=float(getattr(self, "interbank_lcr_target", DAILY_INTERBANK_INTENTION_LCR_TARGET)),
        step=step,
        last_avg_rate=getattr(self, "last_avg_rate", None),
        rollover_blocked=getattr(self, "rollover_blocked_borrowers", set()) or set(),
        rollover_borrow_policy=getattr(self, "rollover_borrow_policy", ROLLOVER_BORROW_COUPON_CLEARED),
        coupon_cleared_borrowers=getattr(self, "rollover_coupon_cleared_borrowers", None),
        coupon_due_borrowers=getattr(self, "rollover_coupon_due_borrowers", None),
        car_cutoff=float(getattr(self, "car_cutoff", 0.08)),
        debt_burden_kappa=float(
            getattr(self, "debt_burden_kappa", DEFAULT_DEBT_BURDEN_KAPPA)
        ),
        ibl_cap_asset_lambda=float(
            getattr(self, "ibl_cap_asset_lambda", DEFAULT_IBL_CAP_ASSET_LAMBDA)
        ),
    )
    for b in self.banks:
        b["demand"] = 0.0
        b["supply"] = 0.0
    for it in intentions:
        if str(it.role).lower() == "lender":
            self.banks[it.bank_idx]["supply"] = float(it.quantity)
        else:
            self.banks[it.bank_idx]["demand"] = float(it.quantity)

    # 每日覆盖最新订单；非拍卖日不撮合
    self._queue_central_orders_from_intentions(intentions)
    auction_day = bool(self._is_auction_day(step))
    self.last_was_auction_day = auction_day
    if not auction_day:
        day_supply = float(sum(float(it.quantity) for it in intentions if str(it.role).lower() == "lender"))
        day_demand = float(sum(float(it.quantity) for it in intentions if str(it.role).lower() != "lender"))
        print(
            f"[diag] step={step} centralized ORDER QUEUED "
            f"(book={len(self.central_order_book)}, auction in "
            f"{max(1, int(self.centralized_cycle_length)) - ((step + 1) % max(1, int(self.centralized_cycle_length)))} days)"
        )
        self.last_match_stats = {
            "total_volume": 0.0,
            "num_trades": 0,
            "total_demand": day_demand,
            "total_supply": day_supply,
            "unmet_demand_rate": np.nan,
            "auction_day": False,
            "order_book_size": int(len(self.central_order_book)),
        }
        return

    confirmed = self._reconfirm_central_order_quantities(intentions)
    lenders, borrowers, supply, demand, r_min, r_max = self._aggregate_central_order_book(confirmed)
    # Clamp borrower demand by bank-level IBL capacity room.
    for bj, j in enumerate(borrowers):
        room = ibl_borrowing_room(
            self.banks[int(j)],
            asset_lambda=float(
                getattr(self, "ibl_cap_asset_lambda", DEFAULT_IBL_CAP_ASSET_LAMBDA)
            ),
        )
        demand[bj] = float(min(float(demand[bj]), room))
    total_supply = float(np.sum(np.asarray(supply, dtype=float))) if supply else 0.0
    total_demand = float(np.sum(np.asarray(demand, dtype=float))) if demand else 0.0
    print(
        f"[diag] step={step} centralized AUCTION cycle={int(self.centralized_cycle_length)} "
        f"lenders={len(lenders)} borrowers={len(borrowers)} "
        f"book_banks={len(self.central_order_book)} confirmed={len(confirmed)}"
    )
    if len(lenders) == 0 or len(borrowers) == 0 or total_supply <= eps or total_demand <= eps:
        print(f"[debug] No matching on auction day: supply={total_supply:.2f}, demand={total_demand:.2f}")
        self.central_order_book.clear()
        self.last_match_stats = {
            "total_volume": 0.0,
            "num_trades": 0,
            "total_demand": total_demand,
            "total_supply": total_supply,
            "unmet_demand_rate": 1.0 if total_demand > eps else 0.0,
            "auction_day": True,
            "order_book_size": 0,
        }
        from bank_econ_shared import note_screening_loss
        note_screening_loss(self, "funding_gap", max(0.0, float(total_demand)))
        return

    print(f"[diag] supply min/mean/max = {np.min(supply):.2f}/{np.mean(supply):.2f}/{np.max(supply):.2f}")
    print(f"[diag] demand  min/mean/max = {np.min(demand):.2f}/{np.mean(demand):.2f}/{np.max(demand):.2f}")

    ctx = getattr(self, "gnn_context", None)
    if ctx is not None and ctx.get("matcher", None) is not None:
        raise RuntimeError(
            "CEN matching forbids GNN matcher; clear gnn_context and use global rate auction."
        )

    # Global information + highest feasible rate allocation (formal CEN).
    plan = self._deterministic_rate_based_matching(
            lenders=lenders,
            borrowers=borrowers,
            supply=supply,
            demand=demand,
            B=B_base,
            deg_init=deg_init,
            r_min=r_min,
            r_max=r_max,
        )

    from collections import defaultdict
    need_by_b = {j: float(d) for j, d in zip(borrowers, demand)}
    got_by_b = defaultdict(float)
    for _, j, a in plan:
        got_by_b[j] += float(a)
    good_borrowers = {j for j in borrowers if got_by_b[j] + 1e-6 >= need_by_b[j]}
    remaining_need = {j: need_by_b[j] for j in good_borrowers}
    edge_count = 0
    actual_lent = 0.0
    deg = deg_init.copy()
    rollover_blocked = getattr(self, "rollover_blocked_borrowers", set()) or set()
    rates_used = []
    from bank_econ_shared import borrower_origination_risk_q

    borrower_q_at_origination = {
        int(j): float(
            borrower_origination_risk_q(
                self.banks[int(j)],
                system=self,
                bank_idx=int(j),
                car_threshold=float(
                    getattr(self, "car_cutoff", 0.08)
                ),
            )
        )
        for j in good_borrowers
    }
    # 成交时再次按当前可用现金约束，避免把 liquid_assets 减成负数
    lender_cash_left = {
        int(i): max(
            0.0,
            float(self.banks[i].get("liquid_assets", 0.0))
            - float(self.reserve_buffer[i]) * float(self.banks[i].get("current_liabilities", 0.0)),
        )
        for i in lenders
    }

    for lender_idx, borrower_idx, amount in plan:
        amt = float(amount)
        if amt <= eps:
            continue
        if (
            borrower_idx in rollover_blocked
            and str(getattr(self, "rollover_borrow_policy", ROLLOVER_BORROW_COUPON_CLEARED)).lower()
            == ROLLOVER_BORROW_BLOCK_ALL
        ):
            continue
        if deg[lender_idx] >= self.max_degree or deg[borrower_idx] >= self.max_degree:
            continue
        if borrower_idx not in good_borrowers:
            continue
        try:
            li = lenders.index(lender_idx)
        except ValueError:
            continue
        need_rem = remaining_need.get(borrower_idx, 0.0)
        if need_rem <= eps:
            continue
        cash_left = float(lender_cash_left.get(lender_idx, 0.0))
        amt = min(amt, supply[li], need_rem, cash_left)
        if amt <= eps:
            continue
        self.banks[lender_idx]["liquid_assets"] = float(
            self.banks[lender_idx].get("liquid_assets", 0.0)
        ) - amt
        if self.banks[lender_idx]["liquid_assets"] < 0.0:
            self.banks[lender_idx]["liquid_assets"] = 0.0
        lender_cash_left[lender_idx] = max(0.0, cash_left - amt)
        self.banks[borrower_idx]["liquid_assets"] = float(
            self.banks[borrower_idx].get("liquid_assets", 0.0)
        ) + amt
        self.borrowed_cash[borrower_idx] += amt
        self.borrowed_origination_risk_sum[borrower_idx] += (
            float(amt)
            * float(borrower_q_at_origination[int(borrower_idx)])
        )
        from bank_econ_shared import accrue_funded_trade
        accrue_funded_trade(
            self,
            float(amt),
            float(borrower_q_at_origination[int(borrower_idx)]),
        )
        rate = 0.5 * (float(r_min[lender_idx]) + float(r_max[borrower_idx]))
        rates_used.append(rate)
        trade = Trade(
            lender_idx=int(lender_idx),
            borrower_idx=int(borrower_idx),
            amount=float(amt),
            rate=float(rate),
            step_executed=step,
        )
        _, dec = self.contract_book.add_from_trade_with_schedule(
            trade, self.banks[borrower_idx], self._schedule_cfg(),
        )
        log = getattr(self, "trade_schedule_log", None)
        if log is not None:
            log.append({
                "step": int(step),
                "lender": int(lender_idx),
                "borrower": int(borrower_idx),
                "amount": float(amt),
                "trade_rate": float(rate),
                "schedule_type": dec["schedule_type"],
                "reason": dec.get("reason", ""),
                "tenor": dec.get("tenor"),
                "coupon_rate": dec.get("coupon_rate"),
                "settlement_rate": dec.get("settlement_rate"),
            })
        supply[li] -= amt
        remaining_need[borrower_idx] -= amt
        deg[lender_idx] += 1
        deg[borrower_idx] += 1
        edge_count += 1
        actual_lent += amt

    self._refresh_exposure_from_contract_book(step)
    unmet_rate = max(0.0, total_demand - actual_lent) / (total_demand + 1e-9)
    self.central_order_book.clear()
    self.last_match_stats = {
        "total_volume": float(actual_lent),
        "num_trades": int(edge_count),
        "total_demand": float(total_demand),
        "total_supply": float(total_supply),
        "unmet_demand_rate": float(unmet_rate),
        "auction_day": True,
        "order_book_size": 0,
    }
    from bank_econ_shared import note_screening_loss
    note_screening_loss(self, "funding_gap", max(0.0, float(total_demand) - float(actual_lent)))
    if rates_used:
        self.last_avg_rate = float(np.mean(rates_used))
    else:
        self.last_avg_rate = None
    print(
        f"[debug] Matching finished: edges={edge_count}, "
        f"lenders={len(lenders)}, borrowers={len(borrowers)}, "
        f"total_supply={total_supply:.2f}, total_demand={total_demand:.2f}, "
        f"actual_lent={actual_lent:.2f}, B_eff={B_base:.2f}"
    )
    if not hasattr(self, "exposure_hist"):
        self.exposure_hist = {}
    self.exposure_hist[step] = self.exposure_matrix.copy()


def update_network(self):
    eta = getattr(self, "eta", 0.1)
    B   = getattr(self, "B", DEFAULT_MATCH_B)
    n = self.exposure_matrix.shape[0]
    rng = getattr(self, "rng_environment", getattr(self, "rng", np.random.default_rng(DEFAULT_RANDOM_SEED)))
    roles = getattr(self, "roles", None)

    for i in range(n):
        for j in range(i + 1, n):
            lij_old = self.exposure_matrix[i, j]
            noise   = rng.uniform(-B, B)
            lij_prop = (1 - eta) * lij_old + eta * noise

            # 如果你希望网络保持“双部图+角色约束”，就保留这个过滤
            if roles is not None:
                if (roles[i] == 0) or (roles[j] == 0) or (roles[i] * roles[j] >= 0):
                    lij_prop = 0.0
                else:
                    mag = abs(lij_prop)
                    lij_prop = +mag if (roles[i] == +1 and roles[j] == -1) else -mag

            self.exposure_matrix[i, j] = lij_prop
            self.exposure_matrix[j, i] = -lij_prop

    np.fill_diagonal(self.exposure_matrix, 0.0)


def calculate_systemic_risk(self, weights=(0.5, 0.3, 0.2), car_threshold=None, return_parts=False):
    """
    SR_t = w1·FR_t + w2·CBS_t + w3·CGR_t, default weights (0.5, 0.3, 0.2).
      FR_t  : Failure Rate (exclude central bank) — already defaulted
      CBS_t : active commercial banks with CAR < θ / active commercial banks
      CGR_t : Σgap / Σrequired over all commercial banks (failed keep gap-at-failure)
    car_threshold 默认跟 measure_car_threshold / car_cutoff，而非写死 0.08。
    """
    if car_threshold is None:
        car_threshold = float(
            getattr(self, "measure_car_threshold", getattr(self, "car_cutoff", 0.08))
        )
    else:
        car_threshold = float(car_threshold)
    sr, fr, cbs, cgr = _systemic_risk_from_banks(
        self.banks, weights=weights, car_threshold=car_threshold
    )
    if return_parts:
        return sr, fr, cbs, cgr
    return sr



def to_pyg_graph(self, y=None, use_prev: bool = False):
    """
    把当前状态转成 PyG Data（GNN 默认每步一张「全图」）：
    - 节点特征：15维 FEATURE_ORDER_15
    - 边：只取 exposure_matrix 的正边 (lender -> borrower)
    - use_prev=True 时：优先用 self.prev_exposure_matrix（上期快照），没有则回退当前
    """
    # ===== 1) node features =====
    env = {
        "market_environment": getattr(self, "market_environment", "bull"),
        "base_rate": getattr(self, "base_rate", DAILY_BULL_BASE_RATE),
        "long_term_rate": getattr(self, "long_term_rate", DAILY_BULL_BASE_RATE + DAILY_LONG_RATE_SPREAD_BULL[0]),
    }
    x = torch.tensor([_bank_to_feature_vec_15(b, env) for b in self.banks], dtype=torch.float)

    # ===== 2) choose which exposure matrix to use =====
    L_src = None
    if use_prev:
        L_prev = getattr(self, "prev_exposure_matrix", None)
        if L_prev is not None:
            L_src = L_prev

    if L_src is None:
        L_src = getattr(self, "exposure_matrix", None)

    if L_src is None:
        # 极端兜底：没有任何矩阵
        n = x.size(0)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 1), dtype=torch.float)
        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if y is not None:
            g.y = torch.tensor([float(y)], dtype=torch.float)
        return g

    L = np.asarray(L_src, dtype=float)

    # ===== 3) edges: positive exposures only =====
    n = L.shape[0]
    src, dst, w = [], [], []
    for i in range(n):
        row = L[i]
        for j in range(n):
            if i == j:
                continue
            val = row[j]
            if val > 1e-9:
                src.append(i)
                dst.append(j)
                w.append(val)

    if len(src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(w, dtype=torch.float).view(-1, 1)

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        g.y = torch.tensor([float(y)], dtype=torch.float)
    return g

def settle_interbank_and_clear(self, use_core=False):
    """
    一期同业到期结算：借款人支付 -> 债权人收款，然后把 exposure_matrix 清零。
    - use_core=False: 只允许用 liquid_assets 付款（更“流动性约束”）
    - use_core=True : 允许用 core+liq（更“偿付能力约束”）
    """
    eps = 1e-9
    L = np.asarray(self.exposure_matrix, float)
    self.prev_exposure_matrix = L.copy()
    n = L.shape[0]
    Lbar = np.maximum(-L, 0.0)  # debtor -> creditor liabilities
    np.fill_diagonal(Lbar, 0.0)
    p_bar = Lbar.sum(axis=1)    # each debtor total due (principal)

    if float(p_bar.sum()) <= eps:
        self.exposure_matrix[:] = 0.0
        return

    # 把一期利息并入应付（简单用 base_rate 或者你也可换成 borrower/lender rate）
    r_ib = float(getattr(self, "base_rate", DAILY_BULL_BASE_RATE))
    Lbar_int = Lbar * (1.0 + r_ib)

    # 构造带利息的 L_int（仍保持 antisymmetric）
    L_int = np.zeros_like(L)
    for debtor in range(n):
        for cred in range(n):
            a = Lbar_int[debtor, cred]
            if a > eps:
                L_int[debtor, cred] = -a
                L_int[cred, debtor] = +a

    # endowment：用 liquid 或 core+liq
    if use_core:
        e = np.array([b["core_capital"] + b["liquid_assets"] for b in self.banks], float)
    else:
        e = np.array([b["liquid_assets"] for b in self.banks], float)

    # 清算支付
    p = self.solve_clearing(L_int, e)

    # 分摊到每个债权人：Pi^T p
    pbar_int = Lbar_int.sum(axis=1) + eps
    Pi = Lbar_int / pbar_int[:, None]
    recv = Pi.T @ p  # creditor receives

    # 现金更新：debtor pay, creditor receive
    for i in range(n):
        self.banks[i]["liquid_assets"] = float(max(0.0, self.banks[i]["liquid_assets"] - p[i]))
    for i in range(n):
        self.banks[i]["liquid_assets"] = float(self.banks[i]["liquid_assets"] + recv[i])

    # 结算后不做 rollover；跨期连续性由 ContractBook 保存。
    self.exposure_matrix[:] = 0.0
    np.fill_diagonal(self.exposure_matrix, 0.0)

def build_candidate_graphs_for_pairs(base_graph, pairs, amounts):
    cand = []
    old_scale = float(getattr(base_graph, "edge_scale", 1.0))
    old_scale = max(old_scale, 1e-9)

    for (i, j), w in zip(pairs, amounts):
        w = float(w)
        new_scale = max(old_scale, w, 1e-9)

        g = Data(
            x=base_graph.x.clone(),
            edge_index=base_graph.edge_index.clone(),
            edge_attr=base_graph.edge_attr.clone()
        )

        if g.edge_attr.numel() > 0 and new_scale != old_scale:
            g.edge_attr = g.edge_attr * (old_scale / new_scale)

        ei_add = torch.tensor([[i], [j]], dtype=torch.long)
        ea_add = torch.tensor([[w / new_scale]], dtype=torch.float)

        g.edge_index = torch.cat([g.edge_index, ei_add], dim=1)
        g.edge_attr  = torch.cat([g.edge_attr,  ea_add], dim=0)
        g.edge_scale = float(new_scale)
        cand.append(g)

    return cand


def decide_investment(self, bank):
    exp_proj = self.long_term_rate
    rho   = bank.get('hurdle_rate', DAILY_HURDLE_RATE)
    alpha = bank.get('risk_appetite', 0.5)
    liq   = bank['liquid_assets']
    invest_amt = alpha * liq if exp_proj > rho else 0.0
    return max(0.0, min(invest_amt, liq * 0.8))

def invest_free_cash_into_projects(self, i, invest_frac: float | None = None):
    """
    把银行 i 的一部分流动资产投到项目（必须写入 project_book，否则 update_project_book 会把 amount 清回去）
    """
    from project_common_random import (
        PROJECT_ORIGIN_FREE_CASH,
        make_project_id,
        project_creation_draws,
    )
    from bank_econ_shared import borrower_origination_risk_q

    bank = self.banks[i]
    if not bank.get("is_active", True) or bank.get("absorbing_default"):
        return
    if invest_frac is None:
        invest_frac = 0.10 if getattr(self, "free_market", False) else 0.30

    liq = float(bank['liquid_assets'])
    avail = max(0.0, liq - self.reserve_buffer[i] * float(bank['current_liabilities']))
    invest = min(avail, liq * float(invest_frac))
    if invest <= 1e-8:
        return

    bank['liquid_assets'] -= invest

    step = int(getattr(self, "current_step", 0))
    pid = make_project_id(
        creation_step=step,
        origin=PROJECT_ORIGIN_FREE_CASH,
        slot=0,
    )
    maturity, pd, lgd = project_creation_draws(
        seed=int(getattr(self, "seed", DEFAULT_RANDOM_SEED)),
        creation_step=step,
        bank_id=i,
        project_id=pid,
        maturity_range=DAILY_PROJECT_MATURITY_DAYS,
        pd_range=DAILY_PROJECT_PD_RANGE,
        lgd_range=(0.30, 0.60),
    )
    loan = ProjectLoan(
        principal=float(invest),
        rate=float(self.long_term_rate + DAILY_PROJECT_SPREAD),
        maturity=maturity,
        pd=pd,
        lgd=lgd,
        project_id=pid,
        origination_risk=float(
            borrower_origination_risk_q(
                bank,
                system=self,
                bank_idx=i,
                car_threshold=float(getattr(self, "car_cutoff", 0.08)),
            )
        ),
    )
    self.project_book[i].append(loan)
    bank['investment']['projects']['amount'] += invest



def allocate_borrowed_to_projects(self, i, spread: float = DAILY_PROJECT_SPREAD):
    """借入现金已入账；这里只把项目份额从 liquid 转入项目台账。"""
    from project_common_random import (
        PROJECT_ORIGIN_BORROWED,
        make_project_id,
        project_creation_draws,
    )
    from bank_econ_shared import borrower_origination_risk_q, borrowed_project_maturity

    if (
        not self.banks[i].get("is_active", True)
        or self.banks[i].get("absorbing_default")
    ):
        self.borrowed_cash[i] = 0.0
        self.borrowed_origination_risk_sum[i] = 0.0
        return
    budget = float(self.borrowed_cash[i])
    if budget <= 1e-8:
        self.borrowed_origination_risk_sum[i] = 0.0
        return

    target_proj = budget * self.project_min_share
    num = max(1, int(target_proj // 5e5))
    if num == 1:
        num = 2
    per = target_proj / num if num > 0 else 0.0

    risk_sum_arr = getattr(
        self,
        "borrowed_origination_risk_sum",
        None,
    )

    if risk_sum_arr is not None:
        q_j = float(
            np.clip(
                float(risk_sum_arr[i]) / max(budget, 1e-9),
                0.0,
                1.0,
            )
        )
    else:
        # 只用于兼容旧对象
        q_j = float(
            borrower_origination_risk_q(
                self.banks[i],
                system=self,
                bank_idx=i,
                car_threshold=float(
                    getattr(self, "car_cutoff", 0.08)
                ),
            )
        )
    self.banks[i]["liquid_assets"] = float(self.banks[i].get("liquid_assets", 0.0)) - float(target_proj)
    step = int(getattr(self, "current_step", 0))
    for slot in range(num):
        pid = make_project_id(
            creation_step=step,
            origin=PROJECT_ORIGIN_BORROWED,
            slot=slot,
        )
        maturity, pd, lgd = project_creation_draws(
            seed=int(getattr(self, "seed", DEFAULT_RANDOM_SEED)),
            creation_step=step,
            bank_id=i,
            project_id=pid,
            maturity_range=DAILY_PROJECT_MATURITY_DAYS,
            pd_range=DAILY_PROJECT_PD_RANGE,
            lgd_range=(0.30, 0.60),
        )
        # Asset-side tenor mix is identical across feature scenarios: half
        # one-period and half 20–60. Rollover changes only liability timing.
        maturity = borrowed_project_maturity(
            maturity,
            rollover_enabled=bool(getattr(self, "rollover_enabled", True)),
            slot=slot,
            n_slots=num,
        )
        loan = ProjectLoan(
            principal=float(per),
            rate=float(self.long_term_rate + spread),
            maturity=maturity,
            pd=pd,
            lgd=lgd,
            project_id=pid,
            origination_risk=q_j,
        )
        self.project_book[i].append(loan)
        self.banks[i]['investment']['projects']['amount'] += float(per)

    self.borrowed_cash[i] = 0.0
    self.borrowed_origination_risk_sum[i] = 0.0

def _daily_project_shock_params(market_environment: str) -> tuple[float, float]:
    if market_environment == "bull":
        return DAILY_PROJECT_SHOCK_MEAN_BULL, DAILY_PROJECT_SHOCK_STD_BULL
    return DAILY_PROJECT_SHOCK_MEAN_BEAR, DAILY_PROJECT_SHOCK_STD_BEAR


def update_project_book(self, i):
    """
    项目台账：每期结算回报 / 负收益 / 到期回本 / 违约扣损。
    随机冲击按 (seed, step, bank, project_id) 键生成，与调用顺序无关。
    借入项目在 ON/OFF 下均约一半 tenor=1，其余 20–60 天。
    OFF 隔夜仍到期全额；核心只垫当期项目亏损，不把锁定期本金兑成现金还债。
    """
    from project_common_random import project_period_draws
    from bank_econ_shared import effective_project_pd, effective_shock_std

    bank = self.banks[i]
    apply = bool(bank.get("is_active", True)) and not bool(
        bank.get("absorbing_default") or bank.get("balance_sheet_frozen")
    )

    mu_shock, sigma_shock = _daily_project_shock_params(self.market_environment)
    clip_lo, clip_hi = DAILY_PROJECT_REALIZED_CLIP
    step = int(getattr(self, "current_step", 0))
    seed = int(getattr(self, "seed", DEFAULT_RANDOM_SEED))
    macro = float(getattr(self, "project_pd_stress_multiplier", 1.0))

    new_book = []
    projects_amt = 0.0
    period_loss = 0.0

    for loan in self.project_book[i]:
        q = float(getattr(loan, "origination_risk", 0.0) or 0.0)
        sigma_eff = effective_shock_std(sigma_shock, q)
        pd_roll, shock = project_period_draws(
            seed=seed,
            step=step,
            bank_id=i,
            project_id=int(getattr(loan, "project_id", 0) or 0),
            shock_mean=mu_shock,
            shock_std=sigma_eff,
        )
        pd_eff = effective_project_pd(float(loan.pd), q, macro)
        if pd_roll < pd_eff:
            if apply:
                recovery = float(loan.principal) * (1.0 - float(loan.lgd))
                period_loss += float(loan.principal) * float(loan.lgd)
                bank['liquid_assets'] = float(bank.get('liquid_assets', 0.0)) + recovery
                from bank_econ_shared import accrue_project_default_loss
                accrue_project_default_loss(
                    self, principal=float(loan.principal), lgd=float(loan.lgd), q=q,
                )
            else:
                new_book.append(loan)
                projects_amt += float(loan.principal)
            continue

        if not apply:
            new_book.append(loan)
            projects_amt += float(loan.principal)
            continue

        realized_r = float(np.clip(loan.rate + shock, clip_lo, clip_hi))
        cashflow = loan.principal * realized_r

        bank['liquid_assets'] += cashflow

        if realized_r < 0.0:
            loss = -cashflow
            period_loss += loss
            from bank_econ_shared import accrue_project_negative_return_loss
            accrue_project_negative_return_loss(self, loss)

        loan.age += 1
        if loan.age >= loan.maturity:
            bank['liquid_assets'] += loan.principal
        else:
            new_book.append(loan)
            projects_amt += loan.principal

    if apply:
        self.project_book[i] = new_book
        bank['investment']['projects']['amount'] = float(projects_amt)
        from bank_econ_shared import cover_overnight_after_project_settlement
        cover_overnight_after_project_settlement(
            self, int(i), bank, self.project_book[i], period_loss=period_loss
        )
def visualize_network(
    self,
    step,
    risk,
    tag: str = "",
    save: bool = True,
    show_first: bool = True,
    edge_quantile: float = 0.0,
    seed: int = DEFAULT_RANDOM_SEED,
):
    """
    交互式银行网络图（悬停查看信息）。
    """
    try:
        import mplcursors  # type: ignore
        HAS_CURSOR = True
    except Exception:
        HAS_CURSOR = False

    L = np.asarray(self.exposure_matrix, float)
    n = L.shape[0]
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)

    # ===== DIAG: edge nnz / sign =====
    nnz = int(np.count_nonzero(np.abs(L) > 1e-12))
    pos_cnt = int(np.count_nonzero(L > 1e-12))
    neg_cnt = int(np.count_nonzero(L < -1e-12))
    print(
        f"[diag-net] step={step} nnz(abs>1e-12)={nnz} | "
        f"pos={pos_cnt} neg={neg_cnt} | max|L|={float(np.max(np.abs(L))):.2f}"
    )

    # ===== build edges (FIX: 如果暴露全是负的，也能画出线) =====
    abs_ws = []
    eps = 1e-12
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = float(L[i, j])
            if abs(val) <= eps:
                continue

            # 统一成正权重来画：val<0 就翻方向
            if val > 0:
                u, v, wpos = i, j, val
            else:
                u, v, wpos = j, i, -val

            G.add_edge(u, v, weight=wpos, absw=wpos)
            abs_ws.append(wpos)

    abs_ws = np.asarray(abs_ws, float) if len(abs_ws) else np.array([])

    strong_edges = set()
    thr = np.quantile(abs_ws, edge_quantile) if abs_ws.size else np.inf
    for u, v, d in G.edges(data=True):
        if d.get("absw", 0.0) >= thr:
            strong_edges.add((u, v))

    types = self.bank_types
    idx_c = [i for i, t in enumerate(types) if t == "central"]
    idx_m = [i for i, t in enumerate(types) if t == "commercial"]
    idx_s = [i for i, t in enumerate(types) if t == "shadow"]

    pos = {}
    if idx_c:
        pos[idx_c[0]] = (0.0, 0.0)
    r1, r2 = 1.2, 2.0
    for k, i in enumerate(idx_m):
        ang = 2 * np.pi * k / max(1, len(idx_m))
        pos[i] = (r1 * np.cos(ang), r1 * np.sin(ang))
    for k, i in enumerate(idx_s):
        ang = 2 * np.pi * k / max(1, len(idx_s))
        pos[i] = (r2 * np.cos(ang), r2 * np.sin(ang))

    pos = nx.spring_layout(
        G, pos=pos, fixed=idx_c, seed=seed, k=0.8 / np.sqrt(max(n, 1))
    )

    car = np.array([self.banks[i].get("capital_adequacy_ratio", 0.0) for i in range(n)], dtype=float)
    lcr = np.array([self.banks[i].get("liquidity_coverage_ratio", 1.0) for i in range(n)], dtype=float)
    solv = np.array([self.banks[i].get("solvency_ratio", 1.0) for i in range(n)], dtype=float)

    CAR_TARGET, LCR_TARGET, SOLV_TARGET = 0.08, 1.0, 1.0

    def smooth_risk(x, target, width=0.7):
        z = (x - target) / (width * target + 1e-9)
        return 1.0 / (1.0 + np.exp(z))

    node_risk = (
        0.5 * smooth_risk(car,  CAR_TARGET,  width=0.7) +
        0.3 * smooth_risk(lcr,  LCR_TARGET,  width=0.7) +
        0.2 * smooth_risk(solv, SOLV_TARGET, width=0.7)
    )

    rmin, rmax = float(np.nanmin(node_risk)), float(np.nanmax(node_risk))
    norm = (node_risk - rmin) / (rmax - rmin + 1e-9)

    node_sizes = 350 + 950 * norm
    cmap = plt.get_cmap("RdYlGn_r")
    node_colors = cmap(norm)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=160)
    ax.set_title(f"Step {step} — Systemic Risk: {float(risk):.02f}", fontsize=20, pad=14)

    bg_lines = []
    if G.number_of_edges():
        for (u, v, d) in G.edges(data=True):
            (x0, y0), (x1, y1) = pos[u], pos[v]
            ln = ax.plot([x0, x1], [y0, y1], color="gray", alpha=0.25, lw=0.8, zorder=0)[0]
            bg_lines.append(ln)

    edge_lines = {}
    for (u, v) in strong_edges:
        (x0, y0), (x1, y1) = pos[u], pos[v]
        w = abs(G[u][v]["weight"])
        ln = ax.plot(
            [x0, x1], [y0, y1],
            color="gray", alpha=0.85,
            lw=1.2 + 4.0 * (w / (thr + 1e-9)), zorder=1
        )[0]
        edge_lines[(u, v)] = ln

    nodes_list = list(range(n))
    coll = ax.scatter(
        [pos[i][0] for i in nodes_list],
        [pos[i][1] for i in nodes_list],
        s=node_sizes, c=node_colors,
        edgecolors="white", linewidths=1.2, zorder=2
    )

    import matplotlib.patheffects as pe
    for i0 in nodes_list:
        x, y = pos[i0]
        label = f"Bank{i0+1}\n{node_risk[i0]:.2f}"
        txt = ax.text(x, y, label, ha="center", va="center", fontsize=9, color="black", zorder=3)
        txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    ax.set_axis_off()
    ax.margins(0.12)

    if HAS_CURSOR:
        cursor = mplcursors.cursor(coll, hover=True)

        @cursor.connect("add")
        def _on_add(sel):
            i_idx = int(sel.index)
            node_id = nodes_list[i_idx]
            b = self.banks[node_id]

            ib_out = float(np.maximum(L[node_id], 0.0).sum())
            ib_in  = float(np.maximum(-L[:, node_id], 0.0).sum())
            proj   = float(b['investment']['projects']['amount'])
            car_v  = float(b.get('capital_adequacy_ratio', 0.0))
            lcr_v  = float(b.get('liquidity_coverage_ratio', 0.0))
            lev    = float(b.get('leverage_ratio', 0.0))
            solv_v = float(b.get('solvency_ratio', 0.0))

            text = (
                f"{b.get('name', f'Bank{node_id+1}')} ({b.get('type','N/A')})\n"
                f"Solvency {solv_v:.2f}  CAR {car_v:.2%}  LCR {lcr_v:.2f}  Lev {lev:.2f}\n"
                f"Capital {b.get('core_capital',0):.0f}  Liquid {b.get('liquid_assets',0):.0f}\n"
                f"IB out {ib_out:.0f}  IB in {ib_in:.0f}  Projects {proj:.0f}"
            )
            sel.annotation.set(text=text, fontsize=9, alpha=0.95)

            if hasattr(sel.annotation, "arrow_patch") and sel.annotation.arrow_patch:
                sel.annotation.arrow_patch.set_visible(False)

            for (uu, vv), ln in edge_lines.items():
                if uu == node_id or vv == node_id:
                    ln.set_alpha(0.95)
                    ln.set_linewidth(max(2.0, ln.get_linewidth()))
                else:
                    ln.set_alpha(0.06)
                    ln.set_linewidth(0.6)
            for ln in bg_lines:
                ln.set_alpha(0.02)
            fig.canvas.draw_idle()

        @cursor.connect("remove")
        def _on_remove(sel):
            for ln in edge_lines.values():
                ln.set_alpha(0.35)
                ln.set_linewidth(1.2)
            for ln in bg_lines:
                ln.set_alpha(0.08)
            fig.canvas.draw_idle()

    if show_first:
        plt.show()
        plt.pause(0.2)

    if save:
        fname = FIG_DIR / f"network_{(tag or 'normal')}_step{step}.png"
        fig.savefig(str(fname), dpi=300, bbox_inches="tight")
        print(f"Saved network figure: {fname}")

    plt.close(fig)

    def smooth_risk(x, target, width=0.7):
        z = (x - target) / (width * target + 1e-9)
        return 1.0 / (1.0 + np.exp(z))

    node_risk = (
        0.5 * smooth_risk(car,  CAR_TARGET,  width=0.7) +
        0.3 * smooth_risk(lcr,  LCR_TARGET,  width=0.7) +
        0.2 * smooth_risk(solv, SOLV_TARGET, width=0.7)
    )

    rmin, rmax = float(np.nanmin(node_risk)), float(np.nanmax(node_risk))
    norm = (node_risk - rmin) / (rmax - rmin + 1e-9)

    node_sizes = 350 + 950 * norm
    cmap = plt.get_cmap("RdYlGn_r")
    node_colors = cmap(norm)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=160)
    ax.set_title(f"Step {step} — Systemic Risk: {float(risk):.02f}", fontsize=20, pad=14)

    bg_lines = []
    if G.number_of_edges():
        for (u, v, d) in G.edges(data=True):
            (x0, y0), (x1, y1) = pos[u], pos[v]
            ln = ax.plot([x0, x1], [y0, y1], color="gray", alpha=0.25, lw=0.8, zorder=0)[0]
            bg_lines.append(ln)

    edge_lines = {}
    for (u, v) in strong_edges:
        (x0, y0), (x1, y1) = pos[u], pos[v]
        w = abs(G[u][v]["weight"])

        # ===== PATCH: stable linewidth (log-compress + clamp) =====
        LW_MIN = 0.6
        LW_MAX = 3.5
        wmax = float(np.max(abs_ws)) if abs_ws.size else w
        wn = np.log1p(w) / (np.log1p(wmax) + 1e-9)   # 0~1
        lw = LW_MIN + (LW_MAX - LW_MIN) * wn

        ln = ax.plot(
            [x0, x1], [y0, y1],
            color="gray", alpha=0.50,
            lw=lw, zorder=1
        )[0]
        edge_lines[(u, v)] = ln

    nodes_list = list(range(n))
    coll = ax.scatter(
        [pos[i][0] for i in nodes_list],
        [pos[i][1] for i in nodes_list],
        s=node_sizes, c=node_colors,
        edgecolors="white", linewidths=1.2, zorder=2
    )

    import matplotlib.patheffects as pe
    for i0 in nodes_list:
        x, y = pos[i0]
        label = f"Bank{i0+1}\n{node_risk[i0]:.2f}"
        txt = ax.text(x, y, label, ha="center", va="center", fontsize=9, color="black", zorder=3)
        txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    ax.set_axis_off()
    ax.margins(0.12)

    if HAS_CURSOR:
        cursor = mplcursors.cursor(coll, hover=True)

        @cursor.connect("add")
        def _on_add(sel):
            i_idx = int(sel.index)
            node_id = nodes_list[i_idx]
            b = self.banks[node_id]

            ib_out = float(np.maximum(L[node_id], 0.0).sum())
            ib_in  = float(np.maximum(-L[:, node_id], 0.0).sum())
            proj   = float(b['investment']['projects']['amount'])
            car_v  = float(b.get('capital_adequacy_ratio', 0.0))
            lcr_v  = float(b.get('liquidity_coverage_ratio', 0.0))
            lev    = float(b.get('leverage_ratio', 0.0))
            solv_v = float(b.get('solvency_ratio', 0.0))

            text = (
                f"{b.get('name', f'Bank{node_id+1}')} ({b.get('type','N/A')})\n"
                f"Solvency {solv_v:.2f}  CAR {car_v:.2%}  LCR {lcr_v:.2f}  Lev {lev:.2f}\n"
                f"Capital {b.get('core_capital',0):.0f}  Liquid {b.get('liquid_assets',0):.0f}\n"
                f"IB out {ib_out:.0f}  IB in {ib_in:.0f}  Projects {proj:.0f}"
            )
            sel.annotation.set(text=text, fontsize=9, alpha=0.95)

            if hasattr(sel.annotation, "arrow_patch") and sel.annotation.arrow_patch:
                sel.annotation.arrow_patch.set_visible(False)

            for (u, v), ln in edge_lines.items():
                if u == node_id or v == node_id:
                    ln.set_alpha(0.95)
                    ln.set_linewidth(max(2.0, ln.get_linewidth()))
                else:
                    ln.set_alpha(0.06)
                    ln.set_linewidth(0.6)
            for ln in bg_lines:
                ln.set_alpha(0.02)
            fig.canvas.draw_idle()

        @cursor.connect("remove")
        def _on_remove(sel):
            for ln in edge_lines.values():
                ln.set_alpha(0.35)
                ln.set_linewidth(1.2)
            for ln in bg_lines:
                ln.set_alpha(0.08)
            fig.canvas.draw_idle()

    if show_first:
        plt.show()   # ✅ 不阻塞，单独窗口弹出
        plt.pause(0.2)          # ✅ 给窗口一点时间刷新
    if save:
        fname = FIG_DIR / f"network_{(tag or 'normal')}_step{step}.png"
        fig.savefig(str(fname), dpi=300, bbox_inches="tight")
        print(f"Saved network figure: {fname}")
    plt.close(fig)

class BankContagionDataset:
    def __init__(
        self,
        num_simulations=1000,
        num_timesteps=5,
        data_file=str(DATA_FILE),
        *,
        force_regenerate: bool = False,
        min_samples: int | None = None,
    ):
        self.num_simulations = num_simulations
        self.num_timesteps = num_timesteps
        self.data_file = data_file
        self.expected_features = 15
        need = int(min_samples) if min_samples is not None else int(num_simulations)
        if (not force_regenerate) and (not should_regenerate_dataset(
            self.data_file, force=False, min_samples=need
        )):
            self.data = load_json_dataset(self.data_file)
            print(
                f"[BankContagionDataset] load cache {self.data_file} "
                f"(n={len(self.data)}, no append)"
            )
            return
        # Regenerate from scratch — never append onto a stale/partial cache.
        self.data = []
        self._generate_data()
        save_json_dataset(self.data_file, self.data)
        print(
            f"[BankContagionDataset] wrote {self.data_file} "
            f"(n={len(self.data)}, force={force_regenerate})"
        )

    def _generate_data(self):
        """
        生成用于训练 GNN+LSTM 的模拟数据。
        """
        simulator = BankNetworkSimulator(num_banks=30, max_steps=self.num_timesteps)

        def make_snapshot():
            # ★ 统一造图逻辑：跟运行时撮合/预测完全一致
            g = simulator.to_pyg_graph()

            return {
                "node_features": g.x.detach().cpu().numpy().astype(float).tolist(),
                "edge_index":   g.edge_index.detach().cpu().numpy().astype(int).tolist(),
                "edge_attr":    g.edge_attr.detach().cpu().numpy().astype(float).tolist(),
            }


        def run_one_trajectory(label):
            try:
                simulator.initialize_network()
            except Exception as e:
                print(f"[{label}] Error in initialize_network: {e}")
                raise

            sequence = []
            final_risk = 0.0
            for step in range(self.num_timesteps):
                try:
                    final_risk = simulator.simulate_step(step)
                except Exception as e:
                    print(f"[{label}] Error in simulate_step(step={step}): {e}")
                    raise

                try:
                    snap = make_snapshot()
                except Exception as e:
                    print(f"[{label}] Error in make_snapshot at step={step}: {e}")
                    print(f"    num_banks={simulator.num_banks}, "
                          f"exposure_matrix.shape={simulator.exposure_matrix.shape}")
                    raise

                snap["systemic_risk"] = float(final_risk)
                sequence.append(snap)

            return sequence, float(final_risk)

        # 1) 常规样本
        for _ in range(self.num_simulations):
            seq, final_risk = run_one_trajectory("normal")
            self.data.append({"sequence": seq, "risk": float(final_risk)})

        # 2) 极端高风险样本
        extreme_threshold = 0.8
        count_extreme, tries_extreme = 0, 0
        max_tries = 2000

        while count_extreme < 200 and tries_extreme < max_tries:
            tries_extreme += 1
            seq, final_risk = run_one_trajectory("extreme")
            if final_risk > extreme_threshold:
                self.data.append({"sequence": seq, "risk": float(final_risk)})
                count_extreme += 1

        # 3) 低风险样本
        normal_threshold = 0.2
        count_normal, tries_normal = 0, 0

        while count_normal < 200 and tries_normal < max_tries:
            tries_normal += 1
            seq, final_risk = run_one_trajectory("low")
            if final_risk < normal_threshold:
                self.data.append({"sequence": seq, "risk": float(final_risk)})
                count_normal += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        item = self.data[idx]
        sequence = item['sequence']
        graph_sequence = []
        for t in range(len(sequence)):
            data = sequence[t]
            x = torch.tensor(data['node_features'], dtype=torch.float)
            edge_index = torch.tensor(data['edge_index'], dtype=torch.long)
            edge_attr = torch.tensor(data['edge_attr'], dtype=torch.float)
            y = torch.tensor([data['systemic_risk']], dtype=torch.float)
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            graph_sequence.append(graph)
        return graph_sequence


class GraphWindowDataset(Dataset):
    """
    Wraps BankContagionDataset to yield fixed-length (seq_len) graph windows.
    Builds (sample_idx, start_t) windows from raw json sequence length.
    """
    def __init__(self, base_dataset, seq_len=5, stride=1, drop_short=True, pad_short=False):
        self.base_dataset = base_dataset
        self.seq_len = seq_len
        self.stride = stride
        self.drop_short = drop_short
        self.pad_short = pad_short
        self._index = []
        for i in range(len(base_dataset.data)):
            L = len(base_dataset.data[i]["sequence"])
            if drop_short and L < seq_len:
                continue
            if pad_short and L < seq_len:
                self._index.append((i, 0))
                continue
            for start_t in range(0, L - seq_len + 1, stride):
                self._index.append((i, start_t))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        sample_idx, start_t = self._index[idx]
        seq_raw = self.base_dataset.data[sample_idx]["sequence"][start_t : start_t + self.seq_len]
        graphs_window = []
        for data in seq_raw:
            x = torch.tensor(data["node_features"], dtype=torch.float)
            edge_index = torch.tensor(data["edge_index"], dtype=torch.long)
            edge_attr = torch.tensor(data["edge_attr"], dtype=torch.float)
            y = torch.tensor([data["systemic_risk"]], dtype=torch.float)
            g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            graphs_window.append(g)
        target = graphs_window[-1].y.clone()
        return (graphs_window, target)


def collate_graph_windows(batch):
    """Collate list of (window, y) into (windows, targets)."""
    windows = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])
    return (windows, targets)


class GNNPairMatcher(nn.Module):
    """
    目的：给定当前图 base_graph，输出任意 pair(i,j) 的 a_ij ∈ (0,1)
    用法：a = matcher.score_pairs(g, pairs=[(i,j),...])  -> numpy array
    """
    def __init__(self, in_dim=15, hid=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hid * 4, hid),
            nn.ReLU(),
            nn.Linear(hid, 1)
        )

    def node_embed(self, g: Data):
        x = g.x
        ei = g.edge_index

        # edge_weight: shape [E]，来自 edge_attr 的金额
        if getattr(g, "edge_attr", None) is not None and g.edge_attr.numel() > 0:
            ew = g.edge_attr.view(-1).to(x.dtype)

        # 关键：稳定化（金额通常跨度很大）
        # 1) 只保留非负（你图里本来就是正边）
            ew = torch.clamp(ew, min=0.0)

        # 2) log 压缩，防止极端大额主导
            ew = torch.log1p(ew)

        # 3) 归一化到均值约 1（可选，但强烈推荐）
            ew = ew / (ew.mean() + 1e-9)
        else:
            ew = None

        h = F.relu(self.conv1(x, ei, edge_weight=ew))
        h = F.relu(self.conv2(h, ei, edge_weight=ew))
        return h

    @torch.no_grad()
    def score_pairs(self, g: Data, pairs, device="cpu"):
        self.eval()
        g = g.to(device)
        h = self.node_embed(g)  # [N,hid]

        ii = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
        jj = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)

        hi, hj = h[ii], h[jj]
        feat = torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=1)  # [K,4hid]
        logit = self.edge_mlp(feat).view(-1)
        a = torch.sigmoid(logit)  # (0,1)
        return a.detach().cpu().numpy()

class GNNLSTMModel(nn.Module):
    """
    无卷积版：
      - 节点：两层 MLP 后做全图平均池化
      - 边：拼接 8 个统计量
      - 序列：送入 LSTM 预测最后一步风险
    """
    def __init__(self, input_dim=15, hidden_dim=64, lstm_hidden_dim=32, output_dim=1, seq_len=5):
        super().__init__()
        self.seq_len = seq_len
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.merge = nn.Linear(hidden_dim + 8, hidden_dim)
        self.lstm  = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        self.fc    = nn.Linear(lstm_hidden_dim, output_dim)

    def _graph_vector(self, g):
        x = g.x
        h = self.node_mlp(x)
        node_pool = h.mean(dim=0)

        N = x.size(0)
        if g.edge_index is not None and g.edge_index.numel() > 0:
            M = g.edge_index.size(1)
            ew = g.edge_attr.view(-1) if (g.edge_attr is not None and g.edge_attr.numel() > 0) else x.new_zeros(M)
            e_num  = x.new_tensor([float(M)])
            e_sum  = ew.sum().unsqueeze(0)
            e_mean = ew.mean().unsqueeze(0)
            e_max  = ew.max().unsqueeze(0)
            out_deg = torch.bincount(g.edge_index[0], minlength=N).float().to(x.device)
            in_deg  = torch.bincount(g.edge_index[1], minlength=N).float().to(x.device)
            d_feats = torch.stack([
                out_deg.mean(), out_deg.std(unbiased=False),
                in_deg.mean(),  in_deg.std(unbiased=False)
            ])
            edge_stats = torch.cat([e_num, e_sum, e_mean, e_max, d_feats], dim=0)
        else:
            edge_stats = x.new_zeros(8)

        gv = torch.cat([node_pool, edge_stats], dim=0)
        gv = F.relu(self.merge(gv))
        return gv

    def forward(self, graph_sequence):
        if not graph_sequence:
            raise ValueError("graph_sequence is empty")

        # Batch input: List[List[Data]] with shape [B][seq_len]
        if isinstance(graph_sequence[0], (list, tuple)):
            feats = []
            for window in graph_sequence:
                f = [self._graph_vector(g) for g in window]
                feats.append(torch.stack(f))
            feats = torch.stack(feats)  # [B, seq_len, F]
            lstm_out, _ = self.lstm(feats)
            out = self.fc(lstm_out[:, -1, :])
            return out

        # Legacy flat input: List[Data], total length = B * seq_len (for gnn_cost_of_candidates)
        feats = [self._graph_vector(g) for g in graph_sequence]
        if len(feats) % self.seq_len != 0:
            raise ValueError(
                f"graphs count {len(feats)} is not a multiple of seq_len={self.seq_len}"
            )

        B = len(feats) // self.seq_len
        feats = torch.stack(feats).view(B, self.seq_len, -1)
        lstm_out, _ = self.lstm(feats)
        out = self.fc(lstm_out[:, -1, :])
        return out


@torch.no_grad()
def gnn_cost_of_candidates(model, past_seq_graphs, candidate_graphs, seq_len=5, device="cpu"):
    model.eval()
    flat = []
    for g_next in candidate_graphs:
        flat.extend(past_seq_graphs + [g_next])
    out = model([g.to(device) for g in flat]).view(-1).detach().cpu().numpy()
    return out


def matching_with_gnn(
    model, past_seq_graphs, base_graph,
    lenders, borrowers, supply, demand,
    seq_len=5, device="cpu",
    deg_init=None, max_degree=None, B=None,
    beta_amt=0.05,      # 金额偏好权重（可调，越大越偏向大额）
):
    """
    用模型(你的 GNN+LSTM 风险预测器)来驱动撮合：
    - 对每个候选 pair (i,j) 构造 “加一条边后的下一期图”
    - 预测该候选下的风险 pred_risk(i,j)
    - score = pred_risk_baseline - pred_risk(i,j)  (越大越好)
    - 再按 score 贪心成交，同时满足 supply/demand/max_degree/B 约束
    """
    eps = 1e-8

    # 需要 past_seq_graphs 长度 = seq_len-1 才能推下一步
    if (model is None) or (past_seq_graphs is None) or (len(past_seq_graphs) < seq_len - 1):
        return []  # 让外层 fallback 到 deterministic 或者你也可以这里直接 deterministic

    # 映射：bank_id -> 在 supply/demand 数组中的位置
    idx_L = {i: k for k, i in enumerate(lenders)}
    idx_B = {j: k for k, j in enumerate(borrowers)}

    supply_left = np.array(supply, dtype=float).copy()
    demand_left = np.array(demand, dtype=float).copy()

    if max_degree is None:
        max_degree = 10**9
    deg = np.array(deg_init, dtype=int).copy() if deg_init is not None else np.zeros(base_graph.x.size(0), dtype=int)

    # 1) 枚举候选 pair（只保留供需都>0的）
    pairs2, amounts2 = [], []
    for i in lenders:
        li = idx_L[i]
        if supply_left[li] <= eps:
            continue
        for j in borrowers:
            bj = idx_B[j]
            if demand_left[bj] <= eps:
                continue
            if i == j:
                continue
            if deg[i] >= max_degree or deg[j] >= max_degree:
                continue
            a = min(supply_left[li], demand_left[bj])
            if (B is not None):
                a = min(a, float(B))
            if a > eps:
                pairs2.append((i, j))
                amounts2.append(float(a))

    if not pairs2:
        return []

    # 2) 先算 baseline（不加边）预测风险
    model.eval()
    with torch.no_grad():
        base_pred = float(model([g.to(device) for g in (past_seq_graphs + [base_graph])]).view(-1)[0].item())

    # 3) 候选图 & 批量预测候选风险
    cand_graphs = build_candidate_graphs_for_pairs(base_graph, pairs2, amounts2)
    cand_preds = gnn_cost_of_candidates(
        model, past_seq_graphs, cand_graphs, seq_len=seq_len, device=device
    )  # shape=(len(pairs2),), 值越小越好

    # 4) score = baseline - candidate （越大越好）
    scores = (base_pred - cand_preds)

    # 可选：加一点“成交量偏好”，避免只挑风险最优但金额极小的边
    # beta_amt 取 0~0.2 之间试；amount 归一到 [0,1]
    amt_norm = np.array(amounts2, dtype=float)
    if amt_norm.size:
        denom = max(amt_norm.max(), 1e-9)
        scores = scores + beta_amt * (amt_norm / denom)

    # 5) 排序：score 高优先；同分再按金额大优先
    ranked = sorted(
        zip(scores, pairs2, amounts2),
        key=lambda x: (float(x[0]), float(x[2])),
        reverse=True
    )

    # 6) 贪心落地（仍保持你的约束：supply/demand/max_degree/B）
    plan = []
    for sc, (i, j), _nom in ranked:
        li = idx_L[i]
        bj = idx_B[j]

        if supply_left[li] <= eps or demand_left[bj] <= eps:
            continue
        if deg[i] >= max_degree or deg[j] >= max_degree:
            continue

        amt = min(supply_left[li], demand_left[bj])
        if B is not None:
            amt = min(amt, float(B))

        if amt <= eps:
            continue

        plan.append((i, j, float(amt)))
        supply_left[li] -= amt
        demand_left[bj] -= amt
        deg[i] += 1
        deg[j] += 1

    return plan


def train_model(dataset, num_epochs=50, batch_size=32, seq_len=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seq_lengths = [len(dataset.data[i]["sequence"]) for i in range(len(dataset.data))]
    if seq_lengths:
        min_len = min(seq_lengths)
        if min_len < seq_len:
            print(
                f"[train_model] cached trajectories shorter than seq_len={seq_len}; "
                f"using seq_len={min_len}"
            )
            seq_len = min_len

    model = GNNLSTMModel(input_dim=15, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    window_dataset = GraphWindowDataset(
        dataset, seq_len=seq_len, stride=1, drop_short=True
    )
    if len(window_dataset) == 0:
        raise ValueError(
            f"No training windows: need trajectories with length >= seq_len={seq_len}"
        )

    n = len(window_dataset)
    train_size = int(0.8 * n)
    test_size = n - train_size
    train_ds, test_ds = random_split(window_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graph_windows,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=collate_graph_windows,
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            windows, targets = batch
            windows_device = [
                [g.to(device) for g in w] for w in windows
            ]
            optimizer.zero_grad()
            out = model(windows_device)
            loss = criterion(out, targets.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                windows, targets = batch
                windows_device = [
                    [g.to(device) for g in w] for w in windows
                ]
                out = model(windows_device)
                test_loss += criterion(out, targets.to(device)).item()
        test_loss = test_loss / max(1, len(test_loader))

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return model


def _edges_from_pyg(g) -> list[tuple[int, int]]:
    ei = getattr(g, "edge_index", None)
    pairs: list[tuple[int, int]] = []
    if ei is None or ei.numel() == 0:
        return pairs
    for k in range(ei.size(1)):
        u = int(ei[0, k].item())
        v = int(ei[1, k].item())
        if u != v:
            pairs.append((u, v))
    return pairs


def _iter_temporal_matcher_pairs(dataset):
    """Yield (input_graph@t-1, target_graph@t) to avoid label leakage into GCN edges."""
    for item in dataset:
        if isinstance(item, tuple) and len(item) == 2:
            seq = item[0]
        else:
            seq = item
        graphs = [g for g in seq if hasattr(g, "x") and hasattr(g, "edge_index")]
        for t in range(1, len(graphs)):
            yield graphs[t - 1], graphs[t]


def train_matcher_from_dataset(dataset, epochs=3, lr=1e-3, neg_ratio=1.0, batch_graphs=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = GNNPairMatcher(in_dim=15, hid=64).to(device)
    opt = torch.optim.Adam(matcher.parameters(), lr=float(lr), weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()

    samples = list(_iter_temporal_matcher_pairs(dataset))
    if not samples:
        raise RuntimeError(
            "train_matcher_from_dataset: no temporal (t-1, t) graph pairs in dataset"
        )

    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)

    for ep in range(epochs):
        random.shuffle(samples)
        total = 0.0
        cnt = 0

        for s in range(0, len(samples), batch_graphs):
            batch = samples[s:s + batch_graphs]
            loss_acc = 0.0
            opt.zero_grad()

            for input_g, target_g in batch:
                g = input_g.to(device)
                g.x = torch.nan_to_num(g.x, nan=0.0, posinf=5.0, neginf=-5.0)
                g.x = torch.clamp(g.x, -5.0, 5.0)
                N = g.x.size(0)

                pos_pairs = _edges_from_pyg(target_g)
                if not pos_pairs:
                    continue

                pos_set = set(pos_pairs)
                num_neg = int(len(pos_pairs) * neg_ratio)
                neg_pairs = []
                while len(neg_pairs) < num_neg:
                    u = int(rng.integers(0, N))
                    v = int(rng.integers(0, N))
                    if u == v or (u, v) in pos_set:
                        continue
                    neg_pairs.append((u, v))

                pairs = pos_pairs + neg_pairs
                y = torch.tensor(
                    [1.0] * len(pos_pairs) + [0.0] * len(neg_pairs),
                    dtype=torch.float,
                    device=device,
                )

                h = matcher.node_embed(g)
                ii = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
                jj = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
                hi, hj = h[ii], h[jj]
                feat = torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=1)
                logit = matcher.edge_mlp(feat).view(-1)
                loss = bce(logit, y)
                loss.backward()
                loss_acc += float(loss.item())
                cnt += 1

            torch.nn.utils.clip_grad_norm_(matcher.parameters(), 1.0)
            opt.step()
            if cnt > 0:
                total += loss_acc

        avg = total / max(1, cnt)
        print(f"[matcher] epoch {ep+1}/{epochs} loss={avg:.4f} (temporal t-1→t)")

    return matcher

def predict_and_regulate(model, matcher, simulator, num_steps=5, seq_len=5, draw_net=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 风险预测模型：仍用于预测SR/出建议
    model = model.to(device).eval()

    # 撮合 matcher：用于 Step3 生成 a_ij（如果 matcher 是 torch.nn.Module）
    if matcher is not None and hasattr(matcher, "to"):
        matcher = matcher.to(device).eval()

    simulator.initialize_network()
    simulator._save_network_snapshot = False
    risks, pred_risks = [], []

    scenario_name = "normal"
    print("\n=== Normal Test ===")

    graph_sequence = []  # 只用于风险预测（可保留）
    for step in range(num_steps):
        # ★ 撮合只用 matcher：不再需要 past_seq
        simulator.gnn_context = {
            "matcher": matcher,
            "device": device,
            "rmax_spread": DAILY_RFQ_QUOTE_SPREAD,   # borrower cap: daily quote spread
            "beta_amt": 0.05,      # 金额偏好（可调）
        }

        # simulate_step 内部：E分角色 -> F撮合(用 matcher) -> I算SR
        risk = simulator.simulate_step(step)

        # —— 风险预测仍可做：用撮合后的图进模型 —— #
        graph = simulator.to_pyg_graph(y=risk).to(device)
        graph_sequence.append(graph)

        risks.append(float(risk))

        # 满 seq_len 才预测
        if len(graph_sequence) == seq_len:
            with torch.no_grad():
                pred_risk = float(model(graph_sequence).item())
            pred_risks.append(pred_risk)

            advisor = RegulatoryAdvisor(simulator.banks, simulator.exposure_matrix, pred_risk)
            recommendations = advisor.generate_recommendations()

            print(f"\nNormal Test Step {step+1}:")
            print(f"Actual Systemic Risk: {risk:.4f}")
            print(f"Predicted Systemic Risk: {pred_risk:.4f}")
            print("Regulatory Recommendations:")
            for rec in recommendations:
                print(rec)

            if draw_net and (step == 0 or step == num_steps - 1):
                simulator.visualize_network(step, risk, scenario_name)

            # 滑动窗口
            graph_sequence.pop(0)
        else:
            pred_risks.append(None)
            print(f"\nNormal Test Step {step+1}:")
            print(f"Actual Systemic Risk: {risk:.4f}")
            print(f"Predicted Systemic Risk: Waiting for {seq_len} timesteps")

        if simulator.network_stable_step is not None:
            break

    errors = [abs(r - p) for r, p in zip(risks, pred_risks) if p is not None]
    if errors:
        print("\nNormal Test Prediction Error Statistics:")
        print(f"Mean Error: {np.mean(errors):.4f}")
        print(f"Std Error: {np.std(errors):.4f}")

    return risks, pred_risks


# ========= 下面保持你原始结构（到你粘贴处为止） =========

_PLOT_SIMULATION_BATCH = None


@dataclass
class StepSnapshot:
    step: int
    banks: list
    exposure_matrix: np.ndarray

    policy_support_total: float = 0.0
    solvency_support_total: float = 0.0

    total_volume: float = 0.0
    num_trades: int = 0
    total_demand: float = 0.0
    total_supply: float = 0.0
    unmet_demand_rate: float = 0.0
    matching_observation: bool = True

    active_links: float = 0.0
    network_density: float = 0.0
    mean_degree: float = 0.0
    degree_std: float = 0.0
    mean_weighted_degree: float = 0.0
    weighted_degree_std: float = 0.0
    exposure_hhi: float = 0.0
    max_counterparty_exposure: float = 0.0
    largest_component_size: float = 0.0
    largest_component_share: float = 0.0

    repeated_trade_count: int = 0

    en_unpaid_amount: float = 0.0
    interbank_writeoff_amount: float = 0.0
    estate_transfer_discount_amount: float = 0.0
    funded_amount: float = 0.0
    funded_q_amount: float = 0.0
    project_default_principal: float = 0.0
    project_default_lgd_loss: float = 0.0
    project_default_q_weighted_lgd_loss: float = 0.0
    project_negative_return_loss: float = 0.0
    funding_gap: float = 0.0
    common_liquidity_shock: bool = False
    mean_lcr: float = 0.0
    is_auction_day: bool = False
    defaulted_bank_ids: list | None = None
    edges: list | None = None


@dataclass
class RecordedRun:
    seed: int
    theta_policy: float
    snapshots: list[StepSnapshot]
    rollover_enabled: bool = True
    policy_support_enabled: bool = True
    collapse_step: int | None = None
    stop_alive_threshold: int = STOP_ALIVE_THRESHOLD
    first_failures: list | None = None


@dataclass
class SimulationPlotBatch:
    T: int
    N: int
    B: float
    sigma: float
    theta_policy: float
    runs: list[RecordedRun]
    rollover_enabled: bool = True
    policy_support_enabled: bool = True


def set_plot_simulation_batch(batch: SimulationPlotBatch | None) -> SimulationPlotBatch | None:
    global _PLOT_SIMULATION_BATCH
    _PLOT_SIMULATION_BATCH = batch
    return batch


def get_plot_simulation_batch() -> SimulationPlotBatch | None:
    return _PLOT_SIMULATION_BATCH


def _resolve_plot_batch(batch: SimulationPlotBatch | None = None) -> SimulationPlotBatch | None:
    return batch if batch is not None else _PLOT_SIMULATION_BATCH


def _plot_batch_summary(batch: SimulationPlotBatch | None) -> dict:
    if batch is None:
        return {}
    runs = batch.runs
    seeds = [int(r.seed) for r in runs]
    avg_steps = float(np.mean([len(r.snapshots) for r in runs])) if runs else 0.0
    max_steps = int(max((len(r.snapshots) for r in runs), default=0))
    min_steps = int(min((len(r.snapshots) for r in runs), default=0))
    return {
        "T_cap": int(batch.T),
        "T": max_steps if max_steps > 0 else int(batch.T),
        "N": int(batch.N),
        "nsim": len(runs),
        "theta_policy": float(batch.theta_policy),
        "seed_first": int(seeds[0]) if seeds else None,
        "seed_last": int(seeds[-1]) if seeds else None,
        "avg_steps": avg_steps,
        "min_steps": min_steps,
        "max_steps": max_steps,
        "rollover_enabled": bool(getattr(batch, "rollover_enabled", True)),
        "policy_support_enabled": bool(getattr(batch, "policy_support_enabled", True)),
    }


def _plot_batch_caption(batch: SimulationPlotBatch | None) -> str:
    meta = _plot_batch_summary(batch)
    if not meta:
        return "live simulation (no shared batch)"
    return (
        f"shared batch: horizon={meta['max_steps']} "
        f"(avg={meta['avg_steps']:.0f}, cap={meta['T_cap']}), "
        f"nsim={meta['nsim']}, policy θ={meta['theta_policy']:.2f}"
    )


def _log_car_distribution_vs_theta(
    batch: SimulationPlotBatch | None,
    theta_min: float,
    theta_max: float,
) -> None:
    if batch is None or not batch.runs:
        return
    snapshots = batch.runs[0].snapshots
    if not snapshots:
        return
    snap = snapshots[-1]
    active = [
        b for b in snap.banks
        if not _is_central_bank(b) and b.get("is_active", True)
    ]
    if not active:
        return
    cars = np.asarray([_bank_measure_car(b) for b in active], dtype=float)
    th_lo, th_hi = float(theta_min), float(theta_max)
    print(
        f"[theta-sweep] CAR dist (run0, last step): "
        f"min={cars.min():.3f} max={cars.max():.3f} | "
        f"frac<{th_lo:.2f}={(cars < th_lo).mean():.3f} "
        f"frac<{th_hi:.2f}={(cars < th_hi).mean():.3f} "
        f"(若两 frac 相等 → θ 未切到 CAR 分布区间)"
    )


def _validate_theta_sweep_baseline(
    batch: SimulationPlotBatch | None,
    curves: list[np.ndarray],
    theta_grid: np.ndarray,
    baseline_theta: float,
    weights,
    track: str = "sr",
    atol: float = 1e-9,
) -> None:
    if batch is None:
        return
    track = str(track).lower()
    base_idx = int(np.argmin(np.abs(np.asarray(theta_grid, dtype=float) - float(baseline_theta))))
    y_sweep = np.asarray(curves[base_idx], dtype=float)
    comp = score_batch_mean_components(
        batch, weights=weights, theta_measure=float(baseline_theta)
    )
    y_ref = np.asarray(comp[track], dtype=float)
    n = min(len(y_sweep), len(y_ref))
    if n == 0:
        print("[theta-sweep] warn: empty baseline curve, skip consistency check")
        return
    max_diff = float(np.nanmax(np.abs(y_sweep[:n] - y_ref[:n])))
    end_sweep = float(y_sweep[n - 1])
    end_ref = float(y_ref[n - 1])
    print(
        f"[theta-sweep] batch consistency: baseline θ={baseline_theta:.2f} "
        f"max|Δ|={max_diff:.2e}, {track.upper()}_end={end_sweep:.4f} (ref={end_ref:.4f})"
    )
    if max_diff > atol:
        print(
            "[theta-sweep] warn: sweep baseline ≠ score_batch_mean_components; "
            "请勿与另一张 baseline 图横向对比"
        )


class _SnapshotSimView:
    def __init__(self, snap: StepSnapshot):
        self.banks = snap.banks
        self.num_banks = len(snap.banks)
        self.exposure_matrix = snap.exposure_matrix
        self.current_step = int(snap.step)


def record_simulation_run(
    T: int,
    seed: int,
    *,
    theta_policy: float = 0.08,
    N: int = 30,
    B: float = DEFAULT_MATCH_B,
    sigma: float = 0.3,
    matcher=None,
    device=None,
    stop_on_network_stable: bool = False,
    stop_alive_threshold: int | None = STOP_ALIVE_THRESHOLD,
    rollover_enabled: bool = True,
    policy_support_enabled: bool = True,
    centralized_cycle_length: int = 1,
) -> RecordedRun:
    """Record one run; early-stop when remaining commercial banks <= stop_alive_threshold."""
    T = min(int(T), MAX_PLOT_STEPS)
    sim = BankNetworkSimulator(num_banks=N, max_steps=T, B=B, sigma=sigma, seed=int(seed))
    configure_simulation_features(
        sim,
        rollover_enabled=rollover_enabled,
        policy_support_enabled=policy_support_enabled,
        centralized_cycle_length=centralized_cycle_length,
    )
    sim._save_network_snapshot = False
    sim.export_policy_logs = False
    sim.car_cutoff = float(theta_policy)
    sim.initialize_network()

    snapshots: list[StepSnapshot] = []
    seen_counterparties: set[tuple[int, int]] = set()
    collapse_step = None
    for s in range(T):
        if matcher is not None:
            sim.gnn_context = {
                "matcher": matcher,
                "device": device,
                "rmax_spread": DAILY_RFQ_QUOTE_SPREAD,
                "beta_amt": 0.05,
            }
        else:
            sim.gnn_context = None
        sim.simulate_step(s)
        last_policy = sim.policy_history[-1] if getattr(sim, "policy_history", None) else {}
        gross_network = (
            aggregate_contracts_to_gross_exposure_matrix_at_step(
                sim.contract_book,
                sim.num_banks,
                int(s),
            )
        )

        net_m = exposure_network_metrics(
            gross_network,
            exclude_central_bank=True,
        )
        match_stats = getattr(sim, "last_match_stats", {}) or {}
        edges = sorted(undirected_edge_set(sim.exposure_matrix))
        defaulted_ids = list(getattr(sim, "last_defaulted_banks", []) or [])

        current_trade_pairs = [
            (
                int(c.lender_idx),
                int(c.borrower_idx),
            )
            for c in sim.contract_book.contracts
            if int(c.created_step) == int(s)
            and int(c.lender_idx) != 0
            and int(c.borrower_idx) != 0
        ]

        repeated_trade_count = sum(
            1
            for pair in current_trade_pairs
            if pair in seen_counterparties
        )

        seen_counterparties.update(
            current_trade_pairs
        )

        from bank_econ_shared import SCREENING_SNAPSHOT_KEYS, screening_snapshot_fields, snapshot_mean_lcr
        snapshots.append(
            StepSnapshot(
                step=int(s),
                banks=[deepcopy(b) for b in sim.banks],
                exposure_matrix=np.array(sim.exposure_matrix, dtype=float, copy=True),
                policy_support_total=(
                    float(last_policy.get("disbursed_liquidity_support", 0.0))
                    + float(last_policy.get("support_total", 0.0))
                    + float(getattr(sim, "last_settlement_support_used", 0.0))
                ),
                solvency_support_total=(
                    float(last_policy.get("disbursed_solvency_support", 0.0))
                    + float(last_policy.get("solvency_support_total", 0.0))
                ),
                total_volume=float(match_stats.get("total_volume", 0.0)),
                num_trades=int(match_stats.get("num_trades", 0)),
                total_demand=float(
                    match_stats.get("total_demand", 0.0)
                ),
                total_supply=float(
                    match_stats.get("total_supply", 0.0)
                ),
                unmet_demand_rate=float(match_stats.get("unmet_demand_rate", 0.0)),
                matching_observation=bool(
                    match_stats.get("auction_day", True)
                ),
                active_links=float(
                    net_m["active_links"]
                ),
                network_density=float(net_m["network_density"]),
                mean_degree=float(
                    net_m["mean_degree"]
                ),
                degree_std=float(
                    net_m["degree_std"]
                ),
                mean_weighted_degree=float(
                    net_m["mean_weighted_degree"]
                ),
                weighted_degree_std=float(
                    net_m["weighted_degree_std"]
                ),
                exposure_hhi=float(net_m["exposure_hhi"]),
                max_counterparty_exposure=float(net_m["max_counterparty_exposure"]),
                largest_component_size=float(
                    net_m["largest_component_size"]
                ),
                largest_component_share=float(
                    net_m["largest_component_share"]
                ),
                repeated_trade_count=int(
                    repeated_trade_count
                ),
                en_unpaid_amount=float(getattr(sim, "last_en_unpaid", 0.0)),
                interbank_writeoff_amount=float(
                    getattr(sim, "last_interbank_writeoff", 0.0)
                ),
                estate_transfer_discount_amount=float(
                    getattr(sim, "last_estate_transfer_discount", 0.0)
                ),
                **{
                    k: float(v)
                    for k, v in screening_snapshot_fields(sim).items()
                    if k in SCREENING_SNAPSHOT_KEYS
                },
                common_liquidity_shock=bool(getattr(sim, "last_common_liquidity_shock", False)),
                mean_lcr=float(snapshot_mean_lcr(sim.banks)),
                is_auction_day=bool(match_stats.get("auction_day", getattr(sim, "last_was_auction_day", False))),
                defaulted_bank_ids=defaulted_ids,
                edges=[list(e) for e in edges],
            )
        )
        alive_noncentral = sum(
            1
            for bank in sim.banks
            if not _is_central_bank(bank)
            and bool(bank.get("is_active", True))
        )
        if (
            stop_alive_threshold is not None
            and alive_noncentral <= int(stop_alive_threshold)
        ):
            collapse_step = int(s + 1)  # 1-based time, matches plot axis
            break
        if sim.all_default_step is not None:
            if collapse_step is None:
                collapse_step = int(s + 1)
            break
        if stop_on_network_stable and sim.network_stable_step is not None:
            break
    from bank_econ_shared import format_screening_run_line
    print(format_screening_run_line(sim, seed=int(seed)))
    return RecordedRun(
        seed=int(seed),
        theta_policy=float(theta_policy),
        snapshots=snapshots,
        rollover_enabled=bool(rollover_enabled),
        policy_support_enabled=bool(policy_support_enabled),
        collapse_step=collapse_step,
        stop_alive_threshold=int(stop_alive_threshold) if stop_alive_threshold is not None else -1,
        first_failures=list(getattr(sim, "first_failure_events", None) or []),
    )


def run_simulation_plot_batch(
    T: int = DEFAULT_SIM_T_CAP,
    nsim: int = 20,
    *,
    theta_policy: float = 0.08,
    N: int = 30,
    B: float = DEFAULT_MATCH_B,
    sigma: float = 0.3,
    seed0: int = DEFAULT_RANDOM_SEED,
    matcher=None,
    device=None,
    stop_on_network_stable: bool = False,
    stop_alive_threshold: int | None = STOP_ALIVE_THRESHOLD,
    rollover_enabled: bool = True,
    policy_support_enabled: bool = True,
    centralized_cycle_length: int = 1,
) -> SimulationPlotBatch:
    """录制 nsim 条轨迹；T 为安全上限，商业银行剩余数达到阈值时早停。"""
    T = min(int(T), MAX_PLOT_STEPS)
    rng = np.random.default_rng(int(seed0))
    runs: list[RecordedRun] = []
    for _ in range(int(nsim)):
        sim_seed = int(rng.integers(0, 2**32 - 1))
        runs.append(
            record_simulation_run(
                T,
                sim_seed,
                theta_policy=float(theta_policy),
                N=N,
                B=B,
                sigma=sigma,
                matcher=matcher,
                device=device,
                stop_on_network_stable=stop_on_network_stable,
                stop_alive_threshold=stop_alive_threshold,
                rollover_enabled=rollover_enabled,
                policy_support_enabled=policy_support_enabled,
                centralized_cycle_length=centralized_cycle_length,
            )
        )
    batch = SimulationPlotBatch(
        T=int(T),
        N=int(N),
        B=float(B),
        sigma=float(sigma),
        theta_policy=float(theta_policy),
        runs=runs,
        rollover_enabled=bool(rollover_enabled),
        policy_support_enabled=bool(policy_support_enabled),
    )
    print(
        f"[plot-batch] recorded nsim={len(runs)} T={T} "
        f"theta_policy={float(theta_policy):.2f} "
        f"rollover={'on' if rollover_enabled else 'off'} "
        f"support={'on' if policy_support_enabled else 'off'} "
        f"avg_steps={np.mean([len(r.snapshots) for r in runs]):.1f} "
        f"stop_alive_threshold={stop_alive_threshold} "
        f"collapse_hit_rate={np.mean([1.0 if r.collapse_step is not None else 0.0 for r in runs]):.2f}"
    )
    return batch


def _pad_series_to_horizon(arr, horizon: int, *, fill: float | None = None) -> np.ndarray:
    """Pad a series to fixed H. fill=None repeats the last value."""
    y = np.asarray(arr, dtype=float)
    H = int(horizon)
    if H <= 0 or len(y) >= H:
        return y[:H] if H > 0 else y
    if len(y) == 0:
        return np.full(H, 1.0 if fill is None else float(fill), dtype=float)
    pad_val = float(y[-1]) if fill is None else float(fill)
    return np.concatenate([y, np.full(H - len(y), pad_val, dtype=float)])


def _coverage_plot_horizon(lengths, *, min_frac: float | None = None, pad_frac: float = 0.08, pad_min: int = 40) -> int:
    """Axis end = latest early-stop + blank margin (not T_cap)."""
    lens = [int(x) for x in lengths if int(x) > 0]
    if not lens:
        return 0
    latest = int(max(lens))
    pad = max(int(pad_min), int(round(float(pad_frac) * latest)))
    return int(latest + pad)


def _coverage_data_horizon(lengths) -> int:
    """Latest early-stop / series length (no pad)."""
    lens = [int(x) for x in lengths if int(x) > 0]
    return int(max(lens)) if lens else 0


def score_recorded_run(
    run: RecordedRun,
    weights=(0.5, 0.3, 0.2),
    theta_measure: float = 0.08,
    horizon: int | None = None,
) -> dict[str, np.ndarray]:
    sr_list, fr_list, cbs_list, cgr_list = [], [], [], []
    for snap in run.snapshots:
        view = _SnapshotSimView(snap)
        sr, fr, cbs, cgr = _components_from_state(
            view, weights=weights, theta=float(theta_measure)
        )
        sr_list.append(sr)
        fr_list.append(fr)
        cbs_list.append(cbs)
        cgr_list.append(cgr)
    out = {
        "sr": np.asarray(sr_list, dtype=float),
        "fr": np.asarray(fr_list, dtype=float),
        "cbs": np.asarray(cbs_list, dtype=float),
        "cgr": np.asarray(cgr_list, dtype=float),
    }
    if horizon is not None and int(horizon) > 0:
        H = int(horizon)
        # Absorbing FR=1 only after all-default; otherwise hold last FR.
        last_fr = float(out["fr"][-1]) if len(out["fr"]) else 1.0
        fr_fill = 1.0 if last_fr >= 1.0 - 1e-12 else last_fr
        out["fr"] = _pad_series_to_horizon(out["fr"], H, fill=fr_fill)
        out["sr"] = _pad_series_to_horizon(out["sr"], H, fill=None)
        out["cbs"] = _pad_series_to_horizon(out["cbs"], H, fill=None)
        out["cgr"] = _pad_series_to_horizon(out["cgr"], H, fill=None)
    return out


def survival_metrics_from_fr(fr, horizon: int | None = None, *, n_commercial: int = 29) -> dict:
    """Fixed-horizon survival summary: mean_alive, t50, alive_H, t_first.

    bank_days_alive = sum_t (1-FR_t) * N_commercial  (bank-days, not share-days).
    """
    y = np.asarray(fr, dtype=float)
    if horizon is not None and int(horizon) > 0:
        last_fr = float(y[-1]) if len(y) else 1.0
        fr_fill = 1.0 if last_fr >= 1.0 - 1e-12 else last_fr
        y = _pad_series_to_horizon(y, int(horizon), fill=fr_fill)
    alive = 1.0 - y
    mean_alive = float(np.mean(alive)) if len(alive) else float("nan")
    alive_H = float(alive[-1]) if len(alive) else float("nan")
    t50_hits = np.where(y >= 0.5)[0]
    t50 = int(t50_hits[0] + 1) if len(t50_hits) else None
    t90_hits = np.where(y >= 0.9)[0]
    t90 = int(t90_hits[0] + 1) if len(t90_hits) else None
    t_first_hits = np.where(y > 1e-12)[0]
    t_first = int(t_first_hits[0] + 1) if len(t_first_hits) else None
    n_comm = max(1, int(n_commercial))
    return {
        "mean_alive": mean_alive,
        "bank_days_alive": float(np.sum(alive) * n_comm) if len(alive) else float("nan"),
        "alive_H": alive_H,
        "t50": t50,
        "t50_hit": bool(t50 is not None),
        "t90": t90,
        "t90_hit": bool(t90 is not None),
        "t_first": t_first,
        "horizon": int(len(y)),
    }


def score_batch_mean_components(
    batch: SimulationPlotBatch,
    weights=(0.5, 0.3, 0.2),
    theta_measure: float = 0.08,
) -> dict[str, np.ndarray]:
    sr_mat, fr_mat, cbs_mat, cgr_mat = [], [], [], []
    for run in batch.runs:
        comp = score_recorded_run(run, weights=weights, theta_measure=float(theta_measure), horizon=None)
        sr_mat.append(comp["sr"])
        fr_mat.append(comp["fr"])
        cbs_mat.append(comp["cbs"])
        cgr_mat.append(comp["cgr"])
    return {
        "sr": np.asarray(_nanmean_variable_length(sr_mat), dtype=float).copy(),
        "fr": np.asarray(_nanmean_variable_length(fr_mat), dtype=float).copy(),
        "cbs": np.asarray(_nanmean_variable_length(cbs_mat), dtype=float).copy(),
        "cgr": np.asarray(_nanmean_variable_length(cgr_mat), dtype=float).copy(),
    }


def score_batch_component_stats(
    batch: SimulationPlotBatch,
    weights=(0.5, 0.3, 0.2),
    theta_measure: float = 0.08,
) -> dict[str, dict[str, np.ndarray]]:

    series = {
        "sr": [],
        "fr": [],
        "cbs": [],
        "cgr": [],
    }

    for run in batch.runs:
        comp = score_recorded_run(
            run,
            weights=weights,
            theta_measure=float(theta_measure),
            horizon=None,
        )

        for key in series:
            series[key].append(comp[key])

    result = {}

    for key, values in series.items():
        mean, std, lower, upper = _mean_ci95(values)

        result[key] = {
            "mean": mean,
            "std": std,
            "ci95_lower": lower,
            "ci95_upper": upper,
        }

    return result


def policy_support_mean_series(batch: SimulationPlotBatch) -> dict[str, np.ndarray]:
    liquidity = []
    solvency = []
    total = []
    for run in batch.runs:
        liq = np.asarray([s.policy_support_total for s in run.snapshots], dtype=float)
        sol = np.asarray([s.solvency_support_total for s in run.snapshots], dtype=float)
        liquidity.append(liq)
        solvency.append(sol)
        total.append(liq + sol)
    return {
        "liquidity": _nanmean_variable_length(liquidity),
        "solvency": _nanmean_variable_length(solvency),
        "total": _nanmean_variable_length(total),
    }


def run_sensitivity_analysis(batch: SimulationPlotBatch | None = None):
    """
    生成 4 张单图 + 1 张 2×2 面板。
    Δt 采用“相对平台阈值”：t@0.9·S∞ − t@0.5·S∞。
    若传入 batch：仅在固定 policy 轨迹上重算 W / θ_measure。
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime

    batch = _resolve_plot_batch(batch)
    plt.close('all')
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    w1_grid    = np.asarray(measure_w1_grid(), dtype=float)
    theta_grid = np.asarray(measure_theta_grid(), dtype=float)
    baseline_w = (0.5, 0.3, 0.2)
    baseline_t = 0.08
    T_steps    = 1000

    RELATIVE_PLATEAU = True

    def _first_cross_time_local(y, level, clip01=False, monotone=True):
        y = np.asarray(y, dtype=float)

        if clip01:
            y = np.clip(y, 0.0, 1.0)

        if monotone:
            y = np.maximum.accumulate(y)

        t = np.arange(len(y), dtype=float)

        if len(y) == 0 or y[-1] < level:
            return np.nan

        k = int(np.argmax(y >= level))

        if k == 0:
            return 0.0

    # ---- linear interpolation between (k-1) and k ----
        y0, y1 = y[k-1], y[k]
        t0, t1 = t[k-1], t[k]
        return t0 + (level - y0) * (t1 - t0) / (y1 - y0 + 1e-12)


    def _metrics(sr):
        sr = np.asarray(sr, dtype=float)
        sr = np.clip(sr, 0.0, 1.0)
        t  = np.arange(len(sr), dtype=float)
        auc01 = float(np.trapz(sr, t) / (t[-1] - t[0])) if len(sr) > 1 else float(sr[0])

        if RELATIVE_PLATEAU:
            s_inf = float(np.nanmedian(sr[-5:]))
            if s_inf <= 1e-9:
                s_inf = float(np.nanmax(sr))
            lo, hi = 0.5 * s_inf, 0.9 * s_inf
        else:
            lo, hi = 0.5, 0.9

        t_lo = _first_cross_time_local(sr, lo)
        t_hi = _first_cross_time_local(sr, hi)
        dt   = (t_hi - t_lo) if (np.isfinite(t_lo) and np.isfinite(t_hi)) else np.nan
        return auc01, dt

    def run_series(weights, theta_measure, theta_policy=0.08, N=30, T=T_steps, B=DEFAULT_MATCH_B, sigma=0.3):
        if batch is not None:
            if float(theta_policy) != float(batch.theta_policy):
                print(
                    f"[plot-batch] sensitivity: ignore theta_policy={theta_policy:.2f}, "
                    f"use batch policy={batch.theta_policy:.2f}"
                )
            comp = score_batch_mean_components(
                batch, weights=weights, theta_measure=float(theta_measure)
            )
            return np.asarray(comp["sr"], dtype=float)

        sim = BankNetworkSimulator(num_banks=N, max_steps=T, B=B, sigma=sigma)

        # ★ policy θ：影响角色分配/网络生成（sim.assign_roles_by_risk 用的 car_cutoff）
        sim.car_cutoff = float(theta_policy)
        sim._save_network_snapshot = False
        sim.export_policy_logs = False

        sim.initialize_network()

        sr = []
        for step in range(T):
            sim.simulate_step(step)

            # ★ measure θ：只影响 SR 计算（CBS/CGR 的阈值）
            current_sr = sim.calculate_systemic_risk(weights=weights, car_threshold=float(theta_measure))
            sr.append(current_sr)
            # 不提前停止：完整运行到 T（保留 _scenario_stable_for_sweep 定义供调试）
        return np.asarray(sr, float)


    w1_auc, w1_dt, th_auc, th_dt = [], [], [], []
    w1_base, w2_base, w3_base = 0.5, 0.3, 0.2   # 你 baseline 是多少就填多少
    den = (w2_base + w3_base)
    for w1 in w1_grid:
        delta = w1 - w1_base
        w2 = w2_base - delta * (w2_base / den)
        w3 = w3_base - delta * (w3_base / den)
        auc, dt = _metrics(run_series((w1, w2, w3), baseline_t, theta_policy=baseline_t))
        w1_auc.append(auc); w1_dt.append(dt)

    for th in theta_grid:
        auc, dt = _metrics(run_series(baseline_w, th, theta_policy=th))
        th_auc.append(auc); th_dt.append(dt)


    if RELATIVE_PLATEAU:
        dt_line_w1   = r"Δt (90%–50% of plateau) vs w1"
        dt_line_th   = r"Δt (90%–50% of plateau) vs θ"
        dt_panel_w1  = r"$\Delta t=t_{0.9S_\infty}-t_{0.5S_\infty}$ vs $w_1$"
        dt_panel_th  = r"$\Delta t=t_{0.9S_\infty}-t_{0.5S_\infty}$ vs \theta$"
        suffix = "plateau"
    else:
        dt_line_w1   = r"Δt (t₀․₉−t₀․₅) vs w1"
        dt_line_th   = r"Δt (t₀․₉−t₀․₅) vs θ"
        dt_panel_w1  = r"$\Delta t=t_{0.9}-t_{0.5}$ vs $w_1$"
        dt_panel_th  = r"$\Delta t=t_{0.9}-t_{0.5}$ vs \theta$"
        suffix = "abs"

    def _save_line(x, y, title, xlabel, outfile):
        fig = plt.figure(figsize=(6, 4))
        plt.plot(x, y, marker='o')
        plt.title(title)
        plt.xlabel(xlabel); plt.ylabel('value'); plt.grid(True)
        path = FIG_DIR / outfile
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
        plt.show(); plt.close(fig)
        print(f"Saved: {path}")

    _save_line(w1_grid,   w1_auc, "AUC vs w1", "w1", f"auc_w1_{run_id}.png")
    _save_line(theta_grid, th_auc, "AUC vs θ",  "θ",  f"auc_theta_{run_id}.png")
    _save_line(w1_grid,   w1_dt,  dt_line_w1,  "w1", f"dt_w1_{suffix}_{run_id}.png")
    _save_line(theta_grid, th_dt,  dt_line_th,  "θ",  f"dt_theta_{suffix}_{run_id}.png")

    def _safe_imshow(ax, M, title, xticks, xlabel, cmap):
        M = np.asarray(M, float)[None, :]
        valid = np.isfinite(M)
        if valid.any():
            vmin = float(np.nanpercentile(M, 5)); vmax = float(np.nanpercentile(M, 95))
            im = ax.imshow(M, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax)
        else:
            ax.imshow(np.zeros_like(M), origin='lower', aspect='auto', cmap='Greys', vmin=0, vmax=1)
            ax.text(0.5, 0.5, 'no valid data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='red')
        ax.set_title(title)
        ax.set_yticks([0]); ax.set_yticklabels([''])
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels([f"{x:.2f}" for x in xticks], rotation=45, ha='right')
        ax.set_xlabel(xlabel)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _safe_imshow(axes[0,0], w1_auc, "AUC vs $w_1$",     w1_grid,    "w1", 'viridis')
    _safe_imshow(axes[0,1], th_auc,  "AUC vs $\\theta$", theta_grid, "θ",  'viridis')
    _safe_imshow(axes[1,0], w1_dt,   dt_panel_w1,       w1_grid,    "w1", 'magma')
    _safe_imshow(axes[1,1], th_dt,   dt_panel_th,       theta_grid, "θ",  'magma')

    fig.suptitle("Sensitivity Summary (AUC & Rise Window)", fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / f"sensitivity_summary_{suffix}_{run_id}.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show(); plt.close(fig)


def _first_cross_time(y, level):
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0.0, 1.0)
    y = np.maximum.accumulate(y)
    t = np.arange(len(y), dtype=float)
    if y[-1] < level:
        return np.nan
    k = int(np.argmax(y >= level))
    if k == 0:
        return 0.0
    y0, y1, t0, t1 = y[k-1], y[k], t[k-1], t[k]
    return t0 + (level - y0) * (t1 - t0) / (y1 - y0 + 1e-12)


def generate_network_snapshots(
    steps=(10, 20, 30, 40, 50),
    tag="normal",
    show=True,
    matcher=None,
    device=None,
    rollover_enabled: bool = True,
    policy_support_enabled: bool = True,
):
    """先跑完模拟并缓存指定 step 的状态，再统一生成网络图。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sim = BankNetworkSimulator(max_steps=max(steps) + 1)
    configure_simulation_features(
        sim,
        rollover_enabled=rollover_enabled,
        policy_support_enabled=policy_support_enabled,
    )
    sim._save_network_snapshot = False
    sim.export_policy_logs = False
    sim.initialize_network()

    steps_set = set(steps)
    snapshots = []
    for s in range(max(steps) + 1):
        if matcher is not None:
            sim.gnn_context = {
                "matcher": matcher,
                "device": device,
                "rmax_spread": DAILY_RFQ_QUOTE_SPREAD,
                "beta_amt": 0.05,
            }
        else:
            sim.gnn_context = None

        risk = sim.simulate_step(s)

        if s in steps_set:
            snapshots.append({
                "step": int(s),
                "risk": float(risk),
                "banks": deepcopy(sim.banks),
                "exposure_matrix": np.array(sim.exposure_matrix, dtype=float, copy=True),
            })
        if sim.network_stable_step is not None:
            break

    saved = []
    current_banks = sim.banks
    current_exposure = sim.exposure_matrix
    try:
        for snap in snapshots:
            sim.banks = deepcopy(snap["banks"])
            sim.exposure_matrix = np.array(snap["exposure_matrix"], dtype=float, copy=True)
            sim.visualize_network(
                step=snap["step"], risk=snap["risk"], tag=tag, save=True, show_first=show,
            )
            path = FIG_DIR / f"network_{tag}_step{snap['step']}.png"
            print(f"Saved: {path}")
            saved.append(path)
    finally:
        sim.banks = current_banks
        sim.exposure_matrix = current_exposure
    return saved



def _components_from_state(sim, weights=(0.5, 0.3, 0.2), theta=0.08):
    return _systemic_risk_from_banks(
        sim.banks, weights=weights, car_threshold=float(theta)
    )


def _nanmean_variable_length(series_list):
    non_empty = [np.asarray(x, dtype=float) for x in series_list if len(x) > 0]
    if not non_empty:
        return np.asarray([], dtype=float)
    max_len = max(len(x) for x in non_empty)
    mat = np.full((len(non_empty), max_len), np.nan, dtype=float)
    for i, arr in enumerate(non_empty):
        mat[i, : len(arr)] = arr
    return np.nanmean(mat, axis=0)


def _stack_series(series_list):
    non_empty = [
        np.asarray(x, dtype=float)
        for x in series_list
        if len(x) > 0
    ]

    if not non_empty:
        return np.empty((0, 0), dtype=float)

    max_len = max(len(x) for x in non_empty)

    mat = np.full(
        (len(non_empty), max_len),
        np.nan,
        dtype=float,
    )

    for i, arr in enumerate(non_empty):
        mat[i, :len(arr)] = arr

    return mat


def _mean_ci95(series_list):
    """
    Monte Carlo mean and pointwise 95% confidence interval.
    """
    mat = _stack_series(series_list)

    if mat.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty

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

    # 95% CI; adequate for the Monte Carlo summaries used here.
    half_width = 1.96 * se

    lower = mean - half_width
    upper = mean + half_width

    return mean, std, lower, upper


def _scenario_stable_for_sweep(
    sim,
    risk,
    window=50,
    exposure_tol=5e-3,
    risk_tol=0.01,
    min_step=None,
):
    """Sweep 用：按当前场景的系统状态判断是否连续稳定。"""
    step = int(getattr(sim, "current_step", 0))
    window = max(1, int(window))
    if min_step is None:
        min_step = window
    L = np.asarray(
        getattr(sim, "exposure_matrix", np.zeros((sim.num_banks, sim.num_banks))),
        dtype=float,
    )
    active = tuple(bool(b.get("is_active", True)) for b in getattr(sim, "banks", []))

    prev_L = getattr(sim, "_sweep_prev_stability_exposure_matrix", None)
    prev_risk = getattr(sim, "_sweep_prev_stability_risk", None)
    prev_active = getattr(sim, "_sweep_prev_stability_active", None)
    if prev_L is None or prev_risk is None or prev_active is None:
        sim._sweep_prev_stability_exposure_matrix = L.copy()
        sim._sweep_prev_stability_risk = float(risk)
        sim._sweep_prev_stability_active = active
        sim._sweep_stable_count = 0
        return False

    denom = max(float(np.linalg.norm(prev_L)), float(np.linalg.norm(L)), 1.0)
    exposure_change = float(np.linalg.norm(L - prev_L) / denom)
    risk_change = abs(float(risk) - float(prev_risk))
    active_changed = active != prev_active

    if (
        step >= int(min_step)
        and not active_changed
        and exposure_change <= float(exposure_tol)
        and risk_change <= float(risk_tol)
    ):
        sim._sweep_stable_count = int(getattr(sim, "_sweep_stable_count", 0)) + 1
    else:
        sim._sweep_stable_count = 0

    sim._sweep_prev_stability_exposure_matrix = L.copy()
    sim._sweep_prev_stability_risk = float(risk)
    sim._sweep_prev_stability_active = active

    return sim._sweep_stable_count >= window


def plot_baseline_trajectory(
    T=DEFAULT_SIM_T_CAP,
    weights=(0.5, 0.3, 0.2),
    theta=0.08,
    seed=None,
    batch: SimulationPlotBatch | None = None,
):
    """
    画基准情景的 SR & 组成（FR/CBS/CGR）随时间的轨迹图。
    """
    T = min(int(T), MAX_PLOT_STEPS)
    batch = _resolve_plot_batch(batch)
    if batch is not None:
        stats = score_batch_component_stats(
            batch,
            weights=weights,
            theta_measure=float(theta),
        )

        sr_list = list(stats["sr"]["mean"])
        fr_list = list(stats["fr"]["mean"])
        cbs_list = list(stats["cbs"]["mean"])
        cgr_list = list(stats["cgr"]["mean"])

        sr_ci_low = np.asarray(
            stats["sr"]["ci95_lower"],
            dtype=float,
        )
        sr_ci_high = np.asarray(
            stats["sr"]["ci95_upper"],
            dtype=float,
        )
    else:
        run_seed = DEFAULT_RANDOM_SEED if seed is None else set_random_seed(seed)

        sim = BankNetworkSimulator(max_steps=T, seed=run_seed)
        sim._save_network_snapshot = False
        sim.export_policy_logs = False
        sim.car_cutoff = theta
        sim.initialize_network()

        sr_list, fr_list, cbs_list, cgr_list = [], [], [], []
        for s in range(T):
            sim.simulate_step(s)
            sr, fr, cbs, cgr = _components_from_state(sim, weights=weights, theta=theta)
            sr_list.append(sr); fr_list.append(fr); cbs_list.append(cbs); cgr_list.append(cgr)
            # 不提前停止：完整运行到 T（保留 _scenario_stable_for_sweep 定义供调试）
    xs = np.arange(1, len(sr_list) + 1)

    plt.close('all')
    fig = plt.figure(figsize=(12, 7))
    ax  = plt.gca()

    ax.plot(xs, sr_list,  lw=2.2, marker='o', ms=4, label='SR (Systemic Risk)')
    if batch is not None:
        ax.fill_between(
            xs,
            sr_ci_low,
            sr_ci_high,
            alpha=0.18,
            label="SR 95% CI",
        )
    ax.plot(xs, fr_list,  lw=1.8, marker='.', ms=3, label='FR (Failure Rate)')
    ax.plot(xs, cbs_list, lw=1.8, marker='.', ms=3, label='CBS (active CAR<θ / active)')
    ax.plot(xs, cgr_list, lw=1.8, marker='.', ms=3, label='CGR (gap / required)')

    sr_end = float(sr_list[-1]) if sr_list else float("nan")
    horizon = int(len(sr_list))
    ax.set_title(
        f"Baseline Trajectory — SR and Components over Time\n"
        f"W={weights}, θ={theta} | horizon={horizon} | SR_end={sr_end:.4f}\n"
        f"{_plot_batch_caption(batch)}"
    )
    ax.set_xlabel(f"Time Step (1–{horizon})")
    ax.set_ylabel("Value (0–1)")
    ax.set_ylim(-0.02, 1.02)
    if horizon > 0:
        ax.set_xlim(1, horizon)
    ax.grid(True, alpha=0.4)
    ax.legend()

    out = FIG_DIR / "baseline_trajectory.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_scenario_comparison(T=DEFAULT_SIM_T_CAP, batch: SimulationPlotBatch | None = None):
    """
    生成情景对比折线图 + t@0.5 竖线。
    """
    T = min(int(T), MAX_PLOT_STEPS)
    batch = _resolve_plot_batch(batch)
    scenarios = [
        ("Baseline",             (0.5, 0.3, 0.2), 0.08, "-"),
        ("High Failure Weight",  (0.7, 0.18, 0.12), 0.08, "-"),
        ("Strict CAR Threshold", (0.5, 0.3, 0.2), 0.10, "-"),
    ]

    def _run_series(weights, theta, T):
        if batch is not None:
            comp = score_batch_mean_components(
                batch, weights=weights, theta_measure=float(theta)
            )
            return np.asarray(comp["sr"], dtype=float)

        sim = BankNetworkSimulator(max_steps=T)
        sim._save_network_snapshot = False
        sim.export_policy_logs = False
        sim.initialize_network()
        sr = []
        for s in range(T):
            sim.simulate_step(s)
            current_sr = sim.calculate_systemic_risk(weights=weights, car_threshold=theta)
            sr.append(current_sr)
            # 不提前停止：完整运行到 T（保留 _scenario_stable_for_sweep 定义供调试）
        return np.asarray(sr, float)

    plt.close('all')
    fig = plt.figure(figsize=(14, 8))
    ax = plt.gca()

    for name, w, th, ls in scenarios:
        sr = _run_series(w, th, T)
        xs = np.arange(1, len(sr) + 1)
        line, = ax.plot(xs, sr, ls=ls, marker='o', ms=4,
                        label=f"{name} (W={w}|θ={th})")

        t05 = _first_cross_time(sr, 0.5)
        if np.isfinite(t05):
            ax.axvline(x=t05, color=line.get_color(), linestyle=':', alpha=0.6)
            ax.text(t05, 0.5, "t₀․₅", color=line.get_color(),
                    ha='left', va='bottom', fontsize=9, alpha=0.8)

        ax.annotate(
            f"{name}\nStep={int(xs[-1])}, SR={sr[-1]:.3f}",
            xy=(xs[-1], sr[-1]),
            xytext=(8, 8),
            textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            fontsize=9,
        )

    ax.set_title("Scenario Comparison (hover lines for values)")
    max_h = 0
    for line in ax.get_lines():
        xdata = line.get_xdata()
        if len(xdata):
            max_h = max(max_h, int(np.nanmax(xdata)))
    ax.set_xlabel(f"Time Step (1–{max_h})" if max_h else "Time Step")
    ax.set_ylabel("Systemic Risk Score")
    ax.grid(True)
    ax.legend()

    try:
        import mplcursors  # type: ignore
        cursor = mplcursors.cursor(hover=True)

        @cursor.connect("add")
        def _on_add(sel):
            line = sel.artist
            x, y = line.get_data(); i = sel.index
            sel.annotation.set_text(f"{line.get_label()}\nStep={int(x[i])}, SR={y[i]:.3f}")
    except Exception:
        pass

    out = FIG_DIR / "scenario_comparison_annotated.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {out}")
    return out

def run_with_gnn_matching(model, T=200, seq_len=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    sim = BankNetworkSimulator(num_banks=30, max_steps=T)
    sim._save_network_snapshot = False
    sim.export_policy_logs = False
    sim.initialize_network()

    graph_seq = []

    for step in range(T):
        # ★ Step3 撮合前塞 context：past_seq 用最近 seq_len-1 个图
        if len(graph_seq) >= seq_len - 1:
            sim.gnn_context = {
                "model": model,
                "past_seq": graph_seq[-(seq_len - 1):],
                "seq_len": seq_len,
                "device": device,
            }
        else:
            sim.gnn_context = None

        # simulate_step 内部：E分角色 -> F撮合(用gnn) -> I算SR
        sr = sim.simulate_step(step)

        # ★ 保存本步“撮合后的图”，供下一步做 past_seq
        g = sim.to_pyg_graph(y=sr).to(device)
        graph_seq.append(g)
        if sim.network_stable_step is not None:
            break

    return sim

def _run_series_mean_components(
    weights,
    theta_measure,
    theta_policy=0.08,
    T=50,
    N=30,
    B=DEFAULT_MATCH_B,
    sigma=0.3,
    nsim=20,
    seed0=DEFAULT_RANDOM_SEED,
    batch: SimulationPlotBatch | None = None,
    use_shared_batch: bool = True,
    matcher=None,
    device=None,
    rollover_enabled=True,
    policy_support_enabled=True,
):
    """
    跑 nsim 次，返回每期 SR/FR/CBS/CGR 的均值轨迹（长度=实际跑到的步数，可变长）。
    T 仅为安全上限；与主实验一致：商业银行只剩 STOP_ALIVE_THRESHOLD 家时早停。
    theta_measure: 只影响“评分口径”（CBS/CGR 的阈值）
    theta_policy : 影响仿真过程（角色分配/网络生成用的 car_cutoff）
    batch: 若提供且 policy θ 一致，则只重算分量，不重跑仿真。
    use_shared_batch: False 时不读取全局 _PLOT_SIMULATION_BATCH（policy sweep 用）。
    """
    import numpy as np

    T = min(int(T), MAX_PLOT_STEPS)
    if use_shared_batch:
        batch = _resolve_plot_batch(batch)
    if batch is not None and float(theta_policy) != float(batch.theta_policy):
        batch = None
    if batch is not None:
        return score_batch_mean_components(
            batch, weights=weights, theta_measure=float(theta_measure)
        )

    key = (
        tuple(float(x) for x in weights),
        float(theta_measure),
        float(theta_policy),
        int(T),
        int(N),
        float(B),
        float(sigma),
        int(nsim),
        int(seed0),
        bool(rollover_enabled),
        bool(policy_support_enabled),
    )
    cached = _MEAN_COMPONENTS_CACHE.get(key)
    if cached is not None:
        return {k: np.asarray(v, dtype=float).copy() for k, v in cached.items()}

    sr_mat, fr_mat, cbs_mat, cgr_mat = [], [], [], []
    rng = np.random.default_rng(seed0)

    for k in range(nsim):
        sim_seed = int(rng.integers(0, 2**32 - 1))
        sim = BankNetworkSimulator(num_banks=N, max_steps=T, B=B, sigma=sigma, seed=sim_seed)

        configure_simulation_features(
            sim,
            rollover_enabled=rollover_enabled,
            policy_support_enabled=policy_support_enabled,
        )

        # policy θ：影响行为/网络
        sim.car_cutoff = float(theta_policy)
        sim._save_network_snapshot = False
        sim.export_policy_logs = False

        sim.initialize_network()

        sr_list, fr_list, cbs_list, cgr_list = [], [], [], []
        for s in range(T):
            if matcher is not None:
                sim.gnn_context = {
                    "matcher": matcher,
                    "device": device,
                    "rmax_spread": DAILY_RFQ_QUOTE_SPREAD,
                    "beta_amt": 0.05,
                }
            else:
                sim.gnn_context = None

            sim.simulate_step(s)

            # 用你现成的分解函数，口径 θ = theta_measure
            sr, fr, cbs, cgr = _components_from_state(
                sim, weights=weights, theta=float(theta_measure)
            )
            sr_list.append(sr); fr_list.append(fr); cbs_list.append(cbs); cgr_list.append(cgr)
            alive_noncentral = sum(
                1
                for b in sim.banks
                if (not _is_central_bank(b)) and b.get("is_active", True)
            )
            if alive_noncentral <= int(STOP_ALIVE_THRESHOLD):
                break
        sr_mat.append(sr_list)
        fr_mat.append(fr_list)
        cbs_mat.append(cbs_list)
        cgr_mat.append(cgr_list)

    out = {
        "sr":  _nanmean_variable_length(sr_mat),
        "fr":  _nanmean_variable_length(fr_mat),
        "cbs": _nanmean_variable_length(cbs_mat),
        "cgr": _nanmean_variable_length(cgr_mat),
    }
    _MEAN_COMPONENTS_CACHE[key] = {
        "sr": np.asarray(out["sr"], dtype=float).copy(),
        "fr": np.asarray(out["fr"], dtype=float).copy(),
        "cbs": np.asarray(out["cbs"], dtype=float).copy(),
        "cgr": np.asarray(out["cgr"], dtype=float).copy(),
    }
    return out


def _theta_sweep_pairwise_spread(curves: list[np.ndarray]) -> float:
    valid = [np.asarray(y, dtype=float) for y in curves if len(y) > 0]
    if len(valid) < 2:
        return 0.0
    min_len = min(len(y) for y in valid)
    mat = np.vstack([y[:min_len] for y in valid])
    return float(np.nanmax(np.nanmax(mat, axis=0) - np.nanmin(mat, axis=0)))


def plot_theta_sweep_lines(
    T=DEFAULT_SIM_T_CAP,
    weights=(0.5, 0.3, 0.2),
    nsim=20,
    theta_policy_fixed=None,
    theta_min=0.08,
    theta_max=0.15,
    n_theta=8,
    track="sr",
    baseline_theta=0.08,
    title_prefix="CAR-threshold sweep",
    output_prefix="theta_sweep_lines",
    batch: SimulationPlotBatch | None = None,
    show_delta_panel: bool = True,
    annotate_ends: bool = True,
    print_spread: bool = True,
    matcher=None,
    device=None,
    rollover_enabled=True,
    policy_support_enabled=True,
):
    """
    固定 W，遍历 θ，每个 θ 一条曲线；baseline θ 加粗+白描边。
    轨迹可因「网络稳定」提前结束，横轴按实际长度绘制。
    若传入 batch：仅在固定 policy 轨迹上扫 θ_measure。
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.patheffects as pe

    track = str(track).lower()
    assert track in ("sr", "cbs", "fr", "cgr")
    T = min(int(T), MAX_PLOT_STEPS)
    # Policy-follow re-sim uses nsim as-is (not max(50, nsim)).
    nsim = int(nsim)

    baseline_theta = float(baseline_theta)
    theta_grid = np.asarray(
        measure_theta_grid(
            theta_min, theta_max, n_theta=n_theta, baseline_theta=baseline_theta
        ),
        dtype=float,
    )

    policy_follow = theta_policy_fixed is None
    batch = _resolve_plot_batch(batch)
    if policy_follow:
        batch = None
        _MEAN_COMPONENTS_CACHE.clear()
        print("[theta-sweep] policy-follow: each θ re-runs simulation (car_cutoff = θ)")
    elif batch is not None:
        print(
            f"[plot-batch] measure-only sweep on shared batch, "
            f"policy θ={batch.theta_policy:.2f}"
        )
        theta_policy_fixed = float(batch.theta_policy)

    curves = []
    for th in theta_grid:
        theta_policy = th if theta_policy_fixed is None else float(theta_policy_fixed)
        comp = _run_series_mean_components(
            weights=weights,
            theta_measure=th,
            theta_policy=theta_policy,
            T=T,
            nsim=nsim,
            batch=batch,
            use_shared_batch=not policy_follow,
            matcher=matcher,
            device=device,
            rollover_enabled=rollover_enabled,
            policy_support_enabled=policy_support_enabled,
        )
        curves.append(np.asarray(comp[track], float))

    if print_spread:
        spread = _theta_sweep_pairwise_spread(curves)
        print(
            f"[theta-sweep] track={track} max pairwise spread={spread:.2e} "
            f"(≈0 表示曲线数值重合)"
        )
        if spread <= 1e-12:
            _log_car_distribution_vs_theta(batch, theta_min, theta_max)

    _validate_theta_sweep_baseline(
        batch, curves, theta_grid, baseline_theta, weights, track=track
    )

    ylab = {
        "sr": "Systemic Risk (SR)",
        "cbs": "CBS (active CAR<θ / active)",
        "fr": "FR (Failure Rate)",
        "cgr": "CGR (gap / required)",
    }[track]
    ttl = {"sr": "SR", "cbs": "CBS", "fr": "FR", "cgr": "CGR"}[track]

    base_idx = int(np.argmin(np.abs(theta_grid - baseline_theta)))
    y_ref = curves[base_idx] if len(curves[base_idx]) > 0 else None

    plt.close("all")
    if show_delta_panel and y_ref is not None and len(y_ref) > 1:
        fig, (ax, ax_delta) = plt.subplots(
            2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
        )
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax_delta = None

    cmap = mpl.colormaps["plasma"].reversed()
    norm = mpl.colors.Normalize(vmin=float(theta_grid.min()), vmax=float(theta_grid.max()))
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    for i, (th, y) in enumerate(zip(theta_grid, curves)):
        if len(y) == 0:
            continue
        xs = np.arange(1, len(y) + 1)
        is_base = np.isclose(th, baseline_theta, atol=1e-12)
        color = cmap(norm(th))
        ls = linestyles[i % len(linestyles)]
        lw = 3.0 if is_base else 1.8
        z = 10 if is_base else 2
        a = 1.0 if is_base else 0.92
        line, = ax.plot(xs, y, lw=lw, alpha=a, color=color, linestyle=ls, zorder=z)
        if is_base:
            line.set_path_effects([
                pe.Stroke(linewidth=6.5, foreground="white", alpha=0.95),
                pe.Normal(),
            ])
            line.set_label(rf"baseline $\theta={baseline_theta:.2f}$")

        if ax_delta is not None and y_ref is not None:
            n = min(len(y), len(y_ref))
            delta = np.asarray(y[:n], float) - np.asarray(y_ref[:n], float)
            ax_delta.plot(
                xs[:n], delta, lw=1.4 if not is_base else 2.2, alpha=a,
                color=color, linestyle=ls, zorder=z,
            )

        if annotate_ends:
            ax.annotate(
                rf"$\theta={th:.2f}$",
                xy=(xs[-1], y[-1]),
                xytext=(6, 6 + (i % 5) * 10),
                textcoords="offset points",
                fontsize=7,
                color=color,
                alpha=0.9,
                clip_on=True,
            )

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label(r"$\theta$ (CAR cutoff)")

    ax.set_title(
        rf"{title_prefix} — {ttl}$_t$ for each $\theta$  $(\mathbf{{W}}={weights})$"
        + (
            "\npolicy-follow: resim per θ (car_cutoff = θ)"
            if policy_follow
            else f"\n{_plot_batch_caption(batch)}"
        )
    )
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.4)
    if any(np.isclose(theta_grid, baseline_theta, atol=1e-12)):
        ax.legend(loc="best")

    if ax_delta is not None:
        ax_delta.axhline(0.0, color="0.45", lw=1.0, ls=":")
        ax_delta.set_ylabel(rf"$\Delta${ttl} vs $\theta={baseline_theta:.2f}$")
        ax_delta.grid(True, alpha=0.35)
        ax.set_xlabel("")
        ax_delta.set_xlabel("Time Step")
    else:
        ax.set_xlabel("Time Step")

    policy_tag = (
        "policy-follow"
        if theta_policy_fixed is None
        else f"policy-fixed{float(theta_policy_fixed):.2f}"
    )
    delta_tag = "_delta" if ax_delta is not None else ""
    out = (
        FIG_DIR
        / f"{output_prefix}_{track}_theta{theta_min:.2f}-{theta_max:.2f}_n{n_theta}"
        f"_baseline{baseline_theta:.2f}_{policy_tag}{delta_tag}.png"
    )
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_theta_measure_sweep_lines(
    T=DEFAULT_SIM_T_CAP,
    weights=(0.5, 0.3, 0.2),
    nsim=20,
    theta_policy=0.08,
    theta_min=0.08,
    theta_max=0.15,
    n_theta=8,
    track="sr",
    batch: SimulationPlotBatch | None = None,
):
    """固定政策阈值，仅改变 SR 测度口径中的 θ。"""
    batch = _resolve_plot_batch(batch)
    policy = float(batch.theta_policy) if batch is not None else float(theta_policy)
    return plot_theta_sweep_lines(
        T=T,
        weights=weights,
        nsim=nsim,
        theta_policy_fixed=policy,
        theta_min=theta_min,
        theta_max=theta_max,
        n_theta=n_theta,
        track=track,
        baseline_theta=policy,
        title_prefix=f"Pure measurement sensitivity (policy theta fixed at {policy:.2f})",
        output_prefix="theta_measure_sweep_lines",
        batch=batch,
        show_delta_panel=True,
        annotate_ends=True,
    )


def plot_theta_sweep_component_grid(
    T=DEFAULT_SIM_T_CAP,
    weights=(0.5, 0.3, 0.2),
    nsim=20,
    theta_policy=0.08,
    theta_min=0.08,
    theta_max=0.15,
    n_theta=8,
    batch: SimulationPlotBatch | None = None,
):
    """SR/CBS/FR/CGR 四张 θ sweep 图；分量图往往比重叠的 SR 更容易分辨。"""
    outs = []
    for track in ("sr", "cbs", "fr", "cgr"):
        outs.append(
            plot_theta_measure_sweep_lines(
                T=T,
                weights=weights,
                nsim=nsim,
                theta_policy=theta_policy,
                theta_min=theta_min,
                theta_max=theta_max,
                n_theta=n_theta,
                track=track,
                batch=batch,
            )
        )
    return outs


def plot_theta_policy_scenario_lines(
    T=DEFAULT_SIM_T_CAP,
    weights=(0.5, 0.3, 0.2),
    nsim=20,
    theta_min=0.05,
    theta_max=0.15,
    n_theta=10,
    track="sr",
    baseline_theta=0.08,
    batch: SimulationPlotBatch | None = None,
    matcher=None,
    device=None,
    rollover_enabled=True,
    policy_support_enabled=True,
):
    """θ 同时影响角色分配（政策）与风险测度；每个 θ 重跑仿真。"""
    if batch is not None:
        print("[theta-sweep] policy scenario ignores shared batch (policy θ varies per curve)")
    return plot_theta_sweep_lines(
        T=T,
        weights=weights,
        nsim=nsim,
        theta_policy_fixed=None,
        theta_min=theta_min,
        theta_max=theta_max,
        n_theta=n_theta,
        track=track,
        baseline_theta=baseline_theta,
        title_prefix="Policy-threshold scenario (θ drives car_cutoff + SR measure)",
        output_prefix="theta_policy_scenario_lines",
        batch=None,
        matcher=matcher,
        device=device,
        rollover_enabled=rollover_enabled,
        policy_support_enabled=policy_support_enabled,
    )


def _run_series_mean(weights, theta_measure, theta_policy=0.08, T=50, N=30, B=DEFAULT_MATCH_B, sigma=0.3,
                     nsim=20, seed0=DEFAULT_RANDOM_SEED, track="sr",
                     batch: SimulationPlotBatch | None = None):
    track = str(track).lower()
    assert track in ("sr", "fr", "cbs", "cgr")
    comp = _run_series_mean_components(
        weights=weights,
        theta_measure=theta_measure,
        theta_policy=theta_policy,
        T=T,
        N=N,
        B=B,
        sigma=sigma,
        nsim=nsim,
        seed0=seed0,
        batch=batch,
    )
    return np.asarray(comp[track], dtype=float)


def plot_weight_sweep_lines(
    T=DEFAULT_SIM_T_CAP,
    theta=0.08,
    nsim=50,
    batch: SimulationPlotBatch | None = None,
):
    """
    固定 θ=0.08，扫描 w1（FR 权重），并保持 w2:w3 = 3:2。
    SR = w1·FR + w2·CBS + w3·CGR。
    输出：FIG_DIR / 'weight_sweep_lines.png'
    """
    T = min(int(T), MAX_PLOT_STEPS)
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    batch = _resolve_plot_batch(batch)
    w1_grid = np.asarray(measure_w1_grid(), dtype=float)

    sr_curves = []
    weight_triples = []
    for w1 in w1_grid:
        remaining = 1.0 - float(w1)
        # 始终保持 w2:w3 = 3:2
        w2 = remaining * 3.0 / 5.0
        w3 = remaining * 2.0 / 5.0
        weights_now = (float(w1), float(w2), float(w3))
        weight_triples.append(list(weights_now))
        if batch is not None:
            comp = score_batch_mean_components(
                batch, weights=weights_now, theta_measure=float(theta)
            )
            sr = comp.get("raw_sr", comp["sr"])
        else:
            sr = _run_series_mean(weights_now, theta, T=T, nsim=nsim)
        sr_curves.append(np.asarray(sr, dtype=float))

    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 7))

    cmap = mpl.colormaps['plasma'].reversed()
    norm = mpl.colors.Normalize(vmin=float(w1_grid.min()), vmax=float(w1_grid.max()))

    for w1, sr in zip(w1_grid, sr_curves):
        if len(sr) == 0:
            continue
        xs = np.arange(1, len(sr) + 1)
        ax.plot(xs, sr, lw=1.6, alpha=0.95, color=cmap(norm(w1)))

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label(r"$w_1$ (FR weight)")

    ax.set_title(r"Failure-weight sweep — $SR_t=w_1FR_t+w_2CBS_t+w_3CGR_t$")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Systemic Risk (SR)")
    ax.grid(True, alpha=0.4)

    out = FIG_DIR / "weight_sweep_lines.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)
    print(f"Saved: {out}")
    return out

def measure_single_run_time(T=50, num_banks=30):
    """
    测一整次模拟 + 每步 SR 计算的墙钟时间（wall-clock time）。
    """
    sim = BankNetworkSimulator(num_banks=num_banks, max_steps=T)
    sim._save_network_snapshot = False
    sim.export_policy_logs = False
    sim.initialize_network()

    start = time.perf_counter()
    sr_list = []
    steps_run = 0
    for t in range(T):
        sim.simulate_step(t)
        sr = sim.calculate_systemic_risk()
        sr_list.append(sr)
        steps_run += 1
        if sim.network_stable_step is not None:
            break
    end = time.perf_counter()

    elapsed = end - start
    print(f"[measure] banks={num_banks}, T={T}, steps_run={steps_run}")
    print(f"  Total time: {elapsed:.4f} seconds")
    print(f"  Per step : {elapsed / max(1, steps_run):.6f} seconds/step")

    return sr_list, elapsed

def generate_gnn_panel(
    steps=(10, 20, 30, 40, 50),
    tag="gnnbase",
    matcher=None,
    device=None,
    show=True,
    rollover_enabled: bool = True,
    policy_support_enabled: bool = True,
):
    """
    先跑完整段模拟并缓存指定 step 的状态，再统一出网络图和拼接 panel。
    """
    paths = generate_network_snapshots(
        steps=steps,
        tag=tag,
        show=False,
        matcher=matcher,
        device=device,
        rollover_enabled=rollover_enabled,
        policy_support_enabled=policy_support_enabled,
    )
    if not paths:
        print("[warn] no images collected for panel.")
        return None

    imgs = []
    for path in paths:
        with Image.open(path) as im:
            imgs.append(im.convert("RGB"))

    w = max(im.size[0] for im in imgs)
    h = max(im.size[1] for im in imgs)
    canvas = Image.new("RGB", (w * len(imgs), h), (255, 255, 255))
    for k, im in enumerate(imgs):
        canvas.paste(im, (k * w, 0))

    out = FIG_DIR / f"network_{tag}_panel.png"
    canvas.save(out)
    print(f"Saved panel: {out}")

    if show:
        plt.figure(figsize=(18, 6))
        plt.imshow(canvas)
        plt.axis("off")
        plt.show()

    return out

def generate_matcher_snapshots(
    steps=(10, 20, 30, 40, 50),
    tag="matcher",
    matcher=None,
    device=None,
    show=True,
):
    """只生成 matcher(GNN) 版本的单步网络图（不拼 panel）。"""
    return generate_network_snapshots(
        steps=steps,
        tag=tag,
        show=show,
        matcher=matcher,
        device=device,
    )


def _init_network_snapshot_schedule(self, block_size=50, start_step=1, seed=42):
    rng = random.Random(seed)
    last_step = min(self.max_steps - 1, MAX_NETWORK_SNAPSHOT_STEP - 1)
    self._net_snapshot_steps = set()
    s = start_step
    while s <= last_step:
        block_end = min(s + block_size - 1, last_step)
        if s <= block_end:
            step_pick = rng.randint(s, block_end)
            self._net_snapshot_steps.add(step_pick)
        s += block_size


def maybe_save_network_snapshot(self, step, risk, tag="centralized", edge_quantile=0.0):
    try:
        if not getattr(self, "_save_network_snapshot", False):
            return
        if step >= int(getattr(self, "max_steps", 10**9)):
            return
        if step >= MAX_NETWORK_SNAPSHOT_STEP:
            return
        tag = getattr(self, "_network_snapshot_tag", tag)
        if not hasattr(self, "_net_snapshot_steps") or self._net_snapshot_steps is None:
            self._init_network_snapshot_schedule(block_size=50, start_step=1, seed=42)
        must_plot = (
            getattr(self, "network_stable_step", None) is not None
            and step == self.network_stable_step
        )
        if (step in self._net_snapshot_steps) or must_plot:
            self.visualize_network(
                step=step,
                risk=risk,
                tag=tag,
                save=True,
                show_first=False,
                edge_quantile=edge_quantile,
            )
    except Exception as e:
        print(f"[maybe_save_network_snapshot] Warning: {e}")


# ===== bind external functions as class methods =====
BankNetworkSimulator._sparse_bipartite_update = _sparse_bipartite_update
BankNetworkSimulator.update_network = update_network
BankNetworkSimulator.calculate_systemic_risk = calculate_systemic_risk
BankNetworkSimulator.to_pyg_graph = to_pyg_graph
BankNetworkSimulator.decide_investment = decide_investment
BankNetworkSimulator.invest_free_cash_into_projects = invest_free_cash_into_projects
BankNetworkSimulator.allocate_borrowed_to_projects = allocate_borrowed_to_projects
BankNetworkSimulator.update_project_book = update_project_book
BankNetworkSimulator.visualize_network = visualize_network
BankNetworkSimulator.settle_interbank_and_clear = settle_interbank_and_clear
BankNetworkSimulator._init_network_snapshot_schedule = _init_network_snapshot_schedule
BankNetworkSimulator.maybe_save_network_snapshot = maybe_save_network_snapshot

def run_and_report(T=200, matcher=None, device=None):
    """
    跑一次仿真并在过程中打印阶段预警（如果你已经在 simulate_step 里加了 stage 打印）。
    最后返回 sim，用于在 __main__ 做总结打印。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sim = BankNetworkSimulator(num_banks=30, max_steps=T)
    sim._save_network_snapshot = False
    sim.export_policy_logs = False
    sim.initialize_network()

    for step in range(T):
        # 每步塞 matcher 进去，让 _sparse_bipartite_update 用 GNN matcher 撮合
        if matcher is not None:
            sim.gnn_context = {
                "matcher": matcher,
                "device": device,
                "rmax_spread": DAILY_RFQ_QUOTE_SPREAD,
                "beta_amt": 0.05,
            }
        else:
            sim.gnn_context = None

        sim.simulate_step(step)

        # 网络长时间稳定后提前结束
        if sim.network_stable_step is not None:
            break

    return sim


def summarize_network_run(
    run: RecordedRun,
) -> dict[str, float]:
    snaps = run.snapshots

    def values(key: str) -> np.ndarray:
        return np.asarray(
            [
                float(getattr(s, key, 0.0))
                for s in snaps
            ],
            dtype=float,
        )

    def mean_value(key: str) -> float:
        x = values(key)
        return float(np.mean(x)) if x.size else 0.0

    # Structural network statistics:
    # time average within each replication.
    result = {
        "active_links":
            mean_value("active_links"),

        "network_density":
            mean_value("network_density"),

        "mean_degree":
            mean_value("mean_degree"),

        "degree_std":
            mean_value("degree_std"),

        "mean_weighted_degree":
            mean_value("mean_weighted_degree"),

        "weighted_degree_std":
            mean_value("weighted_degree_std"),

        "exposure_hhi":
            mean_value("exposure_hhi"),

        "largest_component_size":
            mean_value("largest_component_size"),

        "largest_component_share":
            mean_value("largest_component_share"),

        # Literal largest bilateral exposure observed
        # during the complete simulation.
        "largest_bilateral_exposure":
            float(
                np.max(
                    values(
                        "max_counterparty_exposure"
                    )
                )
            )
            if snaps else 0.0,

        "cumulative_transaction_volume":
            float(np.sum(values("total_volume"))),

        "cumulative_clearing_shortfall":
            float(
                np.sum(
                    values("en_unpaid_amount")
                )
            ),

        "cumulative_interbank_writeoff":
            # Must equal sum(interbank_writeoff_amount over steps); do not re-sum bank.default_writeoff.
            float(
                np.sum(
                    values("interbank_writeoff_amount")
                )
            ),

        "cumulative_estate_transfer_discount":
            float(
                np.sum(
                    values("estate_transfer_discount_amount")
                )
            ),
    }

    total_trades = float(
        np.sum(values("num_trades"))
    )
    total_repeated = float(
        np.sum(
            values("repeated_trade_count")
        )
    )

    result["repeated_counterparty_share"] = (
        total_repeated / total_trades
        if total_trades > 0.0
        else 0.0
    )

    # Funding statistics are evaluated only on actual
    # matching observations. For centralized matching,
    # non-auction days are excluded.
    observed = [
        s
        for s in snaps
        if bool(
            getattr(
                s,
                "matching_observation",
                True,
            )
        )
        and float(
            getattr(s, "total_demand", 0.0)
        ) > 1e-12
    ]

    total_demand = sum(
        float(s.total_demand)
        for s in observed
    )

    funded = sum(
        min(
            float(s.total_volume),
            float(s.total_demand),
        )
        for s in observed
    )

    result["funding_satisfaction_ratio"] = (
        funded / total_demand
        if total_demand > 0.0
        else 0.0
    )

    result["cumulative_unmet_demand"] = max(
        0.0,
        total_demand - funded,
    )

    from bank_econ_shared import screening_from_snapshots
    result.update(screening_from_snapshots(snaps))
    return result


def extra_metrics_mean_series(batch: SimulationPlotBatch) -> dict[str, list]:
    """跨仿真平均的扩展市场/网络指标；defaulted_bank_ids / edges 取 run0。"""
    from bank_econ_shared import mean_extra_series_from_runs

    out = mean_extra_series_from_runs(getattr(batch, "runs", None) or [])
    if not batch.runs:
        out["defaulted_bank_ids"] = []
        out["edges_by_step"] = []
        return out
    run0 = batch.runs[0]
    out["defaulted_bank_ids"] = [
        list(getattr(s, "defaulted_bank_ids", None) or []) for s in run0.snapshots
    ]
    out["edges_by_step"] = [
        [list(e) for e in (getattr(s, "edges", None) or [])] for s in run0.snapshots
    ]
    return out


def plot_extra_market_metrics(batch: SimulationPlotBatch, out_path: Path | None = None) -> Path:
    """输出扩展指标轨迹图。"""
    series = extra_metrics_mean_series(batch)
    panels = [
        ("Total Trade Volume", "total_volume"),
        ("Num Trades", "num_trades"),
        ("Unmet Demand Rate", "unmet_demand_rate"),
        ("Network Density", "network_density"),
        ("Exposure HHI", "exposure_hhi"),
        ("Max Counterparty Exposure", "max_counterparty_exposure"),
        ("EN Unpaid Amount", "en_unpaid_amount"),
        ("Interbank Writeoff", "interbank_writeoff_amount"),
        ("Estate Transfer Discount", "estate_transfer_discount_amount"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    axes = axes.ravel()
    shock = np.asarray(series.get("common_liquidity_shock", []), dtype=float)
    for ax, (title, key) in zip(axes, panels):
        y = np.asarray(series.get(key, []), dtype=float)
        xs = np.arange(1, len(y) + 1)
        ax.plot(xs, y, lw=1.8)
        for t, flag in enumerate(shock[: len(y)], start=1):
            if float(flag) >= 0.5:
                ax.axvline(t, color="#c0392b", alpha=0.18, lw=0.8, zorder=0)
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
    # defaulted banks as count
    defs = series.get("defaulted_bank_ids", [])
    ydef = np.asarray([len(x or []) for x in defs], dtype=float)
    axes[7].plot(np.arange(1, len(ydef) + 1), ydef, lw=1.8, color="crimson")
    axes[7].set_title("Defaulted Bank Count (run0)")
    axes[7].grid(True, alpha=0.35)
    axes[8].axis("off")
    # annotate last-step defaulted ids
    last_ids = defs[-1] if defs else []
    axes[8].text(
        0.05, 0.5,
        f"Last-step defaulted bank ids (run0):\n{last_ids}",
        transform=axes[8].transAxes, va="center", fontsize=10,
    )
    fig.suptitle("Extra Market / Network Metrics", fontsize=14)
    fig.tight_layout()
    out = Path(out_path) if out_path is not None else (FIG_DIR / "extra_market_metrics.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def configure_figure_output(fig_dir: Path | None = None) -> Path:
    """可选：将标准结果图写入独立子目录（对比脚本用）。"""
    global FIG_DIR
    if fig_dir is not None:
        FIG_DIR = Path(fig_dir)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR


def export_compare_artifacts(
    plot_batch,
    fig_dir: Path,
    *,
    weights=(0.5, 0.3, 0.2),
    theta: float = 0.08,
    theta_min: float = 0.08,
    theta_max: float = 0.15,
    n_theta: int = 8,
    matcher=None,
    device=None,
    matcher_info: dict | None = None,
) -> Path:
    """导出对比脚本所需的 JSON（无需再 import 本模块）。"""
    fig_dir = Path(fig_dir)
    comp = score_batch_mean_components(
        plot_batch, weights=weights, theta_measure=float(theta)
    )
    lengths = [len(r.snapshots) for r in plot_batch.runs]
    data_H = _coverage_data_horizon(lengths)
    plot_H = _coverage_plot_horizon(lengths)
    baseline = {
        k: [float(x) for x in np.asarray(comp[k], dtype=float)[:data_H]]
        for k in ("sr", "fr", "cbs", "cgr")
    }
    # sr / raw_sr: paper three-weight SR; collapse_index: running max (display only).
    raw_sr = np.asarray(comp["sr"], dtype=float)
    if data_H > 0:
        raw_sr = raw_sr[:data_H]
    collapse = np.maximum.accumulate(raw_sr)
    baseline["sr"] = [float(x) for x in raw_sr]
    baseline["raw_sr"] = [float(x) for x in raw_sr]
    baseline["collapse_index"] = [float(x) for x in collapse]

    baseline_runs = []

    for run in plot_batch.runs:
        run_comp = score_recorded_run(
            run,
            weights=weights,
            theta_measure=float(theta),
            horizon=None,  # keep native early-stop length; never pad to T_cap
        )
        surv = survival_metrics_from_fr(run_comp["fr"], horizon=None)
        surv_T1000 = survival_metrics_from_fr(run_comp["fr"], horizon=1000)

        collapse_step = getattr(run, "collapse_step", None)
        survival_time = int(collapse_step) if collapse_step is not None else None
        observed_until = int(len(run.snapshots))
        raw_sr = np.asarray(run_comp["sr"], dtype=float)
        collapse_sr = np.maximum.accumulate(raw_sr)
        baseline_runs.append({
            "seed": int(run.seed),
            "raw_sr": [float(x) for x in raw_sr],
            "collapse_index": [float(x) for x in collapse_sr],
            "sr": [float(x) for x in raw_sr],
            "fr": [float(x) for x in run_comp["fr"]],
            "cbs": [float(x) for x in run_comp["cbs"]],
            "cgr": [float(x) for x in run_comp["cgr"]],
            "interbank_writeoff": [
                float(getattr(s, "interbank_writeoff_amount", 0.0) or 0.0)
                for s in run.snapshots
            ],
            "mean_alive": float(surv["mean_alive"]),
            "bank_days_alive": float(surv["bank_days_alive"]),
            "bank_days_alive_T1000": float(surv_T1000["bank_days_alive"]),
            "mean_alive_T1000": float(surv_T1000["mean_alive"]),
            "alive_H": float(surv["alive_H"]),
            "t50": surv["t50"],
            "t50_hit": bool(surv["t50_hit"]),
            "t90": surv["t90"],
            "t90_hit": bool(surv["t90_hit"]),
            "t_first": surv["t_first"],
            # None means right-censored at observed_until; never call the cap a failure time.
            "survival_time": survival_time,
            "collapse_hit": collapse_step is not None,
            "collapse_step": survival_time,
            "observed_until": observed_until,
            "stop_alive_threshold": int(getattr(run, "stop_alive_threshold", STOP_ALIVE_THRESHOLD)),
        })

    network_summary_runs = []

    for run in plot_batch.runs:
        net_summary = summarize_network_run(run)

        from bank_econ_shared import pack_network_summary_run
        network_summary_runs.append(
            pack_network_summary_run(
                int(run.seed),
                net_summary,
                getattr(run, "first_failures", None),
            )
        )

    from bank_econ_shared import EXTRA_SERIES_KEYS, extra_series_from_snapshots

    support = policy_support_mean_series(plot_batch)
    extra = extra_metrics_mean_series(plot_batch)
    if data_H > 0:
        support = {
            k: [float(x) for x in np.asarray(v, dtype=float)[:data_H]]
            for k, v in support.items()
        }
        extra = {
            k: (list(v)[:data_H] if isinstance(v, list) else v)
            for k, v in extra.items()
        }

    # Theta sweep like w1: re-score the same recorded batch; only θ_measure changes.
    theta_grid = np.asarray(
        measure_theta_grid(
            theta_min, theta_max, n_theta=n_theta, baseline_theta=float(theta)
        ),
        dtype=float,
    )
    sr_curves = []
    for th in theta_grid:
        c = score_batch_mean_components(
            plot_batch, weights=weights, theta_measure=float(th)
        )
        y = np.asarray(c["sr"], dtype=float)
        sr_curves.append([float(x) for x in (y[:data_H] if data_H > 0 else y)])

    # FR-weight w1 sweep; keep w2:w3 = 3:2.
    w1_grid = np.asarray(measure_w1_grid(), dtype=float)
    weight_sr_curves = []
    weight_triples = []
    for w1 in w1_grid:
        remaining = 1.0 - float(w1)
        w2 = remaining * 3.0 / 5.0
        w3 = remaining * 2.0 / 5.0
        weight_triples.append([float(w1), float(w2), float(w3)])
        wc = score_batch_mean_components(
            plot_batch,
            weights=(float(w1), float(w2), float(w3)),
            theta_measure=float(theta),
        )
        y = np.asarray(wc.get("raw_sr", wc["sr"]), dtype=float)
        weight_sr_curves.append([float(x) for x in (y[:data_H] if data_H > 0 else y)])

    payload = {
        "artifact_schema_version": int(ARTIFACT_SCHEMA_VERSION),
        "code_fingerprint": simulation_code_fingerprint(),
        "baseline_series_policy": "native_length_no_pad_to_T_cap",
        "plot_horizon": int(plot_H),
        "data_horizon": int(data_H),
        "matching": dict(matcher_info or matcher_meta(
            matcher_mode="off",
            matcher=None,
            checkpoint=None,
            mechanism="centralized",
        )),
        "features": {
            "rollover_enabled": bool(getattr(plot_batch, "rollover_enabled", True)),
            "policy_support_enabled": bool(getattr(plot_batch, "policy_support_enabled", True)),
        },
        "batch_meta": _plot_batch_summary(plot_batch),
        "sweep_mode": "measure",
        "weights": list(weights),
        "theta": float(theta),
        "baseline": baseline,
        # NEW: all Monte Carlo replications
        "baseline_runs": baseline_runs,
        "survival": {
            "T_cap": int(plot_batch.T),
            "horizon": None,  # series not padded; use observed_until / collapse_step
            "series_policy": "native_length_no_pad_to_T_cap",
            "mean_alive_mean": float(np.mean([r["mean_alive"] for r in baseline_runs])) if baseline_runs else None,
            "bank_days_alive_mean": float(np.mean([r["bank_days_alive"] for r in baseline_runs])) if baseline_runs else None,
            "t50_hit_rate": float(np.mean([1.0 if r["t50_hit"] else 0.0 for r in baseline_runs])) if baseline_runs else None,
            "t50_mean_conditional": (
                float(np.mean([r["t50"] for r in baseline_runs if r["t50"] is not None]))
                if any(r["t50"] is not None for r in baseline_runs) else None
            ),
            "t90_hit_rate": float(np.mean([1.0 if r.get("t90_hit") else 0.0 for r in baseline_runs])) if baseline_runs else None,
            "t90_mean_conditional": (
                float(np.mean([r["t90"] for r in baseline_runs if r.get("t90") is not None]))
                if any(r.get("t90") is not None for r in baseline_runs) else None
            ),
            "alive_H_mean": float(np.mean([r["alive_H"] for r in baseline_runs])) if baseline_runs else None,
            "survival_time_mean_conditional": (
                float(np.mean([r["survival_time"] for r in baseline_runs if r["survival_time"] is not None]))
                if any(r["survival_time"] is not None for r in baseline_runs) else None
            ),
            "collapse_hit_rate": float(np.mean([1.0 if r.get("collapse_hit") else 0.0 for r in baseline_runs])) if baseline_runs else None,
            "stop_alive_threshold": int(baseline_runs[0]["stop_alive_threshold"]) if baseline_runs else None,
        },
        "network_summary_runs": network_summary_runs,
        "baseline_sr_end": float(comp["sr"][-1]) if len(comp["sr"]) else None,
        "policy_support": {
            key: [float(x) for x in values]
            for key, values in support.items()
        },
        "theta_sweep": {
            "theta_grid": [float(x) for x in theta_grid],
            "sr_curves": sr_curves,
        },
        "weight_sweep": {
            "w1_grid": [float(x) for x in w1_grid],
            "weights": weight_triples,
            "sr_curves": weight_sr_curves,
        },
        "extra_metrics": {
            **{
                k: [float(x) for x in extra.get(k, [])]
                for k in EXTRA_SERIES_KEYS
            },
            "defaulted_bank_ids": extra["defaulted_bank_ids"],
            "edges_by_step": extra["edges_by_step"],
        },
        "baseline_run0": None,
        "extra_metrics_run0": None,
        "shock_path_runs": [],
    }
    if plot_batch.runs:
        run0 = plot_batch.runs[0]
        batch0 = SimulationPlotBatch(
            T=int(plot_batch.T),
            N=int(plot_batch.N),
            B=float(plot_batch.B),
            sigma=float(plot_batch.sigma),
            theta_policy=float(plot_batch.theta_policy),
            runs=[run0],
            rollover_enabled=bool(getattr(plot_batch, "rollover_enabled", True)),
            policy_support_enabled=bool(getattr(plot_batch, "policy_support_enabled", True)),
        )
        r0 = score_batch_mean_components(batch0, weights=weights, theta_measure=float(theta))
        payload["baseline_run0"] = {k: [float(x) for x in r0[k]] for k in ("sr", "fr", "cbs", "cgr")}
        payload["extra_metrics_run0"] = extra_series_from_snapshots(run0.snapshots)
        payload["shock_path_runs"] = [
            {"seed": int(r.seed), **extra_series_from_snapshots(r.snapshots)}
            for r in plot_batch.runs
        ]
    out = fig_dir / "compare_artifacts.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved compare artifacts: {out}")
    return out


def run_standard_figure_pipeline(
    matcher=None,
    device=None,
    *,
    dataset=None,
    matcher_mode: str = "off",
    train_models: bool | None = None,
    T: int = DEFAULT_SIM_T_CAP,
    nsim: int = 20,
    B: float = DEFAULT_MATCH_B,
    fig_dir: Path | None = None,
    network_steps=(10, 20, 30, 40, 50, 100, 150, 200),
    network_tag: str = "centralized",
    show: bool = False,
    rollover_enabled: bool = True,
    policy_support_enabled: bool = True,
    stop_alive_threshold: int | None = STOP_ALIVE_THRESHOLD,
    centralized_cycle_length: int = 1,
    seed0: int = DEFAULT_RANDOM_SEED,
) -> dict:
    """标准结果图。CEN 主实验 matcher_mode=off（集中分配，不用 GNN）。"""
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configure_figure_output(fig_dir)

    if train_models is True and str(matcher_mode).lower() == "off":
        raise SystemExit(
            "CEN 禁止训练/加载 GNN matcher；matcher_mode 必须为 off。"
        )
    elif train_models is False:
        matcher_mode = "off"

    if str(matcher_mode).strip().lower() != "off":
        raise SystemExit(
            f"CEN 禁止 GNN：matcher_mode={matcher_mode!r}；只允许 off（全局利率撮合）。"
        )

    # Centralized allocation does not use the GNN matcher.
    matcher = None
    matcher_info = matcher_meta(
        matcher_mode="off",
        matcher=None,
        checkpoint=None,
        mechanism="centralized",
    )

    network_panel = generate_gnn_panel(
        steps=network_steps,
        tag=network_tag,
        matcher=None,
        device=device,
        show=show,
        rollover_enabled=rollover_enabled,
        policy_support_enabled=policy_support_enabled,
    )

    plot_batch = run_simulation_plot_batch(
        T=T,
        nsim=nsim,
        theta_policy=0.08,
        N=30,
        B=B,
        sigma=0.3,
        seed0=int(seed0),
        matcher=None,
        device=device,
        rollover_enabled=rollover_enabled,
        policy_support_enabled=policy_support_enabled,
        stop_alive_threshold=stop_alive_threshold,
        centralized_cycle_length=centralized_cycle_length,
    )
    set_plot_simulation_batch(plot_batch)

    theta_sweep = None
    baseline = plot_baseline_trajectory(
        T=T, weights=(0.5, 0.3, 0.2), theta=0.08, batch=plot_batch
    )
    extra_fig = plot_extra_market_metrics(plot_batch)
    weight_sweep = None
    compare_artifacts = export_compare_artifacts(
        plot_batch,
        FIG_DIR,
        matcher=None,
        device=device,
        matcher_info=matcher_info,
    )

    return {
        "fig_dir": FIG_DIR,
        "network_panel": network_panel,
        "theta_sweep": theta_sweep,
        "weight_sweep": weight_sweep,
        "baseline_trajectory": baseline,
        "compare_artifacts": compare_artifacts,
        "plot_batch": plot_batch,
        "matcher": None,
        "matcher_info": matcher_info,
        "device": device,
    }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Centralized baseline 标准结果图")
    parser.add_argument("--fig-dir", type=Path, default=None, help="图输出目录")
    parser.add_argument("--show", action="store_true", help="显示 matplotlib 窗口")
    parser.add_argument(
        "--T",
        type=int,
        default=DEFAULT_SIM_T_CAP,
        help="安全上限（默认 4000）；商业银行只剩 stop-alive-threshold 家时早停，存活时间用 collapse_step",
    )
    parser.add_argument("--nsim", type=int, default=20, help="plot batch 轨迹数")
    parser.add_argument("--seed0", type=int, default=DEFAULT_RANDOM_SEED, help="配对仿真的根种子")
    parser.add_argument("--B", type=float, default=DEFAULT_MATCH_B, help="单笔成交上限 B（默认 1200）")
    parser.add_argument(
        "--matcher-mode",
        choices=["off"],
        default="off",
        help="CEN 必须为 off（全局利率撮合，禁止 GNN）",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="(兼容) 等价于 --matcher-mode off",
    )
    parser.add_argument("--no-rollover", action="store_true", help="关闭同业 rollover 分期续借")
    parser.add_argument("--no-policy-support", action="store_true", help="关闭央行 liquidity/capital support 注入")
    parser.add_argument(
        "--stop-alive-threshold",
        type=int,
        default=STOP_ALIVE_THRESHOLD,
        help="剩余商业银行数 <= 该阈值时早停（主实验 4；稳健性 3；央行不计）",
    )
    parser.add_argument(
        "--centralized-cycle-length",
        type=int,
        default=1,
        help="集中撮合周期（默认 1=每日；稳健性可用 15）",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("错误：未安装 PyTorch。请运行：")
        print(f"  {sys.executable} -m pip install torch torch-geometric")
        raise SystemExit(1)

    matcher_mode = "off" if args.no_train else str(args.matcher_mode)

    t_all = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_standard_figure_pipeline(
        device=device,
        matcher_mode=matcher_mode,
        T=args.T,
        nsim=args.nsim,
        B=args.B,
        fig_dir=args.fig_dir,
        network_tag="centralized",
        show=args.show,
        rollover_enabled=not args.no_rollover,
        policy_support_enabled=not args.no_policy_support,
        stop_alive_threshold=args.stop_alive_threshold,
        centralized_cycle_length=int(args.centralized_cycle_length),
        seed0=int(args.seed0),
    )
    print(f"[time] TOTAL: {time.perf_counter() - t_all:.2f}s")
