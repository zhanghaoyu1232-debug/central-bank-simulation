# ===== Runtime setup (paste from line 1) =====
import warnings
import hashlib
warnings.filterwarnings(
    "ignore",
    message=r"networkx backend defined more than once",
    category=RuntimeWarning,
    module=r"networkx\.utils\.backends",
)

from pathlib import Path
from output_paths import (
    DECENTRALIZED_FIG_DIR,
    ARTIFACT_SCHEMA_VERSION,
    simulation_code_fingerprint,
)

OUTPUT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = OUTPUT_ROOT / "输入"
OUTPUT_DIR = OUTPUT_ROOT / "输出"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = DECENTRALIZED_FIG_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)
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
from dataclasses import dataclass, field
from PIL import Image

from interbank_installment_rollover import (
    DEFAULT_DEBT_BURDEN_KAPPA,
    DEFAULT_IBL_CAP_ASSET_LAMBDA,
    ROLLOVER_BORROW_BLOCK_ALL,
    ROLLOVER_BORROW_COUPON_CLEARED,
    ROLLOVER_BORROW_PROJECT_ONLY,
    SCHEDULE_INSTALLMENT,
    ScheduleConfig,
    choose_trade_schedule,
    effective_notional,
    outstanding_principal,
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
DAILY_BORROW_SPREAD = 0.00005
DAILY_INTERBANK_ROLE_LCR_CUTOFF = 0.85
DAILY_INTERBANK_INTENTION_LCR_TARGET = 0.85
DAILY_INTERBANK_CONTRACT_MATURITY = 1  # OFF single_payment: next-period P(1+r); ON uses 5–20 installment
DAILY_OPPORTUNITY_MARGIN = 0.00012
DAILY_SWITCH_HYSTERESIS = 0.00004
DAILY_OPPORTUNITY_BORROW_SCALE = 0.25
DAILY_RFQ_QUOTE_SPREAD = 0.00005
DAILY_RFQ_MARKUP_FLOOR = 0.000005
# 局部 RFQ：提高执行效率，不扩大信息视野。禁止遍历全部商业银行。
RFQ_CANDIDATE_K = 4                      # 每轮向局部池中最多 4 家询价
DAILY_RFQ_K = RFQ_CANDIDATE_K
DAILY_RFQ_MAX_ROUNDS = 4
LOCAL_UNIQUE_LENDER_BUDGET = 8           # 每期新发现对手上限；rollover 原 lender 不计入
MAX_LOCAL_DEGREE = 12
DEN_MAX_DEGREE = MAX_LOCAL_DEGREE
RFQ_RANDOM_EXPLORE_N = 1                 # 局部池最多 1 个随机探索节点
INITIAL_ACQUAINTANCE_DEGREE = 4          # 环形 ±1/±2；只表示认识/可询价，不产生贷款
RFQ_EXHAUST_TOL = 1e-9
RFQ_REJECT_REASONS = (
    "no_local_candidate",
    "pair_cap_exhausted",
    "lender_cash_exhausted",
    "degree_cap",
    "gnn_reject",
    "rate_reject",
    "B_cap_partial",
    "hard_rejected",
    "lender_failed",
)
DAILY_RFQ_BORROWER_RISK_MARKUP = 0.00006   # 高风险 borrower 额外报价加点上限（日频）；基线默认关
DAILY_RFQ_MIN_TRADE_SIZE = 1e-9           # 基线关闭实质最小成交额门槛
# 关系型借贷：历史对手优先询价，降风险加点、提高额度与展期比例（基线默认关）
RELATIONSHIP_MARKUP_DISCOUNT = 0.75       # 关系对手风险加点折扣
RELATIONSHIP_SIZE_BOOST = 0.35           # 关系对手单笔额度上浮
RELATIONSHIP_STRENGTH_SCALE = 1.0        # 每笔成交计入的关系强度
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
DAILY_PROJECT_RETURN_DEFAULT = 0.00008
DAILY_PROJECT_RISK_DEFAULT = 0.00016
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


# ===== Decentralized: Trade / Contract + ContractBook（到期 / 现金流）=====
# 指标口径与 baseline 主循环不变；本模块为后续去中心化主循环预留。

@dataclass
class Trade:
    """单笔同业交易执行记录（撮合成交时产生）。"""
    lender_idx: int
    borrower_idx: int
    amount: float
    rate: float           # 该笔利率（可与 base_rate 或双方报价一致）
    step_executed: int   # 成交所在步数


@dataclass
class Contract:
    """同业合约：single_payment（OFF，当天借、本期结束后本息一次清）；installment（ON，成交即 5–20 期）。"""
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

    def is_due_at(self, step: int) -> bool:
        return step >= self.maturity_step

    def cashflow_at_maturity(self) -> tuple[float, float]:
        """兼容旧接口：返回 (0, 本息合计)。利息只对本金计；arrears 原样计入。"""
        P = outstanding_principal(self)
        arrears = max(0.0, float(getattr(self, "arrears_due", 0.0) or 0.0))
        r = float(self.settlement_rate or self.rate)
        return (0.0, P * (1.0 + r) + arrears)


class ContractBook:
    """
    合约簿：按到期与现金流聚合，供 Decentralized 主循环使用。
    不改变现有 exposure_matrix / 指标口径；可与 baseline 并行维护或后续替代矩阵。
    """
    _next_id: int = 0

    def __init__(self):
        self.contracts: list[Contract] = []

    def _new_id(self) -> str:
        ContractBook._next_id += 1
        return f"c{ContractBook._next_id}"

    def add_from_trade(self, t: Trade, maturity_in_periods: int = 1) -> Contract:
        """由一笔 Trade 生成并登记 Contract（默认一期到期）。"""
        c = Contract(
            contract_id=self._new_id(),
            lender_idx=t.lender_idx,
            borrower_idx=t.borrower_idx,
            principal=t.amount,
            rate=t.rate,
            created_step=t.step_executed,
            maturity_step=t.step_executed + maturity_in_periods,
            settlement_rate=float(t.rate),
        )
        self.contracts.append(c)
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

    def add_contract(self, c: Contract) -> None:
        self.contracts.append(c)

    def remove_contract(self, c: Contract) -> None:
        if c in self.contracts:
            self.contracts.remove(c)

    def contracts_due_at(self, step: int) -> list[Contract]:
        """到期步为 step 的合约（含已过期未结清）。"""
        return [c for c in self.contracts if c.maturity_step <= step]

    def contracts_due_by(self, step: int) -> list[Contract]:
        """到期步 <= step 的合约。"""
        return [c for c in self.contracts if c.maturity_step <= step]

    def cashflow_at_step(self, step: int) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
        """
        在 step 日发生的现金流（仅考虑该步到期的合约）。
        返回 (lender_receives, borrower_pays): (i,j) -> 金额。
        lender_receives[(lender_idx, borrower_idx)] = 债权人 l 从债务人 b 收到的本+息；
        borrower_pays[(lender_idx, borrower_idx)] = 债务人 b 向债权人 l 支付的本+息（数值相等）。
        """
        lender_receives: dict[tuple[int, int], float] = defaultdict(float)
        borrower_pays: dict[tuple[int, int], float] = defaultdict(float)
        for c in self.contracts_due_at(step):
            interest, principal = c.cashflow_at_maturity()
            total = interest + principal
            key = (c.lender_idx, c.borrower_idx)
            lender_receives[key] += total
            borrower_pays[key] += total
        return dict(lender_receives), dict(borrower_pays)

    def active_contracts(self) -> list[Contract]:
        """当前簿内全部未移除合约。"""
        return list(self.contracts)


def configure_simulation_features(
    sim,
    *,
    rollover_enabled: bool = True,
    policy_support_enabled: bool = True,
    relationship_lending_enabled: bool = False,
    rfq_k: int = DAILY_RFQ_K,
    borrower_risk_markup_enabled: bool = False,
    rfq_min_trade_size: float = DAILY_RFQ_MIN_TRADE_SIZE,
):
    """统一设置实验开关，供模型脚本和 compare 脚本复用。

    纯机制基线默认关闭关系借贷 / 风险加点 / 实质 min-trade；
    消融时再分别打开 relationship / risk markup；正式局部 RFQ 默认 K=4、度=12。
    """
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
    sim.relationship_lending_enabled = bool(relationship_lending_enabled)
    sim.rfq_k = int(rfq_k)
    sim.rfq_max_rounds = int(getattr(sim, "rfq_max_rounds", DAILY_RFQ_MAX_ROUNDS))
    sim.local_unique_lender_budget = int(
        getattr(sim, "local_unique_lender_budget", LOCAL_UNIQUE_LENDER_BUDGET)
    )
    sim.max_degree = int(getattr(sim, "max_degree", MAX_LOCAL_DEGREE))
    sim.rfq_borrower_risk_markup = (
        float(DAILY_RFQ_BORROWER_RISK_MARKUP) if borrower_risk_markup_enabled else 0.0
    )
    sim.rfq_min_trade_size = float(rfq_min_trade_size)
    sim.feature_config = {
        "rollover_enabled": bool(rollover_enabled),
        "policy_support_enabled": bool(policy_support_enabled),
        "relationship_lending_enabled": bool(relationship_lending_enabled),
        "rfq_k": int(rfq_k),
        "rfq_max_rounds": int(sim.rfq_max_rounds),
        "local_unique_lender_budget": int(sim.local_unique_lender_budget),
        "max_degree": int(sim.max_degree),
        "borrower_risk_markup_enabled": bool(borrower_risk_markup_enabled),
        "rfq_min_trade_size": float(rfq_min_trade_size),
    }
    return sim


# ----- 聚合函数：ContractBook -> 与 baseline 指标口径一致的敞口/总量 -----

def aggregate_contracts_to_exposure_matrix(book: ContractBook, n: int) -> np.ndarray:
    """
    将 ContractBook 中全部合约聚合成与 baseline 一致的 exposure 矩阵 L。
    L[i,j] > 0 表示 i 对 j 的债权（i 借出给 j），与现有 exposure_matrix 约定一致。
    """
    L = np.zeros((n, n), dtype=float)
    for c in book.contracts:
        P = effective_notional(c)
        L[c.lender_idx, c.borrower_idx] += P
        L[c.borrower_idx, c.lender_idx] -= P
    return L


def aggregate_contracts_to_exposure_matrix_at_step(book: ContractBook, n: int, current_step: int) -> np.ndarray:
    """仅将到期步 > current_step 的合约聚合成 exposure 矩阵（未到期债权）。"""
    L = np.zeros((n, n), dtype=float)
    for c in book.contracts:
        if c.maturity_step > current_step:
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


def total_interbank_assets_liabilities_from_book(book: ContractBook, bank_idx: int, current_step: int) -> tuple[float, float]:
    """从 ContractBook 聚合单家银行的同业资产与同业负债（未到期部分）。"""
    assets = 0.0
    liabilities = 0.0
    for c in book.contracts:
        if c.maturity_step <= current_step:
            continue
        if c.lender_idx == bank_idx:
            assets += effective_notional(c)
        if c.borrower_idx == bank_idx:
            liabilities += effective_notional(c)
    return assets, liabilities


def total_notional_by_bank_from_book(book: ContractBook, n: int, current_step: int) -> tuple[np.ndarray, np.ndarray]:
    """按银行聚合未到期名义本金：(assets_per_bank, liabilities_per_bank)，长度 n。"""
    assets = np.zeros(n, dtype=float)
    liabilities = np.zeros(n, dtype=float)
    for c in book.contracts:
        if c.maturity_step <= current_step:
            continue
        P = effective_notional(c)
        assets[c.lender_idx] += P
        liabilities[c.borrower_idx] += P
    return assets, liabilities


# ===== banks 统一为 list[dict]，全文件使用 dict 访问 bank["key"] / bank.get("key") =====


# =============================================================================
# Decentralized 主流程（1–8）：baseline 主循环不变，本块独立调用
# =============================================================================

# --- 1) Trade 事件 + aggregate 回 X_t ---
def aggregate_trades_to_exposure(trades: list[Trade], n: int) -> np.ndarray:
    """将当步或历史 Trade 列表聚合成敞口矩阵 X_t，与 baseline exposure_matrix 口径一致。"""
    X_t = np.zeros((n, n), dtype=float)
    for t in trades:
        X_t[t.lender_idx, t.borrower_idx] += t.amount
        X_t[t.borrower_idx, t.lender_idx] -= t.amount
    return X_t


# --- 2) banks 聚合 ---
def aggregate_bank_states_to_exposure(banks: list) -> np.ndarray:
    """从 banks 的 interbank 科目反推敞口矩阵（仅当存在双边明细时完整；否则用 contract_book）。"""
    n = len(banks)
    L = np.zeros((n, n), dtype=float)
    for i, b in enumerate(banks):
        ib_a = float(b.get("interbank_assets", 0.0))
        ib_l = float(b.get("interbank_liabilities", 0.0))
        if ib_a > 0 or ib_l > 0:
            pass  # 无法从单行恢复矩阵，需配合 ContractBook
    return L


def update_bank_states_from_contract_book(
    banks: list, book: ContractBook, n: int, current_step: int
) -> None:
    """用 ContractBook 未到期合约更新每家银行的 interbank_assets / interbank_liabilities。"""
    assets, liabilities = total_notional_by_bank_from_book(book, n, current_step)
    for i in range(min(n, len(banks))):
        # 吸收态/冻结墓碑：同业边已关闭，强制 IB 科目为 0，勿覆盖其余冻结科目
        if banks[i].get("absorbing_default") or banks[i].get("balance_sheet_frozen"):
            banks[i]["interbank_assets"] = 0.0
            banks[i]["interbank_liabilities"] = 0.0
            continue
        banks[i]["interbank_assets"] = float(assets[i])
        banks[i]["interbank_liabilities"] = float(liabilities[i])


# --- 3) ContractBook + maturity 到期现金流 ---
def apply_contract_cashflows_at_step(
    banks: list, book: ContractBook, step: int
) -> list[Contract]:
    """【禁止用于主仿真】直接按面值扣款/收款，会制造清算先后顺序。

    正式路径只能走 ``settle_interbank_period``：全部到期债务先入 EN 矩阵，
    再一次清算转账。本函数仅保留作离线测试，不参与 simulate_step。
    """
    lender_receives, borrower_pays = book.cashflow_at_step(step)
    due = book.contracts_due_at(step)
    for c in due:
        total = c.principal * (1.0 + c.rate)
        # 债务人支付
        if c.borrower_idx < len(banks):
            banks[c.borrower_idx]["liquid_assets"] = max(
                0.0,
                float(banks[c.borrower_idx].get("liquid_assets", 0.0)) - total,
            )
        # 债权人收取
        if c.lender_idx < len(banks):
            banks[c.lender_idx]["liquid_assets"] = float(
                banks[c.lender_idx].get("liquid_assets", 0.0)
            ) + total
        book.remove_contract(c)
    return due


# --- 4) Step2: intentions 共用 interbank_intentions.collect_intentions ---

def _plan_to_trades_midpoint(plan, r_min: dict, r_max: dict, step: int):
    """plan -> Trade；利率取双方保留价中点。"""
    trades: list[Trade] = []
    for i, j, amt in plan:
        i = int(i); j = int(j)
        rate = 0.5 * (float(r_min[i]) + float(r_max[j]))
        trades.append(
            Trade(
                lender_idx=i,
                borrower_idx=j,
                amount=float(amt),
                rate=float(rate),
                step_executed=int(step),
            )
        )
    return trades


def _daily_project_shock_params(market_environment: str) -> tuple[float, float]:
    if market_environment == "bull":
        return DAILY_PROJECT_SHOCK_MEAN_BULL, DAILY_PROJECT_SHOCK_STD_BULL
    return DAILY_PROJECT_SHOCK_MEAN_BEAR, DAILY_PROJECT_SHOCK_STD_BEAR


def _borrower_credit_risk(bank, car_threshold: float = 0.08) -> float:
    """借款人信用风险代理 [0,1]，越大表示 CAR/LCR/偿付能力越弱。"""
    car = float(regulatory_car(bank))
    lcr = float(bank.get("liquidity_coverage_ratio", 1.0))
    solv = float(bank.get("solvency_ratio", 1.0))
    car_risk = 1.0 - min(max(car / max(car_threshold, 1e-9), 0.0), 1.0)
    lcr_risk = 1.0 - min(max(lcr, 0.0), 1.0)
    solv_risk = 1.0 - min(max(solv, 0.0), 1.0)
    return float(min(max(0.5 * car_risk + 0.3 * lcr_risk + 0.2 * solv_risk, 0.0), 1.0))


def _borrower_debt_burden(bank, *, kappa: float = DEFAULT_DEBT_BURDEN_KAPPA) -> float:
    """同业负债相对权益负担 [0,1]。"""
    from interbank_installment_rollover import bank_equity_proxy

    ibl = max(0.0, float(bank.get("interbank_liabilities", 0.0)))
    eq = max(float(bank_equity_proxy(bank)), 1e-9)
    k = max(1e-9, float(kappa))
    return float(min(max(ibl / (k * eq), 0.0), 1.0))


def public_car_risk_grade(bank, car_threshold: float = 0.08) -> float:
    """Coarse public CAR stress grade in [0,1] (higher = weaker)."""
    car = float(regulatory_car(bank))
    thr = max(float(car_threshold), 1e-9)
    if car >= 1.5 * thr:
        return 0.0
    if car >= thr:
        return 0.33
    if car >= 0.5 * thr:
        return 0.66
    return 1.0


def public_lcr_risk_grade(bank) -> float:
    """Coarse public LCR stress grade in [0,1] (higher = weaker)."""
    lcr = float(bank.get("liquidity_coverage_ratio", 1.0) or 0.0)
    if lcr >= 1.2:
        return 0.0
    if lcr >= 1.0:
        return 0.33
    if lcr >= 0.8:
        return 0.66
    return 1.0


def public_repayment_stats(system, bank_idx: int) -> tuple[float, float]:
    """
    Returns (underpay_intensity, late_ratio) in [0,1] from observable contract stats.
    underpay_intensity: normalized consecutive-miss mass; late_ratio: share of live notes overdue.
    """
    book = getattr(system, "contract_book", None) if system is not None else None
    contracts = list(getattr(book, "contracts", []) or [])
    bj = int(bank_idx)
    n_notes = 0
    n_late = 0
    miss_sum = 0.0
    for c in contracts:
        if int(getattr(c, "borrower_idx", -1)) != bj:
            continue
        n_notes += 1
        misses = int(getattr(c, "consecutive_misses", 0) or 0)
        arrears = float(getattr(c, "arrears_due", 0.0) or 0.0)
        miss_sum += float(misses)
        if misses > 0 or arrears > 1e-9:
            n_late += 1
    if n_notes <= 0:
        # Fall back to bank-level counters if maintained.
        under = float((getattr(system, "banks", [{}])[bj] if system is not None and 0 <= bj < len(getattr(system, "banks", [])) else {}).get("public_underpay_events", 0.0) or 0.0)
        ontime = float((getattr(system, "banks", [{}])[bj] if system is not None and 0 <= bj < len(getattr(system, "banks", [])) else {}).get("public_ontime_events", 0.0) or 0.0)
        tot = under + ontime
        if tot <= 1e-12:
            return 0.0, 0.0
        return float(min(under / max(tot, 1.0), 1.0)), float(min(under / max(tot, 1.0), 1.0))
    underpay = float(min(miss_sum / max(3.0 * n_notes, 1.0), 1.0))
    late_ratio = float(n_late / max(n_notes, 1))
    return underpay, late_ratio


def bilateral_gross_exposure(
    system,
    lender_idx: int,
    borrower_idx: int,
    *,
    extra: float = 0.0,
) -> float:
    """
    Gross lender→borrower claim = Σ (outstanding principal + arrears_due).
    Never use the net exposure matrix (reverse claims must not hide concentration).
    """
    total = 0.0
    book = getattr(system, "contract_book", None) if system is not None else None
    li, bj = int(lender_idx), int(borrower_idx)
    for c in list(getattr(book, "contracts", []) or []):
        if int(getattr(c, "lender_idx", -1)) != li:
            continue
        if int(getattr(c, "borrower_idx", -1)) != bj:
            continue
        total += float(effective_notional(c))
    return float(total + max(0.0, float(extra)))


# Backward name used by older call sites.
def bilateral_lender_exposure(system, lender_idx: int, borrower_idx: int) -> float:
    return bilateral_gross_exposure(system, lender_idx, borrower_idx)


def _gross_lender_claim_map(system, lender_idx: int) -> dict[int, float]:
    out: dict[int, float] = {}
    book = getattr(system, "contract_book", None) if system is not None else None
    li = int(lender_idx)
    for c in list(getattr(book, "contracts", []) or []):
        if int(getattr(c, "lender_idx", -1)) != li:
            continue
        bj = int(getattr(c, "borrower_idx", -1))
        if bj < 0 or bj == li:
            continue
        out[bj] = float(out.get(bj, 0.0) + float(effective_notional(c)))
    return out


def lender_delta_hhi(
    system,
    lender_idx: int,
    borrower_idx: int,
    *,
    proposed_amt: float = 0.0,
) -> float:
    """
    Concentration penalty = max(0, HHI_post - HHI_pre) on gross claims.
    Cold start (no prior IB assets): return 0 — avoid HHI=1 on the first loan.
    """
    claims = _gross_lender_claim_map(system, int(lender_idx))
    pre_tot = float(sum(claims.values()))
    if pre_tot <= 1e-12:
        return 0.0
    pre_hhi = float(sum((v / pre_tot) ** 2 for v in claims.values()))
    post = dict(claims)
    j = int(borrower_idx)
    post[j] = float(post.get(j, 0.0) + max(0.0, float(proposed_amt)))
    post_tot = float(sum(post.values()))
    if post_tot <= 1e-12:
        return 0.0
    post_hhi = float(sum((v / post_tot) ** 2 for v in post.values()))
    return float(np.clip(post_hhi - pre_hhi, 0.0, 1.0))


def lender_portfolio_hhi(system, lender_idx: int) -> float:
    """Pre-trade gross HHI (diagnostics). Prefer lender_delta_hhi for scoring."""
    claims = _gross_lender_claim_map(system, int(lender_idx))
    tot = float(sum(claims.values()))
    if tot <= 1e-12:
        return 0.0
    return float(np.clip(sum((v / tot) ** 2 for v in claims.values()), 0.0, 1.0))


def pair_bilateral_exposure_ratio(
    system,
    banks: list,
    lender_idx: int,
    borrower_idx: int,
    *,
    proposed_amt: float = 0.0,
    extra_existing: float = 0.0,
) -> float:
    """Pair feature in [0,1]: (gross existing + proposed) / lender core capital."""
    i = int(lender_idx)
    bi = banks[i] if 0 <= i < len(banks) else {}
    core = max(float(bi.get("core_capital", 0.0)), 1e-9)
    existing = bilateral_gross_exposure(
        system, i, int(borrower_idx), extra=float(extra_existing)
    )
    return float(np.clip((existing + max(0.0, float(proposed_amt))) / core, 0.0, 1.0))


def den_pair_trade_cap(
    banks: list,
    lender_idx: int,
    borrower_idx: int,
    *,
    initial_demand: float,
    B_max: float,
    system=None,
    core_frac: float = 0.25,
    demand_frac: float = 0.50,
    extra_existing: float = 0.0,
) -> float:
    """
    DEN bilateral new-trade cap using gross exposure:
      pair_room = max(0, 0.25*core - existing_gross)
      cap = min(pair_room, 0.50*initial_demand, B_max)

    ``initial_demand`` must be the pre-match demand for this borrower (not residual).
    """
    i = int(lender_idx)
    bi = banks[i] if 0 <= i < len(banks) else {}
    core = max(0.0, float(bi.get("core_capital", 0.0)))
    existing = bilateral_gross_exposure(
        system, i, int(borrower_idx), extra=float(extra_existing)
    )
    pair_room = max(0.0, float(core_frac) * core - existing)
    return float(
        max(
            0.0,
            min(
                pair_room,
                float(demand_frac) * max(0.0, float(initial_demand)),
                float(B_max),
            ),
        )
    )


def build_borrower_local_graph(
    system,
    borrower_idx: int,
    candidates: list[int] | set[int],
    hist_cps: set[int] | list[int] | None = None,
    *,
    use_prev: bool = True,
    mask_private: bool = True,
):
    """
    Shared train/inference local graph for one borrower RFQ.
    Non-observer nodes: private levels masked; lagged public risk grades overlaid.
    Bilateral exposure is NOT a node feature — it goes to the pair decoder.
    """
    j = int(borrower_idx)
    hist = {int(x) for x in (hist_cps or [])}
    cands = {int(x) for x in candidates}
    local_nodes = {j} | hist | cands
    graph, id_map = system.to_local_pyg_graph(
        local_nodes,
        use_prev=use_prev,
        observer_idx=j,
        mask_private=mask_private,
    )
    graph.global_node_ids = [g for g, _ in sorted(id_map.items(), key=lambda kv: kv[1])]
    return graph, id_map


def teacher_rfq_pair_score(
    system,
    banks: list,
    lender_idx: int,
    borrower_idx: int,
    *,
    quote_rate: float = 0.0,
    proposed_amt: float = 0.0,
    car_threshold: float = 0.08,
    extra_existing: float = 0.0,
) -> float:
    """
    Systemic-safe teacher (v6); every term clipped to [0,1]:

      Score = 1.0 Liq + 0.8 Sound + 0.4 Rel
            - 2.5 q_j - 1.5 Exposure_ij - 1.0 ΔHHI_i

    ``q_j`` is the same origination risk frozen onto projects, so teacher
    ranking and funded_q speak the same language.
    """
    from bank_econ_shared import borrower_origination_risk_q

    _ = quote_rate
    i, j = int(lender_idx), int(borrower_idx)
    bi = banks[i] if 0 <= i < len(banks) else {}
    bj = banks[j] if 0 <= j < len(banks) else {}
    rel = float(relationship_strength(system, i, j)) if system is not None else 0.0
    rel_n = float(np.clip(rel / (rel + 2.0), 0.0, 1.0))
    liq = float(bi.get("liquid_assets", 0.0))
    lia = max(1.0, float(bi.get("current_liabilities", 1.0)))
    liquidity_i = float(np.clip(np.tanh(liq / max(200.0, 0.5 * lia)), 0.0, 1.0))
    car_l = float(
        bi.get("lag_car", bi.get("capital_adequacy_ratio", 0.0)) or 0.0
    )
    soundness_i = 1.0 if bi.get("is_active", True) else 0.0
    soundness_i *= float(
        np.clip((car_l - 0.5 * car_threshold) / max(car_threshold, 1e-6), 0.0, 1.0)
    )
    q_j = float(
        borrower_origination_risk_q(
            bj,
            system=system,
            bank_idx=j,
            car_threshold=float(car_threshold),
        )
    )
    exposure_ij = pair_bilateral_exposure_ratio(
        system, banks, i, j,
        proposed_amt=float(proposed_amt),
        extra_existing=float(extra_existing),
    )
    conc_i = lender_delta_hhi(
        system, i, j, proposed_amt=float(proposed_amt) + float(extra_existing)
    )
    return float(
        1.0 * liquidity_i
        + 0.8 * soundness_i
        + 0.4 * rel_n
        - 2.5 * q_j
        - 1.5 * exposure_ij
        - 1.0 * conc_i
    )


def _teacher_score_to_unit(score: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(score, -30.0, 30.0)))))


class CompetitionTeacherMatcher:
    """Adapter that runs the v6 teacher through the formal simultaneous RFQ path."""

    def __init__(self, system):
        self.system = system

    def score_pairs(self, graph, pairs, device="cpu", pair_features=None):
        _ = device, pair_features
        ids = [int(x) for x in getattr(graph, "global_node_ids", [])]
        if not ids:
            raise RuntimeError("Teacher matcher requires graph.global_node_ids")
        out = []
        for li, bj in pairs:
            i, j = ids[int(li)], ids[int(bj)]
            raw = teacher_rfq_pair_score(
                self.system, self.system.banks, i, j,
                car_threshold=float(getattr(self.system, "car_cutoff", 0.08)),
            )
            out.append(_teacher_score_to_unit(raw))
        return np.asarray(out, dtype=float)


def _blend_gnn_teacher_score(gnn_score: float, teacher_unit: float) -> float:
    """Inference screen: teacher (q-aligned) outweighs frozen GNN pair score."""
    g = float(np.clip(gnn_score, 0.0, 1.0))
    t = float(np.clip(teacher_unit, 0.0, 1.0))
    return float(0.40 * g + 0.60 * t)


def lender_risk_adjusted_accept_score(
    *,
    gnn_score: float,
    deal_rate: float,
    lender_min_rate: float,
    borrower_risk: float,
    post_trade_exposure_ratio: float,
    rate_norm_scale: float = 0.0005,
    teacher_unit: float = 0.5,
) -> float:
    """
    Cross-borrower ranking on origination q.

    GNN / teacher / rate are tie-breaks: a 0.3 q gap cannot be overturned by
    a maxed GNN score. Rate margin remains a small bonus.
    """
    margin = max(0.0, float(deal_rate) - float(lender_min_rate))
    norm_margin = float(np.clip(margin / max(float(rate_norm_scale), 1e-9), 0.0, 1.0))
    g = float(np.clip(gnn_score, 0.0, 1.0))
    t = float(np.clip(teacher_unit, 0.0, 1.0))
    br = float(np.clip(borrower_risk, 0.0, 1.0))
    er = float(np.clip(post_trade_exposure_ratio, 0.0, 1.0))
    return float(0.15 * g + 0.10 * t + 0.10 * norm_margin - 0.70 * br - 0.15 * er)


def append_rfq_training_event(
    system,
    *,
    step: int,
    borrower_idx: int,
    candidates: list[int],
    accepted: list[int],
    hist_cps: set[int] | list[int],
    pair_feats: dict[int, list[float]] | None = None,
    teacher_scores: dict[int, float] | None = None,
) -> None:
    if system is None or not bool(getattr(system, "collect_rfq_events", False)):
        return
    buf = getattr(system, "step_rfq_events", None)
    if not isinstance(buf, list):
        system.step_rfq_events = []
        buf = system.step_rfq_events
    # Public repayment overlays at match time (same source as inference).
    repay_ids = {int(borrower_idx)} | {int(x) for x in candidates} | {
        int(x) for x in (hist_cps or [])
    }
    repay_feats: dict[str, list[float]] = {}
    for bid in repay_ids:
        under, late = public_repayment_stats(system, int(bid))
        repay_feats[str(int(bid))] = [float(under), float(late)]
    graph, id_map = build_borrower_local_graph(
        system, int(borrower_idx), candidates, hist_cps,
        use_prev=True, mask_private=True,
    )
    teacher_raw = {int(k): float(v) for k, v in dict(teacher_scores or {}).items()}
    teacher_unit = {int(k): _teacher_score_to_unit(v) for k, v in teacher_raw.items()}
    buf.append(
        {
            "step": int(step),
            "borrower": int(borrower_idx),
            "candidates": [int(x) for x in candidates],
            "accepted": [int(x) for x in accepted],
            "hist": sorted(int(x) for x in hist_cps),
            "pair_feats": {
                str(int(k)): [float(v) for v in vals]
                for k, vals in dict(pair_feats or {}).items()
            },
            "repay_feats": repay_feats,
            "candidate_labels": {
                str(int(i)): int(int(i) in {int(x) for x in accepted})
                for i in candidates
            },
            "teacher_scores": {str(k): v for k, v in teacher_raw.items()},
            "teacher_score_unit": {str(k): v for k, v in teacher_unit.items()},
            "all_negative": bool(not accepted),
            "local_graph": {
                "node_ids": [int(x) for x in getattr(graph, "global_node_ids", [])],
                "node_features": graph.x.detach().cpu().numpy().astype(float).tolist(),
                "edge_index": graph.edge_index.detach().cpu().numpy().astype(int).tolist(),
                "edge_attr": graph.edge_attr.detach().cpu().numpy().astype(float).tolist(),
                "borrower_local_id": int(id_map[int(borrower_idx)]),
                "candidate_local_ids": {
                    str(int(i)): int(id_map[int(i)])
                    for i in candidates if int(i) in id_map
                },
            },
        }
    )


def log_rfq_match_diagnostics(
    system,
    step: int,
    intentions: list[Intention],
    trades: list[Trade],
    *,
    B_max: float = 1200.0,
) -> None:
    """RFQ 撮合摘要：与 _sparse_bipartite_update 同款 [diag]/[debug] 输出。"""
    if not getattr(system, "verbose_matching", True):
        return
    lenders_int = [x for x in intentions if x.role == "lender"]
    borrowers_int = [x for x in intentions if x.role == "borrower"]
    banks = getattr(system, "banks", [])
    n = int(getattr(system, "num_banks", len(banks)))
    eps = 1e-9

    print(
        f"[diag] step={int(step)} RFQ lenders={len(lenders_int)} borrowers={len(borrowers_int)}"
    )
    blocked = getattr(system, "rollover_blocked_borrowers", None) or set()
    if blocked:
        print(
            f"[diag] rollover_blocked_borrowers={len(blocked)} "
            f"indices={sorted(blocked)}"
        )
    rejects = getattr(system, "rfq_reject_counts", None)
    if isinstance(rejects, dict) and any(int(v) > 0 for v in rejects.values()):
        parts = [f"{k}={int(v)}" for k, v in rejects.items() if int(v) > 0]
        print("[diag] rfq_reject " + " ".join(parts))
    local_cap = _rfq_local_capacity_summary(system)
    if local_cap:
        print(
            "[diag] rfq_local_capacity "
            f"borrowers={local_cap['borrowers']} "
            f"eligible min/mean/max="
            f"{local_cap['eligible_local_min']}/"
            f"{local_cap['eligible_local_mean']:.2f}/"
            f"{local_cap['eligible_local_max']} "
            f"ratio min/mean/max="
            f"{local_cap['capacity_ratio_min']:.3f}/"
            f"{local_cap['capacity_ratio_mean']:.3f}/"
            f"{local_cap['capacity_ratio_max']:.3f} "
            f"feasible_share={local_cap['capacity_feasible_share']:.3f} "
            f"capped_known={local_cap['capped_known_total']}"
        )

    if n > 0 and banks:
        LCR_TARGET = 1.0
        ALPHA_STRESS_BORROWER = 1.0
        liq_arr = np.array([float(banks[k]["liquid_assets"]) for k in range(n)], dtype=float)
        lia_arr = np.array([float(banks[k]["current_liabilities"]) for k in range(n)], dtype=float)
        out_arr = np.array([float(banks[k].get("outflow_rate", 0.4)) for k in range(n)], dtype=float)
        res_arr = np.array([float(system.reserve_buffer[k]) for k in range(n)], dtype=float)
        req_arr = res_arr * lia_arr
        target_arr = np.maximum(req_arr, LCR_TARGET * ALPHA_STRESS_BORROWER * (lia_arr * out_arr))
        gap_arr = np.maximum(0.0, target_arr - liq_arr)
        active_mask = np.array([bool(banks[k].get("is_active", True)) for k in range(n)])
        gap_active = gap_arr[active_mask]
        if gap_active.size:
            print(
                f"[diag-gap] active_gap>0={int((gap_active > 1e-6).sum())}/"
                f"{int(active_mask.sum())} | "
                f"gap min/mean/max={gap_active.min():.2f}/{gap_active.mean():.2f}/"
                f"{gap_active.max():.2f}"
            )

    borrowers = [x.bank_idx for x in borrowers_int]
    if borrowers:
        LCR_TARGET = 1.0
        ALPHA_STRESS_BORROWER = 1.0
        K_EXPAND_BEAR = 0.06
        K_EXPAND_BULL = 0.12
        need_gap_list = []
        extra_need_list = []
        base_rate = float(getattr(system, "base_rate", 0.0))
        long_term_rate = float(getattr(system, "long_term_rate", base_rate))
        for j in borrowers:
            liq = float(banks[j]["liquid_assets"])
            lia = float(banks[j]["current_liabilities"])
            req = float(system.reserve_buffer[j] * lia)
            outflow_target = lia * float(banks[j].get("outflow_rate", 0.4))
            target_liq = max(req, LCR_TARGET * ALPHA_STRESS_BORROWER * outflow_target)
            need_gap_list.append(max(0.0, target_liq - liq))
            phi = float(banks[j].get("risk_appetite", 0.5))
            exp_proj = float(banks[j].get("investment_interest_rate", long_term_rate))
            loan_rt = float(banks[j].get("loan_interest_rate", base_rate))
            spread_pos = max(0.0, exp_proj - loan_rt)
            K = K_EXPAND_BEAR if getattr(system, "market_environment", "bull") == "bear" else K_EXPAND_BULL
            extra_need_list.append(K * lia * phi * (spread_pos / (loan_rt + 1e-9)))
        print(
            f"[diag-need] borrowers={len(borrowers)} | "
            f"gap(min/mean/max)={np.min(need_gap_list):.2f}/{np.mean(need_gap_list):.2f}/"
            f"{np.max(need_gap_list):.2f} | "
            f"extra(min/mean/max)={np.min(extra_need_list):.2f}/{np.mean(extra_need_list):.2f}/"
            f"{np.max(extra_need_list):.2f}"
        )

    supply = [float(x.quantity) for x in lenders_int]
    demand = [float(x.quantity) for x in borrowers_int]
    if supply:
        print(
            f"[diag] supply min/mean/max = {np.min(supply):.2f}/{np.mean(supply):.2f}/"
            f"{np.max(supply):.2f}"
        )
    if demand:
        print(
            f"[diag] demand  min/mean/max = {np.min(demand):.2f}/{np.mean(demand):.2f}/"
            f"{np.max(demand):.2f}"
        )

    if len(borrowers_int) == 0:
        print("[debug] No effective demand (RFQ). Market idle this step.")
        return

    total_supply = float(np.sum(np.asarray(supply, dtype=float))) if supply else 0.0
    total_demand = float(np.sum(np.asarray(demand, dtype=float))) if demand else 0.0
    if total_supply <= eps or total_demand <= eps:
        print(
            f"[debug] No matching (RFQ): total_supply={total_supply:.2f}, "
            f"total_demand={total_demand:.2f}"
        )
        return

    actual_lent = float(sum(t.amount for t in trades))
    print(
        f"[debug] Matching finished: edges={len(trades)}, "
        f"lenders={len(lenders_int)}, borrowers={len(borrowers_int)}, "
        f"total_supply={total_supply:.2f}, total_demand={total_demand:.2f}, "
        f"actual_lent={actual_lent:.2f}, B_eff={float(B_max):.2f}"
    )


# --- 5) Step3: RFQMarket 多轮报价成交 ---

def build_ring_acquaintance_sets(
    n_banks: int,
    rng,
    *,
    degree: int = INITIAL_ACQUAINTANCE_DEGREE,
    skip_idx: int = 0,
) -> list[set[int]]:
    """Random-permutation ring: each commercial bank links to ±k neighbors.

    Edges mean “can RFQ”, not loans. No ranking and no full-market scan.
    Central bank ``skip_idx`` is excluded. Degree is ``2k`` (default 4).
    """
    n = max(0, int(n_banks))
    neigh = [set() for _ in range(n)]
    skip = int(skip_idx)
    ids = [i for i in range(n) if i != skip]
    m = len(ids)
    k = max(0, int(degree) // 2)
    if m < 2 or k <= 0:
        return neigh
    if rng is not None:
        order = [int(ids[int(i)]) for i in rng.permutation(m)]
    else:
        order = list(ids)
        random.shuffle(order)
    max_k = min(k, (m - 1) // 2)
    if max_k <= 0:
        a, b = int(order[0]), int(order[1])
        neigh[a].add(b)
        neigh[b].add(a)
        return neigh
    for pos, i in enumerate(order):
        ii = int(i)
        for d in range(1, max_k + 1):
            j = int(order[(pos + d) % m])
            jj = int(order[(pos - d) % m])
            if j != ii:
                neigh[ii].add(j)
                neigh[j].add(ii)
            if jj != ii:
                neigh[ii].add(jj)
                neigh[jj].add(ii)
    return neigh


def _acquaintance_neighbor_sets(system, n: int) -> list[set[int]]:
    n = max(0, int(n))
    out = [set() for _ in range(n)]
    acq = getattr(system, "acquaintance_sets", None) if system is not None else None
    if not acq:
        return out
    for i, s in enumerate(acq):
        if i >= n:
            break
        out[i] = {int(x) for x in s if 0 <= int(x) < n and int(x) != i}
    return out


def _info_neighbor_sets(
    system,
    exposure_neigh: list[set[int]] | None,
    n: int,
) -> list[set[int]]:
    """Undirected local-info graph: initial acquaintances ∪ current exposure hops."""
    n = max(0, int(n))
    out = [set() for _ in range(n)]
    if exposure_neigh:
        for i, s in enumerate(exposure_neigh):
            if i >= n:
                break
            out[i].update(int(x) for x in s if 0 <= int(x) < n and int(x) != i)
    acq = _acquaintance_neighbor_sets(system, n)
    for i in range(n):
        out[i].update(acq[i])
    return out


def _exposure_neighbor_sets(exposure_matrix, n: int, eps: float = 1e-9) -> list[set[int]]:
    """Undirected counterparty sets from signed/unsigned exposure matrix."""
    neigh = [set() for _ in range(int(n))]
    if exposure_matrix is None:
        return neigh
    L = np.asarray(exposure_matrix, dtype=float)
    if L.ndim != 2:
        return neigh
    n = min(int(n), int(L.shape[0]), int(L.shape[1]))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(float(L[i, j])) > eps:
                neigh[i].add(j)
    return neigh


def _degree_allows_pair(neigh: list[set[int]], i: int, j: int, max_degree: int) -> bool:
    """Existing links always allowed; new links require both sides under max_degree."""
    i, j = int(i), int(j)
    if i == j:
        return False
    if j in neigh[i]:
        return True
    md = int(max_degree)
    return len(neigh[i]) < md and len(neigh[j]) < md


def _register_pair_degree(neigh: list[set[int]], i: int, j: int) -> None:
    i, j = int(i), int(j)
    neigh[i].add(j)
    neigh[j].add(i)


def relationship_strength(system, lender_idx: int, borrower_idx: int) -> float:
    """Directed lending relationship strength (lender → borrower), plus weak reverse link."""
    if system is None:
        return 0.0
    counts = getattr(system, "relationship_counts", None) or {}
    li, bj = int(lender_idx), int(borrower_idx)
    direct = float(counts.get((li, bj), 0.0))
    reverse = float(counts.get((bj, li), 0.0))
    return direct + 0.35 * reverse


def record_relationship_trade(system, lender_idx: int, borrower_idx: int, amount: float = 0.0) -> None:
    if system is None:
        return
    if not bool(getattr(system, "relationship_lending_enabled", True)):
        return
    counts = getattr(system, "relationship_counts", None)
    if counts is None:
        system.relationship_counts = {}
        counts = system.relationship_counts
    key = (int(lender_idx), int(borrower_idx))
    bump = float(getattr(system, "relationship_strength_scale", RELATIONSHIP_STRENGTH_SCALE))
    # Count trades; large notional adds a little extra stickiness.
    bump += 0.05 * min(5.0, float(amount) / 200.0) if amount > 0 else 0.0
    counts[key] = float(counts.get(key, 0.0)) + bump


def _historical_counterparties(
    system,
    bank_idx: int,
    *,
    use_prev: bool = True,
    step: int | None = None,
) -> set[int]:
    """Direct historical counterparties within an optional time window."""
    out: set[int] = set()
    if system is None:
        return out
    window = int(getattr(system, "relationship_history_window", 10) or 10)
    touch = getattr(system, "counterparty_touch_step", None)
    cur = int(step) if step is not None else int(getattr(system, "current_step", 0) or 0)
    if isinstance(touch, dict):
        row = touch.get(int(bank_idx)) or {}
        for other, last in row.items():
            if window <= 0 or (cur - int(last)) <= window:
                out.add(int(other))
        if out:
            return out
    # Fallback: current/prev exposure edges (still window-unaware).
    L_src = None
    if use_prev:
        L_src = getattr(system, "prev_exposure_matrix", None)
    if L_src is None:
        L_src = getattr(system, "exposure_matrix", None)
    if L_src is None:
        return out
    L = np.asarray(L_src, dtype=float)
    i = int(bank_idx)
    n = min(L.shape[0], L.shape[1])
    if i < 0 or i >= n:
        return out
    for j in range(n):
        if i == j:
            continue
        if abs(float(L[i, j])) > 1e-9 or abs(float(L[j, i])) > 1e-9:
            out.add(int(j))
    return out


def touch_counterparty(system, a: int, b: int, step: int) -> None:
    if system is None:
        return
    hist = getattr(system, "counterparty_touch_step", None)
    if not isinstance(hist, dict):
        system.counterparty_touch_step = {}
        hist = system.counterparty_touch_step
    i, j, t = int(a), int(b), int(step)
    hist.setdefault(i, {})[j] = t
    hist.setdefault(j, {})[i] = t


def _shuffle_ids(values, rng_m) -> list[int]:
    values = [int(x) for x in values]
    if len(values) <= 1:
        return values
    if rng_m is not None:
        order = rng_m.permutation(len(values))
        return [values[int(x)] for x in order]
    random.shuffle(values)
    return values


def _note_rfq_reject(system, reason: str, n: int = 1) -> None:
    if system is None:
        return
    stats = getattr(system, "rfq_reject_counts", None)
    if not isinstance(stats, dict):
        system.rfq_reject_counts = {k: 0 for k in RFQ_REJECT_REASONS}
        stats = system.rfq_reject_counts
    key = str(reason)
    stats[key] = int(stats.get(key, 0)) + int(n)


def _reset_rfq_reject_counts(system) -> None:
    if system is None:
        return
    system.rfq_reject_counts = {k: 0 for k in RFQ_REJECT_REASONS}
    system.rfq_unique_lenders_queried = {}
    system.rfq_local_rollover = {}
    system.rfq_local_capacity = {}


def _rfq_local_capacity_summary(system) -> dict:
    recs = getattr(system, "rfq_local_capacity", None) if system is not None else None
    if not isinstance(recs, dict) or not recs:
        return {}
    rows = [v for v in recs.values() if isinstance(v, dict)]
    if not rows:
        return {}
    ratios = np.asarray(
        [float(v.get("local_capacity_ratio", 0.0)) for v in rows], dtype=float
    )
    eligible = np.asarray(
        [float(v.get("eligible_local_count", 0.0)) for v in rows], dtype=float
    )
    room = np.asarray(
        [float(v.get("local_room_sum", 0.0)) for v in rows], dtype=float
    )
    return {
        "borrowers": int(len(rows)),
        "capacity_ratio_min": float(np.min(ratios)),
        "capacity_ratio_mean": float(np.mean(ratios)),
        "capacity_ratio_max": float(np.max(ratios)),
        "capacity_feasible_share": float(np.mean(ratios >= 1.0 - 1e-9)),
        "eligible_local_min": int(np.min(eligible)),
        "eligible_local_mean": float(np.mean(eligible)),
        "eligible_local_max": int(np.max(eligible)),
        "local_room_sum": float(np.sum(room)),
        "capped_known_total": int(
            sum(int(v.get("capped_known_count", 0)) for v in rows)
        ),
    }


def _note_rfq_queries(system, borrower_idx: int, lenders) -> None:
    if system is None:
        return
    rec = getattr(system, "rfq_unique_lenders_queried", None)
    if not isinstance(rec, dict):
        system.rfq_unique_lenders_queried = {}
        rec = system.rfq_unique_lenders_queried
    rec.setdefault(int(borrower_idx), set()).update(int(i) for i in (lenders or []))


def _outstanding_lenders_for_borrower(system, borrower_idx: int) -> list[int]:
    """Original lenders of outstanding contracts (relationship rollover)."""
    out: list[int] = []
    seen: set[int] = set()
    j = int(borrower_idx)
    if system is None:
        return out
    book = getattr(system, "contract_book", None)
    contracts = getattr(book, "contracts", None) if book is not None else None
    if not contracts:
        return out
    for c in contracts:
        if int(getattr(c, "borrower_idx", -1)) != j:
            continue
        if float(getattr(c, "remaining_principal", 0.0) or 0.0) <= 1e-9:
            continue
        i = int(getattr(c, "lender_idx", -1))
        if i < 0 or i == j or i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def build_local_lender_pool(
    lender_ids: list[int],
    borrower_idx: int,
    system,
    *,
    neigh: list[set[int]] | None = None,
    budget: int | None = None,
    banks: list | None = None,
    supply_left: dict | None = None,
    initial_demand: float | None = None,
    B_cap: float | None = None,
    pair_filled: dict[tuple[int, int], float] | None = None,
    min_trade: float = DAILY_RFQ_MIN_TRADE_SIZE,
) -> tuple[list[int], set[int], list[int]]:
    """
    Local RFQ universe for one borrower.

    Credit-eligible rollover / outstanding lenders are included and do **not**
    count against the discovery budget. New counterparties, in order, come from:
    historical trades → initial acquaintances ∪ current 1-hop exposure
    neighbors → 2-hop referrals → at most one random explorer.
    Never iterates remaining commercial banks.

    When balance-sheet context is supplied, known failed/cash-exhausted/
    pair-cap-exhausted names are skipped *before* they can consume one of the
    eight local discovery slots. The scan still stays inside the same permitted
    local sources; this is local execution filtering, not global search.
    """
    j = int(borrower_idx)
    if budget is None:
        budget = int(
            getattr(system, "local_unique_lender_budget", LOCAL_UNIQUE_LENDER_BUDGET)
            if system is not None
            else LOCAL_UNIQUE_LENDER_BUDGET
        )
    budget = max(0, int(budget))
    rng_m = getattr(system, "rng_matching", None) if system is not None else None
    lender_set = {int(i) for i in lender_ids if int(i) != j}

    n_banks = max(int(j) + 1, 1)
    if system is not None:
        n_banks = max(n_banks, int(getattr(system, "num_banks", n_banks) or n_banks))
    if neigh is None:
        L0 = getattr(system, "prev_exposure_matrix", None) if system is not None else None
        if L0 is None:
            L0 = getattr(system, "exposure_matrix", None) if system is not None else None
        neigh = _exposure_neighbor_sets(L0, n_banks)
    if j >= len(neigh):
        neigh = list(neigh) + [set() for _ in range(j - len(neigh) + 1)]

    hist = _historical_counterparties(system, j, use_prev=True)
    rollover = [
        int(i) for i in _outstanding_lenders_for_borrower(system, j)
        if int(i) in lender_set
    ]
    rollover_set = set(rollover)

    info_neigh = _info_neighbor_sets(system, neigh, n_banks)
    hop1 = [int(i) for i in info_neigh[j] if int(i) in lender_set]
    hop2: list[int] = []
    seen_h2: set[int] = set()
    for n in hop1:
        if int(n) >= len(info_neigh):
            continue
        for k in info_neigh[int(n)]:
            kk = int(k)
            if kk == j or kk not in lender_set or kk in seen_h2:
                continue
            seen_h2.add(kk)
            hop2.append(kk)

    pool: list[int] = []
    seen: set[int] = set()
    considered: set[int] = set()
    pair_filled = pair_filled or {}
    capacity_filter = (
        banks is not None
        and supply_left is not None
        and initial_demand is not None
        and B_cap is not None
    )
    tol = max(float(RFQ_EXHAUST_TOL), float(min_trade))
    skipped_pair_cap = 0
    skipped_cash = 0
    skipped_failed = 0

    def _append(idx: int) -> None:
        nonlocal skipped_pair_cap, skipped_cash, skipped_failed
        ii = int(idx)
        if ii in considered or ii not in lender_set:
            return
        considered.add(ii)
        if capacity_filter:
            failed = (
                ii < 0
                or ii >= len(banks)
                or (not banks[ii].get("is_active", True))
                or bool(banks[ii].get("absorbing_default"))
            )
            if failed:
                skipped_failed += 1
                _note_rfq_reject(system, "lender_failed")
                return
            if float(supply_left.get(ii, 0.0)) <= tol:
                skipped_cash += 1
                _note_rfq_reject(system, "lender_cash_exhausted")
                return
            pair_room = den_pair_trade_cap(
                banks,
                ii,
                j,
                initial_demand=float(initial_demand),
                B_max=float(B_cap),
                system=system,
                extra_existing=float(pair_filled.get((ii, j), 0.0)),
            )
            if float(pair_room) <= tol:
                skipped_pair_cap += 1
                _note_rfq_reject(system, "pair_cap_exhausted")
                return
        pool.append(ii)
        seen.add(ii)

    for i in rollover:
        _append(i)

    def discovery_n() -> int:
        return sum(1 for x in pool if x not in rollover_set)

    for group in (
        [int(i) for i in hist if int(i) in lender_set],
        hop1,
        hop2,
    ):
        if discovery_n() >= budget:
            break
        for i in _shuffle_ids(group, rng_m):
            if discovery_n() >= budget:
                break
            if int(i) in considered:
                continue
            _append(i)

    if discovery_n() < budget:
        rest = [
            int(i) for i in lender_ids
            if int(i) in lender_set and int(i) not in considered
        ]
        rest = _shuffle_ids(rest, rng_m)[: max(0, int(RFQ_RANDOM_EXPLORE_N))]
        for i in rest:
            if discovery_n() >= budget:
                break
            _append(i)

    if capacity_filter and system is not None:
        local_room_sum = 0.0
        for ii in pool:
            pair_room = den_pair_trade_cap(
                banks,
                int(ii),
                j,
                initial_demand=float(initial_demand),
                B_max=float(B_cap),
                system=system,
                extra_existing=float(pair_filled.get((int(ii), j), 0.0)),
            )
            local_room_sum += max(
                0.0,
                min(float(supply_left.get(int(ii), 0.0)), float(pair_room)),
            )
        cap_rec = getattr(system, "rfq_local_capacity", None)
        if not isinstance(cap_rec, dict):
            system.rfq_local_capacity = {}
            cap_rec = system.rfq_local_capacity
        dem = max(0.0, float(initial_demand))
        cap_rec[j] = {
            "initial_demand": dem,
            "local_room_sum": float(local_room_sum),
            "local_capacity_ratio": float(local_room_sum / max(dem, 1e-9)),
            "eligible_local_count": int(len(pool)),
            "eligible_discovered_count": int(discovery_n()),
            "eligible_rollover_count": int(sum(1 for x in pool if x in rollover_set)),
            "capped_known_count": int(skipped_pair_cap),
            "cash_exhausted_known_count": int(skipped_cash),
            "failed_known_count": int(skipped_failed),
        }

    return pool, hist, rollover


def select_round_rfq_candidates(
    pool: list[int],
    K: int,
    exhausted: set[int],
    *,
    rollover: list[int] | set[int] | None = None,
    pair_filled: dict[tuple[int, int], float] | None = None,
    borrower_idx: int = -1,
    queried_once: set[int] | None = None,
) -> list[int]:
    """Up to K lenders from one fixed local pool.

    Query every fresh local name before revisiting a partial/no-fill quote.
    This improves execution coverage without discovering any extra lender.
    """
    k = max(1, int(K))
    exh = {int(x) for x in (exhausted or set())}
    roll = {int(x) for x in (rollover or set())}
    j = int(borrower_idx)
    filled = pair_filled or {}
    queried = {int(x) for x in (queried_once or set())}
    avail = [int(i) for i in pool if int(i) not in exh]
    avail.sort(
        key=lambda i: (
            0 if i in roll and i not in queried else
            1 if i not in queried else
            2 if float(filled.get((int(i), j), 0.0)) > 1e-12 else
            3,
        )
    )
    return avail[:k]


def _filter_local_round_candidates(
    *,
    pool: list[int],
    exhausted: set[int],
    rollover: list[int],
    K: int,
    borrower_idx: int,
    banks: list,
    system,
    neigh: list[set[int]],
    max_degree: int,
    supply_left: dict,
    pair_filled: dict[tuple[int, int], float],
    demand_initial: dict,
    B_cap: float,
    min_trade: float,
    queried_once: set[int] | None = None,
) -> list[int]:
    """Drop truly exhausted local lenders, then take up to K for this round.

    Exhausted = hard reject / pair room / lendable cash / lender failed.
    Degree-cap names are skipped this round only so released edges can re-enter.
    """
    j = int(borrower_idx)
    init_dem = float(demand_initial.get(j, 0.0))
    tol = float(RFQ_EXHAUST_TOL)
    min_sz = float(min_trade)
    skip_this_round: set[int] = set()
    for i in list(pool):
        ii = int(i)
        if ii in exhausted:
            continue
        failed = (
            ii < 0
            or ii >= len(banks)
            or (not banks[ii].get("is_active", True))
            or bool(banks[ii].get("absorbing_default"))
        )
        if failed:
            exhausted.add(ii)
            _note_rfq_reject(system, "lender_failed")
            continue
        if not _degree_allows_pair(neigh, ii, j, max_degree):
            skip_this_round.add(ii)
            _note_rfq_reject(system, "degree_cap")
            continue
        if float(supply_left.get(ii, 0.0)) <= max(tol, min_sz):
            exhausted.add(ii)
            _note_rfq_reject(system, "lender_cash_exhausted")
            continue
        pair_cap = den_pair_trade_cap(
            banks,
            ii,
            j,
            initial_demand=init_dem,
            B_max=float(B_cap),
            system=system,
            extra_existing=float(pair_filled.get((ii, j), 0.0)),
        )
        if float(pair_cap) <= max(tol, min_sz):
            exhausted.add(ii)
            _note_rfq_reject(system, "pair_cap_exhausted")
    blocked = set(exhausted) | skip_this_round
    return select_round_rfq_candidates(
        pool, K, blocked,
        rollover=rollover,
        pair_filled=pair_filled,
        borrower_idx=j,
        queried_once=queried_once,
    )


def _mark_partial_or_exhausted_after_fill(
    *,
    system,
    exhausted: set[int],
    lender_idx: int,
    borrower_idx: int,
    amt: float,
    need: float,
    supply_left: dict,
    pair_filled: dict[tuple[int, int], float],
    banks: list,
    demand_initial: dict,
    B_cap: float,
    B_eff: float,
    min_trade: float,
) -> None:
    """Keep topping up after B-cap partial fills; exhaust only true capacity ends."""
    i = int(lender_idx)
    j = int(borrower_idx)
    tol = float(RFQ_EXHAUST_TOL)
    min_sz = float(min_trade)
    cash_left = float(supply_left.get(i, 0.0))
    pair_left = den_pair_trade_cap(
        banks, i, j,
        initial_demand=float(demand_initial.get(j, 0.0)),
        B_max=float(B_cap),
        system=system,
        extra_existing=float(pair_filled.get((i, j), 0.0)),
    )
    b_lim = min(float(B_eff), float(B_cap))
    if cash_left <= max(tol, min_sz):
        exhausted.add(i)
        _note_rfq_reject(system, "lender_cash_exhausted")
        return
    if float(pair_left) <= max(tol, min_sz):
        exhausted.add(i)
        _note_rfq_reject(system, "pair_cap_exhausted")
        return
    if float(amt) + tol < min(float(need), cash_left + float(amt), float(pair_left) + float(amt)):
        if abs(float(amt) - b_lim) <= max(tol, 1e-6) or float(amt) + tol >= b_lim:
            _note_rfq_reject(system, "B_cap_partial")


def discover_local_rfq_candidates(
    lender_ids: list[int],
    borrower_idx: int,
    K: int,
    system,
    exclude: set[int] | None = None,
    *,
    neigh: list[set[int]] | None = None,
    pool: list[int] | None = None,
    rollover: list[int] | None = None,
) -> tuple[list[int], set[int]]:
    """
    Limited-info RFQ: query up to K names from a local pool of size ≤ 8.

    Does not scan remaining commercial banks. ``exclude`` is the exhausted
    set (hard reject / pair room / cash / failed), not “already asked”.
    """
    if pool is None:
        pool, hist, roll = build_local_lender_pool(
            lender_ids, borrower_idx, system, neigh=neigh,
        )
    else:
        hist = _historical_counterparties(system, int(borrower_idx), use_prev=True)
        roll = list(rollover or [])
    chosen = select_round_rfq_candidates(
        pool, K, set(exclude or set()),
        rollover=roll,
        borrower_idx=int(borrower_idx),
    )
    return chosen, hist


def select_rfq_lender_candidates(
    cand_pool: list[int],
    borrower_idx: int,
    K: int,
    system,
    exclude: set[int] | None = None,
) -> list[int]:
    """Up to K names from the local pool. Never samples remaining commercial banks."""
    discovered, _hist = discover_local_rfq_candidates(
        cand_pool, borrower_idx, K, system, exclude=exclude,
    )
    return discovered


def relationship_deal_adjustments(
    system,
    lender_idx: int,
    borrower_idx: int,
    *,
    risk_markup: float,
    B_max: float,
) -> tuple[float, float, float]:
    """
    Returns (adjusted_markup, adjusted_B_max, strength).
    Related pairs: cheaper risk markup, larger bilateral capacity.
    """
    strength = relationship_strength(system, lender_idx, borrower_idx)
    if strength <= 1e-12 or system is None:
        return float(risk_markup), float(B_max), 0.0
    if not bool(getattr(system, "relationship_lending_enabled", True)):
        return float(risk_markup), float(B_max), strength
    disc = float(getattr(system, "relationship_markup_discount", RELATIONSHIP_MARKUP_DISCOUNT))
    boost = float(getattr(system, "relationship_size_boost", RELATIONSHIP_SIZE_BOOST))
    # Concave in strength: first trades matter most.
    w = strength / (strength + 2.0)
    adj_markup = float(risk_markup) * (1.0 - disc * w)
    adj_B = float(B_max) * (1.0 + boost * w)
    return adj_markup, adj_B, strength


class RFQMarket:
    """多轮局部 RFQ：固定 ≤8 家局部池，每轮向其中最多 K 家询价。

    有限信息范围（每家 borrower）：
    - 自身节点；
    - 原合约 lender 的 rollover（不计入发现预算）；
    - 历史交易对手；
    - 初始熟人 + 当前暴露一跳邻居；
    - 二跳推荐；
    - 最多 1 个随机探索节点。

    正式路径禁止全市场 ``to_pyg_graph`` 打分，禁止遍历尚未尝试的全部银行。
    """
    def __init__(self, max_rounds: int = DAILY_RFQ_MAX_ROUNDS, min_trade_size: float = DAILY_RFQ_MIN_TRADE_SIZE):
        self.max_rounds = max_rounds
        self.min_trade_size = min_trade_size

    def _run_rate_only(
        self,
        intentions: list[Intention],
        banks: list,
        step: int,
        B_max: float = 1200.0,
        K: int = DAILY_RFQ_K,
        borrower_risk_markup_spread: float = DAILY_RFQ_BORROWER_RISK_MARKUP,
        system=None,
    ) -> list[Trade]:
        """无 matcher 时退化为利率优先的 RFQ（关系对手优先进入候选集）。"""
        lenders = [x for x in intentions if x.role == "lender"]
        borrowers = [x for x in intentions if x.role == "borrower"]
        trades: list[Trade] = []
        supply_left = {x.bank_idx: x.quantity for x in lenders}
        demand_left = {x.bank_idx: x.quantity for x in borrowers}
        # Clamp borrower demand by bank-level IBL capacity room.
        asset_lambda = float(
            getattr(system, "ibl_cap_asset_lambda", DEFAULT_IBL_CAP_ASSET_LAMBDA)
            if system is not None
            else DEFAULT_IBL_CAP_ASSET_LAMBDA
        )
        demand_initial = {}
        for j in list(demand_left.keys()):
            if 0 <= int(j) < len(banks):
                room = ibl_borrowing_room(banks[int(j)], asset_lambda=asset_lambda)
                demand_left[j] = float(min(float(demand_left[j]), room))
            demand_initial[j] = float(demand_left[j])
        lender_ids = [x.bank_idx for x in lenders]
        n_banks = max(len(banks), 1)
        max_degree = int(getattr(system, "max_degree", DEN_MAX_DEGREE) if system is not None else DEN_MAX_DEGREE)
        L0 = getattr(system, "exposure_matrix", None) if system is not None else None
        neigh = _exposure_neighbor_sets(L0, n_banks)
        pair_filled: dict[tuple[int, int], float] = {}
        B_cap = float(B_max)
        _reset_rfq_reject_counts(system)

        local_pools: dict[int, list[int]] = {}
        local_hist: dict[int, set[int]] = {}
        local_rollover: dict[int, list[int]] = {}
        exhausted_by_borrower: dict[int, set[int]] = {}
        queried_once_by_borrower: dict[int, set[int]] = {}
        budget = int(
            getattr(system, "local_unique_lender_budget", LOCAL_UNIQUE_LENDER_BUDGET)
            if system is not None
            else LOCAL_UNIQUE_LENDER_BUDGET
        )
        for bo in borrowers:
            j = int(bo.bank_idx)
            pool, hist, roll = build_local_lender_pool(
                lender_ids,
                j,
                system,
                neigh=neigh,
                budget=budget,
                banks=banks,
                supply_left=supply_left,
                initial_demand=float(demand_initial.get(j, 0.0)),
                B_cap=B_cap,
                pair_filled=pair_filled,
                min_trade=self.min_trade_size,
            )
            local_pools[j] = pool
            local_hist[j] = hist
            local_rollover[j] = roll
            exhausted_by_borrower[j] = set()
            queried_once_by_borrower[j] = set()
            if not pool:
                _note_rfq_reject(system, "no_local_candidate")
        if system is not None:
            system.rfq_local_rollover = {int(j): list(r) for j, r in local_rollover.items()}

        for _round in range(self.max_rounds):
            round_trades: list[tuple[int, int, float, float]] = []
            teacher_mode = str(getattr(system, "rfq_teacher_mode", "") or "").lower()
            use_teacher = teacher_mode in ("multi_factor", "multifactor", "v4", "v5", "v6")
            any_open = False
            for bo in borrowers:
                j, need = bo.bank_idx, demand_left.get(bo.bank_idx, 0.0)
                if need < self.min_trade_size:
                    continue
                init_dem = float(demand_initial.get(j, 0.0))
                theta_risk = float(getattr(system, "car_cutoff", 0.08)) if system is not None else 0.08
                b_risk = _borrower_credit_risk(banks[j], car_threshold=theta_risk) if j < len(banks) else 0.0
                base_markup = float(borrower_risk_markup_spread) * b_risk
                exhausted = exhausted_by_borrower.setdefault(int(j), set())
                cand_pool = _filter_local_round_candidates(
                    pool=local_pools.get(int(j), []),
                    exhausted=exhausted,
                    rollover=local_rollover.get(int(j), []),
                    K=K,
                    borrower_idx=int(j),
                    banks=banks,
                    system=system,
                    neigh=neigh,
                    max_degree=max_degree,
                    supply_left=supply_left,
                    pair_filled=pair_filled,
                    demand_initial=demand_initial,
                    B_cap=B_cap,
                    min_trade=self.min_trade_size,
                    queried_once=queried_once_by_borrower.setdefault(int(j), set()),
                )
                hist_cps = local_hist.get(int(j), set())
                if not cand_pool:
                    continue
                any_open = True
                _note_rfq_queries(system, int(j), cand_pool)
                queried_once_by_borrower.setdefault(int(j), set()).update(
                    int(i) for i in cand_pool
                )
                lender_by_idx = {x.bank_idx: x for x in lenders}
                candidates = []
                pair_feats: dict[int, list[float]] = {}
                for i in cand_pool:
                    if not _degree_allows_pair(neigh, i, j, max_degree):
                        continue
                    # Exact remaining supply is private until quote/fill time.
                    supp = supply_left.get(i, 0.0)
                    if supp < self.min_trade_size:
                        continue
                    le = lender_by_idx[i]
                    risk_markup, B_eff, rel = relationship_deal_adjustments(
                        system, i, j, risk_markup=base_markup, B_max=B_max
                    )
                    B_eff = min(float(B_eff), float(B_max))
                    pair_feats[int(i)] = [
                        pair_bilateral_exposure_ratio(
                            system, banks, int(i), int(j),
                            proposed_amt=0.0,
                            extra_existing=float(pair_filled.get((int(i), int(j)), 0.0)),
                        )
                    ]
                    if le.reserve_bid > bo.reserve_ask:
                        exhausted.add(int(i))
                        _note_rfq_reject(system, "rate_reject")
                        _note_rfq_reject(system, "hard_rejected")
                        continue
                    if le.reserve_bid <= bo.reserve_ask:
                        rate = (le.reserve_bid + bo.reserve_ask) / 2.0 + risk_markup
                        if rate > float(bo.reserve_ask):
                            exhausted.add(int(i))
                            _note_rfq_reject(system, "rate_reject")
                            _note_rfq_reject(system, "hard_rejected")
                            continue
                        room = ibl_borrowing_room(
                            banks[j], asset_lambda=asset_lambda,
                            extra_new_borrowing=init_dem - float(need),
                        ) if 0 <= j < len(banks) else need
                        pair_cap = den_pair_trade_cap(
                            banks, i, j,
                            initial_demand=init_dem,
                            B_max=float(B_max),
                            system=system,
                            extra_existing=float(pair_filled.get((int(i), int(j)), 0.0)),
                        )
                        amt = min(supp, need, B_eff, room, pair_cap)
                        min_sz = self.min_trade_size * (0.5 if rel > 1e-12 else 1.0)
                        if amt >= min_sz:
                            if use_teacher:
                                score = teacher_rfq_pair_score(
                                    system, banks, i, j,
                                    quote_rate=float(rate),
                                    proposed_amt=float(amt),
                                    car_threshold=theta_risk,
                                    extra_existing=float(
                                        pair_filled.get((int(i), int(j)), 0.0)
                                    ),
                                )
                                # Higher teacher score first; rate as tie-break.
                                candidates.append((-score, rate, i, j, amt))
                            else:
                                candidates.append((0 if rel > 1e-12 else 1, rate, i, j, amt))
                candidates.sort(key=lambda x: (x[0], x[1]))
                accepted_ids: list[int] = []
                for _rel_rank, rate, i, j, amt in candidates:
                    if not _degree_allows_pair(neigh, i, j, max_degree):
                        continue
                    supp = supply_left.get(i, 0.0)
                    ne = demand_left.get(j, 0.0)
                    _m, B_eff, rel = relationship_deal_adjustments(
                        system, i, j, risk_markup=0.0, B_max=B_max
                    )
                    room = ibl_borrowing_room(
                        banks[j], asset_lambda=asset_lambda,
                        extra_new_borrowing=init_dem - float(ne),
                    ) if 0 <= j < len(banks) else ne
                    pair_cap = den_pair_trade_cap(
                        banks, i, j,
                        initial_demand=init_dem,
                        B_max=float(B_max),
                        system=system,
                        extra_existing=float(pair_filled.get((int(i), int(j)), 0.0)),
                    )
                    amt = min(amt, supp, ne, room, B_eff, pair_cap)
                    min_sz = self.min_trade_size * (0.5 if rel > 1e-12 else 1.0)
                    if amt < min_sz:
                        continue
                    round_trades.append((i, j, amt, rate))
                    accepted_ids.append(int(i))
                    supply_left[i] = supply_left.get(i, 0.0) - amt
                    demand_left[j] = demand_left.get(j, 0.0) - amt
                    pair_filled[(int(i), int(j))] = float(
                        pair_filled.get((int(i), int(j)), 0.0) + amt
                    )
                    _register_pair_degree(neigh, i, j)
                    _mark_partial_or_exhausted_after_fill(
                        system=system,
                        exhausted=exhausted,
                        lender_idx=int(i),
                        borrower_idx=int(j),
                        amt=float(amt),
                        need=float(ne),
                        supply_left=supply_left,
                        pair_filled=pair_filled,
                        banks=banks,
                        demand_initial=demand_initial,
                        B_cap=B_cap,
                        B_eff=float(B_eff),
                        min_trade=self.min_trade_size,
                    )
                    need = demand_left.get(j, 0.0)
                    if need < self.min_trade_size:
                        break
                append_rfq_training_event(
                    system,
                    step=step,
                    borrower_idx=int(bo.bank_idx),
                    candidates=list(cand_pool),
                    accepted=accepted_ids,
                    hist_cps=hist_cps,
                    pair_feats=pair_feats,
                )
            for i, j, amt, rate in round_trades:
                trades.append(Trade(lender_idx=i, borrower_idx=j, amount=amt, rate=rate, step_executed=step))
            if not any_open:
                break
        return trades

    def run(
        self,
        intentions: list[Intention],
        banks: list,
        system,              # ★新增：拿 to_pyg_graph / gnn_context
        step: int,
        B_max: float = 1200.0,
        K: int = DAILY_RFQ_K,         # ★每轮每个 borrower 询价候选数
        delta_r: float = DAILY_RFQ_QUOTE_SPREAD,   # ★报价上浮空间
        combine: str = "min",    # "min" or "geom"
        bargain_power_borrower: float = 0.5,  # ★borrower 还价力度
        lender_markup_floor: float = DAILY_RFQ_MARKUP_FLOOR,   # ★lender 最低接受加点
        borrower_risk_markup_spread: float = DAILY_RFQ_BORROWER_RISK_MARKUP,  # ★高风险 borrower 额外加点
        max_negotiation_rounds: int = 1,      # ★每对 borrower-lender 的议价轮数
    ) -> list[Trade]:
        blocked = getattr(system, "rollover_blocked_borrowers", None)
        policy = getattr(system, "rollover_borrow_policy", ROLLOVER_BORROW_COUPON_CLEARED)
        if blocked and str(policy).lower() == ROLLOVER_BORROW_BLOCK_ALL:
            intentions = [
                x for x in intentions
                if not (x.role == "borrower" and x.bank_idx in blocked)
            ]
        lenders = [x for x in intentions if x.role == "lender"]
        borrowers = [x for x in intentions if x.role == "borrower"]
        trades: list[Trade] = []
        supply_left = {x.bank_idx: float(x.quantity) for x in lenders}
        demand_left = {x.bank_idx: float(x.quantity) for x in borrowers}
        # Clamp borrower demand by bank-level IBL capacity room.
        asset_lambda = float(
            getattr(system, "ibl_cap_asset_lambda", DEFAULT_IBL_CAP_ASSET_LAMBDA)
        )
        demand_initial = dict(demand_left)
        for j in list(demand_left.keys()):
            if 0 <= int(j) < len(banks):
                room = ibl_borrowing_room(banks[int(j)], asset_lambda=asset_lambda)
                demand_left[j] = float(min(float(demand_left[j]), room))
                demand_initial[j] = float(demand_left[j])
        # ===== 本地打分器（GNN/Matcher）=====
        ctx = getattr(system, "gnn_context", None) or {}
        matcher = ctx.get("matcher", None)
        device = ctx.get("device", None)
        # Prefer per-sim RFQ knobs (baseline / ablation) when present.
        K = int(getattr(system, "rfq_k", K))
        self.min_trade_size = float(
            getattr(system, "rfq_min_trade_size", self.min_trade_size)
        )
        borrower_risk_markup_spread = float(
            getattr(system, "rfq_borrower_risk_markup", borrower_risk_markup_spread)
        )
        # 正式 DEN 必须有 GNN matcher；rate-only 仅调试（allow_rate_only_fallback）
        if matcher is None:
            allow_fallback = bool(getattr(system, "allow_rate_only_fallback", False))
            require_gnn = bool(getattr(system, "require_gnn", True))
            if require_gnn or not allow_fallback:
                raise RuntimeError(
                    "DEN RFQ requires a GNN matcher; rate-only is disabled for formal "
                    "experiments. Load gnn_pair_matcher_v6_local.pth, or set "
                    "allow_rate_only_fallback=True only for debugging."
                )
            trades = self._run_rate_only(
                intentions, banks, step, B_max=B_max, K=K,
                borrower_risk_markup_spread=borrower_risk_markup_spread,
                system=system,
            )
            log_rfq_match_diagnostics(system, step, intentions, trades, B_max=B_max)
            return trades
        # 预取每家 bank 的 reserve 值（r_min/r_max）；GNN 打分在每家 borrower 的局部子图上完成
        r_min = {x.bank_idx: float(x.reserve_bid) for x in lenders}
        r_max = {x.bank_idx: float(x.reserve_ask) for x in borrowers}
        lender_ids = [x.bank_idx for x in lenders]
        bargain_power_borrower = min(max(float(bargain_power_borrower), 0.0), 1.0)
        lender_markup_floor = max(float(lender_markup_floor), 0.0)
        borrower_risk_markup_spread = max(float(borrower_risk_markup_spread), 0.0)
        max_negotiation_rounds = max(1, int(max_negotiation_rounds))
        n_banks = max(len(banks), 1)
        max_degree = int(getattr(system, "max_degree", DEN_MAX_DEGREE))
        neigh = _exposure_neighbor_sets(getattr(system, "exposure_matrix", None), n_banks)
        B_cap = float(B_max)
        pair_filled: dict[tuple[int, int], float] = {}
        _reset_rfq_reject_counts(system)

        local_pools: dict[int, list[int]] = {}
        local_hist: dict[int, set[int]] = {}
        local_rollover: dict[int, list[int]] = {}
        exhausted_by_borrower: dict[int, set[int]] = {}
        queried_once_by_borrower: dict[int, set[int]] = {}
        budget = int(
            getattr(system, "local_unique_lender_budget", LOCAL_UNIQUE_LENDER_BUDGET)
        )
        for bo in borrowers:
            j = int(bo.bank_idx)
            pool, hist, roll = build_local_lender_pool(
                lender_ids,
                j,
                system,
                neigh=neigh,
                budget=budget,
                banks=banks,
                supply_left=supply_left,
                initial_demand=float(demand_initial.get(j, 0.0)),
                B_cap=B_cap,
                pair_filled=pair_filled,
                min_trade=self.min_trade_size,
            )
            local_pools[j] = pool
            local_hist[j] = hist
            local_rollover[j] = roll
            exhausted_by_borrower[j] = set()
            queried_once_by_borrower[j] = set()
            if not pool:
                _note_rfq_reject(system, "no_local_candidate")
        if system is not None:
            system.rfq_local_rollover = {int(j): list(r) for j, r in local_rollover.items()}

        for _round in range(self.max_rounds):
            any_trade = False
            any_open = False
            proposals_by_lender: dict[int, list[tuple[float, float, int, float, float]]] = {}
            rfq_round_meta: dict[int, dict] = {}
            accepted_ids_round: dict[int, list[int]] = {}

            # --- 阶段1：本轮所有 borrower 先同时发 RFQ / counter offer，不立即占用 lender 额度 ---
            for bo in borrowers:
                j = bo.bank_idx
                need = float(demand_left.get(j, 0.0))
                if need < self.min_trade_size:
                    continue
                init_dem = float(demand_initial.get(j, 0.0))
                theta_risk = float(getattr(system, "car_cutoff", 0.08)) if system is not None else 0.08
                from bank_econ_shared import borrower_origination_risk_q
                q_j = (
                    float(
                        borrower_origination_risk_q(
                            banks[j],
                            system=system,
                            bank_idx=int(j),
                            car_threshold=theta_risk,
                        )
                    )
                    if j < len(banks)
                    else 0.0
                )
                b_risk = q_j
                base_markup = borrower_risk_markup_spread * b_risk
                exhausted = exhausted_by_borrower.setdefault(int(j), set())
                cand_pool = _filter_local_round_candidates(
                    pool=local_pools.get(int(j), []),
                    exhausted=exhausted,
                    rollover=local_rollover.get(int(j), []),
                    K=K,
                    borrower_idx=int(j),
                    banks=banks,
                    system=system,
                    neigh=neigh,
                    max_degree=max_degree,
                    supply_left=supply_left,
                    pair_filled=pair_filled,
                    demand_initial=demand_initial,
                    B_cap=B_cap,
                    min_trade=self.min_trade_size,
                    queried_once=queried_once_by_borrower.setdefault(int(j), set()),
                )
                hist_cps = local_hist.get(int(j), set())
                if not cand_pool:
                    continue
                any_open = True
                _note_rfq_queries(system, int(j), cand_pool)
                queried_once_by_borrower.setdefault(int(j), set()).update(
                    int(i) for i in cand_pool
                )
                # 局部子图：自身 + 历史直接对手 + 本轮候选（与训练共用）
                local_graph, id_map = build_borrower_local_graph(
                    system, j, cand_pool, hist_cps, use_prev=True, mask_private=True
                )
                if int(j) not in id_map:
                    continue
                pairs_fwd = []
                pair_feat_rows = []
                scored_lenders = []
                pair_feats_map: dict[int, list[float]] = {}
                for i in cand_pool:
                    if i not in id_map:
                        continue
                    scored_lenders.append(int(i))
                    pairs_fwd.append((int(id_map[i]), int(id_map[j])))
                    pf = [
                        pair_bilateral_exposure_ratio(
                            system, banks, int(i), int(j),
                            proposed_amt=0.0,
                            extra_existing=float(pair_filled.get((int(i), int(j)), 0.0)),
                        )
                    ]
                    pair_feat_rows.append(pf)
                    pair_feats_map[int(i)] = pf
                if not pairs_fwd:
                    continue
                rfq_round_meta[int(j)] = {
                    "candidates": list(cand_pool),
                    "hist": hist_cps,
                    "pair_feats": pair_feats_map,
                }
                a_fwd = matcher.score_pairs(
                    local_graph, pairs_fwd, device=device, pair_features=pair_feat_rows
                )  # (m,)
                score_by_lender = {
                    int(scored_lenders[t]): float(a_fwd[t]) for t in range(len(scored_lenders))
                }
                teacher_scores_map = {
                    int(i): teacher_rfq_pair_score(
                        system, banks, int(i), int(j),
                        car_threshold=theta_risk,
                        extra_existing=float(pair_filled.get((int(i), int(j)), 0.0)),
                    )
                    for i in scored_lenders
                }
                rfq_round_meta[int(j)]["teacher_scores"] = teacher_scores_map
                for idx, i in enumerate(cand_pool):
                    if not _degree_allows_pair(neigh, i, j, max_degree):
                        _note_rfq_reject(system, "degree_cap")
                        continue
                    if float(supply_left.get(i, 0.0)) < self.min_trade_size:
                        exhausted.add(int(i))
                        _note_rfq_reject(system, "lender_cash_exhausted")
                        continue
                    if int(i) not in score_by_lender:
                        exhausted.add(int(i))
                        _note_rfq_reject(system, "gnn_reject")
                        _note_rfq_reject(system, "hard_rejected")
                        continue
                    a_gnn = float(score_by_lender[int(i)])
                    t_unit = _teacher_score_to_unit(
                        float(teacher_scores_map.get(int(i), 0.0))
                    )
                    a = _blend_gnn_teacher_score(a_gnn, t_unit)
                    a_l = a
                    a_b = a
                    risk_markup, B_eff, rel = relationship_deal_adjustments(
                        system, i, j, risk_markup=base_markup, B_max=B_cap
                    )
                    B_eff = min(float(B_eff), B_cap)
                    # lender根据偏好 + 借款人信用风险抬价/拒绝；关系对手加点更低
                    quote_rate = float(r_min[i] + delta_r * (1.0 - a_l) + risk_markup)
                    if quote_rate > float(r_max[j]):
                        exhausted.add(int(i))
                        _note_rfq_reject(system, "rate_reject")
                        _note_rfq_reject(system, "hard_rejected")
                        continue
                    # Relationship soft information: small boost only — cannot
                    # overturn origination-q screening.
                    if rel > 1e-12:
                        a_l = min(1.0, a_l + 0.05 * (rel / (rel + 2.0)))
                        a_b = min(1.0, a_b + 0.05 * (rel / (rel + 2.0)))
                    if combine == "geom":
                        a = (max(a_b, 0.0) * max(a_l, 0.0)) ** 0.5
                    else:
                        a = min(a_b, a_l)

                    deal_rate = quote_rate
                    accepted = False
                    for _neg in range(max_negotiation_rounds):
                        counter_rate = quote_rate - bargain_power_borrower * (quote_rate - float(r_min[i])) * max(a_b, 0.0)
                        counter_rate = min(float(r_max[j]), max(float(r_min[i]), float(counter_rate)))
                        lender_accept_rate = float(
                            r_min[i] + risk_markup + lender_markup_floor * (1.0 - max(a_l, 0.0))
                        )
                        if counter_rate >= lender_accept_rate:
                            deal_rate = counter_rate
                            accepted = True
                            break
                        quote_rate = min(float(r_max[j]), (quote_rate + lender_accept_rate) / 2.0)
                    if not accepted:
                        exhausted.add(int(i))
                        _note_rfq_reject(system, "gnn_reject")
                        _note_rfq_reject(system, "hard_rejected")
                        continue

                    # 真实surplus用协商后的成交利率（更像双边市场）
                    surplus = float(r_max[j] - deal_rate)
                    if surplus <= 1e-12:
                        exhausted.add(int(i))
                        _note_rfq_reject(system, "rate_reject")
                        _note_rfq_reject(system, "hard_rejected")
                        continue
                    borrower_score = a * surplus
                    room = ibl_borrowing_room(
                        banks[j],
                        asset_lambda=asset_lambda,
                        extra_new_borrowing=float(demand_initial.get(j, 0.0)) - float(need),
                    ) if 0 <= j < len(banks) else need
                    init_dem = float(demand_initial.get(j, 0.0))
                    pair_cap = den_pair_trade_cap(
                        banks, i, j,
                        initial_demand=init_dem,
                        B_max=B_cap,
                        system=system,
                        extra_existing=float(pair_filled.get((int(i), int(j)), 0.0)),
                    )
                    amt_cap = min(
                        float(B_eff), float(need), float(supply_left.get(i, 0.0)),
                        float(room), B_cap, float(pair_cap),
                    )
                    post_exp = pair_bilateral_exposure_ratio(
                        system, banks, i, j,
                        proposed_amt=float(amt_cap),
                        extra_existing=float(pair_filled.get((int(i), int(j)), 0.0)),
                    )
                    lender_score = lender_risk_adjusted_accept_score(
                        gnn_score=float(a_l),
                        deal_rate=float(deal_rate),
                        lender_min_rate=float(r_min[i]),
                        borrower_risk=float(q_j),
                        post_trade_exposure_ratio=post_exp,
                        rate_norm_scale=max(float(delta_r), 1e-6),
                        teacher_unit=float(t_unit),
                    )
                    min_sz = self.min_trade_size * (0.5 if rel > 1e-12 else 1.0)
                    if amt_cap >= min_sz:
                        if (
                            float(amt_cap) + 1e-9 < min(float(need), float(supply_left.get(i, 0.0)), float(pair_cap))
                            and abs(float(amt_cap) - min(float(B_eff), B_cap)) <= max(1e-9, 1e-6)
                        ):
                            _note_rfq_reject(system, "B_cap_partial")
                        proposals_by_lender.setdefault(int(i), []).append(
                            (float(lender_score), float(borrower_score), int(j), float(deal_rate), float(amt_cap))
                        )

            if not any_open:
                break

            # --- 阶段2：每个 lender 汇总本轮收到的多家 borrower 请求，再按自身偏好和收益统一接受 ---
            accepted_by_borrower: dict[int, list[tuple[float, int, float, float]]] = {}
            provisional_supply_left = dict(supply_left)
            for i, proposals in proposals_by_lender.items():
                proposals.sort(key=lambda x: (x[0], x[3], x[1]), reverse=True)
                for lender_score, borrower_score, j, deal_rate, amt_cap in proposals:
                    if not _degree_allows_pair(neigh, i, j, max_degree):
                        continue
                    supp = float(provisional_supply_left.get(i, 0.0))
                    if supp < self.min_trade_size:
                        break
                    need = float(demand_left.get(j, 0.0))
                    if need < self.min_trade_size:
                        continue
                    room = ibl_borrowing_room(
                        banks[j],
                        asset_lambda=asset_lambda,
                        extra_new_borrowing=float(demand_initial.get(j, 0.0)) - float(need),
                    ) if 0 <= j < len(banks) else need
                    pair_cap = den_pair_trade_cap(
                        banks, i, j,
                        initial_demand=float(demand_initial.get(j, 0.0)),
                        B_max=B_cap,
                        system=system,
                        extra_existing=float(pair_filled.get((int(i), int(j)), 0.0)),
                    )
                    accepted_amt = min(float(amt_cap), supp, need, room, B_cap, pair_cap)
                    if accepted_amt < self.min_trade_size:
                        continue
                    provisional_supply_left[i] = supp - accepted_amt
                    accepted_by_borrower.setdefault(int(j), []).append(
                        (float(borrower_score), int(i), float(deal_rate), float(accepted_amt))
                    )

            # --- 阶段3：borrower 在被 lender 接受的报价中排序确认成交 ---
            for bo in borrowers:
                j = bo.bank_idx
                accepted_quotes = accepted_by_borrower.get(int(j), [])
                if not accepted_quotes:
                    continue
                accepted_quotes.sort(key=lambda x: x[0], reverse=True)
                for borrower_score, i, deal_rate, accepted_amt in accepted_quotes:
                    if not _degree_allows_pair(neigh, i, j, max_degree):
                        continue
                    need = float(demand_left.get(j, 0.0))
                    if need < self.min_trade_size:
                        break
                    supp = float(supply_left.get(i, 0.0))
                    if supp < self.min_trade_size:
                        continue
                    room = ibl_borrowing_room(
                        banks[j],
                        asset_lambda=asset_lambda,
                        extra_new_borrowing=float(demand_initial.get(j, 0.0)) - float(need),
                    ) if 0 <= j < len(banks) else need
                    pair_cap = den_pair_trade_cap(
                        banks, i, j,
                        initial_demand=float(demand_initial.get(j, 0.0)),
                        B_max=B_cap,
                        system=system,
                        extra_existing=float(pair_filled.get((int(i), int(j)), 0.0)),
                    )
                    amt = min(float(accepted_amt), need, supp, room, B_cap, pair_cap)
                    if amt < self.min_trade_size:
                        continue
                    trades.append(Trade(lender_idx=int(i), borrower_idx=int(j), amount=float(amt), rate=float(deal_rate), step_executed=int(step)))
                    supply_left[i] = supp - amt
                    demand_left[j] = need - amt
                    pair_filled[(int(i), int(j))] = float(
                        pair_filled.get((int(i), int(j)), 0.0) + amt
                    )
                    accepted_ids_round.setdefault(int(j), []).append(int(i))
                    _register_pair_degree(neigh, i, j)
                    _m, B_eff_fill, _rel = relationship_deal_adjustments(
                        system, i, j, risk_markup=0.0, B_max=B_cap
                    )
                    _mark_partial_or_exhausted_after_fill(
                        system=system,
                        exhausted=exhausted_by_borrower.setdefault(int(j), set()),
                        lender_idx=int(i),
                        borrower_idx=int(j),
                        amt=float(amt),
                        need=float(need),
                        supply_left=supply_left,
                        pair_filled=pair_filled,
                        banks=banks,
                        demand_initial=demand_initial,
                        B_cap=B_cap,
                        B_eff=float(B_eff_fill),
                        min_trade=self.min_trade_size,
                    )
                    any_trade = True
            for j_meta, meta in rfq_round_meta.items():
                append_rfq_training_event(
                    system,
                    step=step,
                    borrower_idx=int(j_meta),
                    candidates=list(meta.get("candidates") or []),
                    accepted=list(accepted_ids_round.get(int(j_meta), [])),
                    hist_cps=meta.get("hist") or [],
                    pair_feats=meta.get("pair_feats") or {},
                    teacher_scores=meta.get("teacher_scores") or {},
                )
            if not any_open:
                break
        log_rfq_match_diagnostics(system, step, intentions, trades, B_max=B_max)
        return trades


# --- 6) 央行走廊 + 便利 ---
@dataclass
class CentralBankCorridor:
    """央行利率走廊：存款便利利率、贷款便利利率；0 号银行为央行。"""
    deposit_rate: float   # 存款便利（银行存央行）
    lending_rate: float   # 贷款便利（央行借给银行）
    base_rate: float     # 政策利率（走廊中点附近）

    def use_deposit_facility(self, bank_idx: int, amount: float, banks: list) -> None:
        """银行将 amount 存入央行（0）：增加央行负债、银行资产为 0（或记入 liquid）。"""
        if bank_idx == 0 or amount <= 0:
            return
        if bank_idx < len(banks):
            banks[bank_idx]["liquid_assets"] = float(banks[bank_idx].get("liquid_assets", 0.0)) - amount
        if len(banks) > 0:
            banks[0]["liquid_assets"] = float(banks[0].get("liquid_assets", 0.0)) + amount

    def use_lending_facility(self, bank_idx: int, amount: float, banks: list) -> None:
        """央行向银行借出 amount：央行资产增加，银行 liquid 增加。"""
        if bank_idx == 0 or amount <= 0:
            return
        if len(banks) > 0:
            banks[0]["liquid_assets"] = float(banks[0].get("liquid_assets", 0.0)) - amount
        if bank_idx < len(banks):
            banks[bank_idx]["liquid_assets"] = float(banks[bank_idx].get("liquid_assets", 0.0)) + amount


def build_L_from_contracts(contracts: list[Contract], n: int) -> np.ndarray:
    """从合约列表构建敞口矩阵 L（用于 EN）。"""
    L = np.zeros((n, n), dtype=float)
    for c in contracts:
        L[c.lender_idx, c.borrower_idx] += c.principal
        L[c.borrower_idx, c.lender_idx] -= c.principal
    return L


# --- 7) Step4: liquidity default + EN + recovery ---
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


# --- 8) 指标扩展与验证 ---
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
    """从 Decentralized 状态（banks + ContractBook）计算与 baseline 口径一致的系统性风险 SR。"""
    _ = aggregate_contracts_to_exposure_matrix_at_step(book, n, current_step)
    sr, _, _, _ = _systemic_risk_from_banks(
        banks, weights=weights, car_threshold=float(car_threshold)
    )
    return sr


def validate_decentralized_vs_baseline(
    sr_baseline: float, sr_decentralized: float, tol: float = 0.15
) -> bool:
    """若两者主指标（SR）在 tol 内即视为一致。"""
    return abs(sr_baseline - sr_decentralized) <= tol


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
        "core_capital":             float(b.get("core_capital", 0.0)),
        "liquid_assets":            float(b.get("liquid_assets", 0.0)),
        "current_liabilities":      float(b.get("current_liabilities", 0.0)),
        "interbank_assets":         float(b.get("interbank_assets", 0.0)),
        "interbank_liabilities":    float(b.get("interbank_liabilities", 0.0)),

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

    from interbank_matcher_shared import normalize_matcher_feature_vector

    return normalize_matcher_feature_vector([vals[k] for k in FEATURE_ORDER_15])


# 保留：项目根路径（可用作备用）
BASE_DIR = Path(__file__).resolve().parent

# —— 统一输入/输出目录 —— 
MODEL_DIR  = INPUT_DIR
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "gnn_lstm_model.pth"        # legacy local path
DATA_FILE  = MODEL_DIR / "bank_contagion_data_decentralized.json"  # DEN scratch cache
from interbank_matcher_shared import (
    SHARED_DATA_FILE,
    SHARED_LOCAL_DATA_FILE,
    SHARED_MATCHER_PATH,
    SHARED_GNN_LSTM_PATH,
    GNN_PAIR_MATCHER_V2_PATH,
    GNN_PAIR_MATCHER_V3_LOCAL_PATH,
    GNN_PAIR_MATCHER_V4_LOCAL_PATH,
    GNN_PAIR_MATCHER_V5_LOCAL_PATH,
    GNN_PAIR_MATCHER_V6_LOCAL_PATH,
    DATA_FILE_DECENTRALIZED,
    DATASET_SCHEMA_VERSION,
    LABEL_DEFINITION_V6,
    load_json_dataset,
    save_json_dataset,
    should_regenerate_dataset,
    matcher_meta,
    save_matcher_checkpoint,
    load_matcher_state_dict,
)
DATA_FILE = DATA_FILE_DECENTRALIZED
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
                    # L[i,j]>0 表示 i 借出给 j，即 j 欠 i；表述为“j 对 i 的债务”
                    high_exposure.append(
                        (self.banks[j]['name'], self.banks[i]['name'], self.exposure_matrix[i, j])
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
        env_ss, proj_ss, match_ss, acq_ss = ss.spawn(4)
        self.rng_environment = np.random.default_rng(env_ss)
        self.rng_project = np.random.default_rng(proj_ss)
        self.rng_matching = np.random.default_rng(match_ss)
        self.rng_acquaintance = np.random.default_rng(acq_ss)
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
        self.max_degree = DEN_MAX_DEGREE
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
            self.max_degree   = DEN_MAX_DEGREE
            self.central_edge_ratio = 0.40

        # （可选）避免 A↔B 来回拆借的“反向记忆”
        self.forbid_reciprocal_history = True
        self.reciprocal_cooldown = None
        self._pair_dir  = {}
        self._pair_step = {}

        # 让 CAR 阈值变成实例属性
        self.car_cutoff = 0.08
        self.lcr_cutoff = DAILY_INTERBANK_ROLE_LCR_CUTOFF
        self.interbank_lcr_target = DAILY_INTERBANK_INTENTION_LCR_TARGET

        self.prev_exposure_matrix = None

        # Decentralized：合约簿（到期/现金流）；baseline 主循环仍只用 exposure_matrix，指标口径不变
        self.contract_book = ContractBook()
        self.last_avg_rate: float | None = None  # 第二阶段：上一期真实成交利率均值
        self.interbank_contract_maturity = DAILY_INTERBANK_CONTRACT_MATURITY

    # === 统一的“安全版 CAR”计算函数（唯一 RWA 口径：regulatory_rwa）===
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
            int(step) >= int(getattr(self, "network_stability_min_step", 50))
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


    def initialize_network(self):
        """
        初始化银行、仅项目投资；同业边由撮合函数生成并做现金结算。
        """
        # 1. 随机设定市场环境与相关参数
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
        self.borrowed_cash[:] = 0.0
        self.borrowed_origination_risk_sum[:] = 0.0
        self.ib_asset[:] = 0.0
        self.ib_liab[:] = 0.0
        for i in range(self.num_banks):
            t = self.bank_types[i]

            # —— 环境相关参数设定 ——
            if self.market_environment == 'bull':
                cap_mul = random.uniform(1.1, 1.2)
                liq_mul = random.uniform(1.1, 1.2)
                lia_mul = random.uniform(0.8, 0.9)
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

            # 投资额先按 liquid_target 估算，再进入 CAR 反推闭合
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
            bank["lag_car"] = float(bank["capital_adequacy_ratio"])
            bank["lag_lcr"] = float(bank["liquidity_coverage_ratio"])
            self.banks.append(bank)

        # —— 同业矩阵置零；不在初始化撮合，首次交易从 simulate_step(0) 的 RFQ 开始 ——
        self.exposure_matrix = np.zeros((self.num_banks, self.num_banks), dtype=float)
        self.contract_book = ContractBook()
        from bank_econ_shared import reset_screening_loss
        reset_screening_loss(self, totals=True, step=True)
        # 环形初始熟人网：只表示认识/可询价，不产生贷款和资产负债。
        acq_rng = getattr(self, "rng_acquaintance", None) or getattr(self, "rng_matching", None)
        self.initial_acquaintance_degree = int(
            getattr(self, "initial_acquaintance_degree", INITIAL_ACQUAINTANCE_DEGREE)
        )
        self.acquaintance_sets = build_ring_acquaintance_sets(
            self.num_banks,
            acq_rng,
            degree=self.initial_acquaintance_degree,
            skip_idx=0,
        )
        self.current_step = 0
        self.roles = (
            self.assign_roles_by_risk(
                car_cutoff=getattr(self, "car_cutoff", 0.08),
                lcr_cutoff=getattr(self, "lcr_cutoff", DAILY_INTERBANK_ROLE_LCR_CUTOFF),
            )
            if hasattr(self, 'assign_roles_by_risk') else self.assign_roles()
        )

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
            b['lag_car'] = float(b.get('capital_adequacy_ratio', 0.0) or 0.0)
            b['lag_lcr'] = float(b.get('liquidity_coverage_ratio', 0.0) or 0.0)
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

        # 5. 清零自身对自身同业敞口，初始化违约状态
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
        self.rfq_history: list[dict] = []
        self.step_rfq_events: list[dict] = []
        self.collect_rfq_events = bool(getattr(self, "collect_rfq_events", False))
        self.rfq_teacher_mode = str(getattr(self, "rfq_teacher_mode", "") or "")
        self.counterparty_touch_step: dict[int, dict[int, int]] = {}
        self.relationship_history_window = int(
            getattr(self, "relationship_history_window", 10)
        )
        self.export_policy_logs = getattr(self, "export_policy_logs", True)
        self.last_policy_note = "policy_init"
        self.initial_state_export_prefix = "decentralized_central_policy"
        self.relationship_counts: dict[tuple[int, int], float] = {}
        # Formal DEN: relationship / risk markup off; local RFQ K=4, degree=12; GNN required.
        self.relationship_lending_enabled = bool(
            getattr(self, "relationship_lending_enabled", False)
        )
        self.require_gnn = bool(getattr(self, "require_gnn", True))
        self.allow_rate_only_fallback = bool(
            getattr(self, "allow_rate_only_fallback", False)
        )
        self.rfq_k = int(getattr(self, "rfq_k", DAILY_RFQ_K))
        self.rfq_max_rounds = int(getattr(self, "rfq_max_rounds", DAILY_RFQ_MAX_ROUNDS))
        self.local_unique_lender_budget = int(
            getattr(self, "local_unique_lender_budget", LOCAL_UNIQUE_LENDER_BUDGET)
        )
        self.rfq_borrower_risk_markup = float(getattr(self, "rfq_borrower_risk_markup", 0.0))
        self.rfq_min_trade_size = float(
            getattr(self, "rfq_min_trade_size", DAILY_RFQ_MIN_TRADE_SIZE)
        )
        self.relationship_markup_discount = float(
            getattr(self, "relationship_markup_discount", RELATIONSHIP_MARKUP_DISCOUNT)
        )
        self.relationship_size_boost = float(
            getattr(self, "relationship_size_boost", RELATIONSHIP_SIZE_BOOST)
        )
        self.relationship_strength_scale = float(
            getattr(self, "relationship_strength_scale", RELATIONSHIP_STRENGTH_SCALE)
        )
        self.rollover_blocked_borrowers: set[int] = set()
        self.rollover_active_borrowers: set[int] = set()
        self.rollover_borrow_policy = ROLLOVER_BORROW_COUPON_CLEARED
        self.rollover_coupon_cleared_borrowers: set[int] = set()
        self.rollover_coupon_due_borrowers: set[int] = set()
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
        # Opportunity-borrow discount φ and bank-level IBL capacity
        self.debt_burden_kappa = float(
            getattr(self, "debt_burden_kappa", DEFAULT_DEBT_BURDEN_KAPPA)
        )
        self.ibl_cap_asset_lambda = float(
            getattr(self, "ibl_cap_asset_lambda", DEFAULT_IBL_CAP_ASSET_LAMBDA)
        )
        # Pre-trade liquidity window off: liquidity support is EN settlement backstop only
        # Hard-disabled: liquidity support must settle next period, never pre-trade.
        self.policy_pre_trade_liquidity_window = False
        self._settlement_support_issued: dict[int, float] = {}
        self._settlement_support_la_pre: dict[int, float] = {}
        self._settlement_support_due: dict[int, float] = {}
        self.last_defaulted_banks: list[int] = []
        self.pending_policy_support: list[dict] = []
        self.interbank_lgd = float(getattr(self, "interbank_lgd", 0.4))
        self.rfq_max_rounds = int(getattr(self, "rfq_max_rounds", DAILY_RFQ_MAX_ROUNDS))
        self.step_metrics_history: list[dict] = []
        self.interbank_lcr_target = float(
            getattr(self, "interbank_lcr_target", DAILY_INTERBANK_INTENTION_LCR_TARGET)
        )
        self.central_corridor = CentralBankCorridor(
            deposit_rate=max(0.0, self.base_rate - DAILY_CB_DEPOSIT_SPREAD),
            lending_rate=self.base_rate + DAILY_CB_LENDING_SPREAD,
            base_rate=self.base_rate,
        )

        # 初始化不预填同业合约；首次成交由 simulate_step(0) 的 RFQ 产生
        export_initial_bank_table(
            self.banks,
            INITIAL_STATE_DIR,
            self.initial_state_export_prefix,
        )
        if hasattr(self, "calculate_systemic_risk"):
            self._record_systemic_risk(self.calculate_systemic_risk())

    def _seed_contract_book_from_exposure(self, step0_maturity: int = 0):
        """用当前 exposure_matrix 填充 contract_book，供主循环首步到期结算。"""
        L = np.asarray(self.exposure_matrix, dtype=float)
        n = L.shape[0]
        r = float(getattr(self, "base_rate", DAILY_BULL_BASE_RATE))
        for i in range(n):
            for j in range(n):
                if i == j or L[i, j] <= 1e-12:
                    continue
                c = Contract(
                    contract_id=self.contract_book._new_id(),
                    lender_idx=i,
                    borrower_idx=j,
                    principal=float(L[i, j]),
                    rate=r,
                    created_step=0,
                    maturity_step=int(step0_maturity),
                    settlement_rate=r,
                )
                self.contract_book.add_contract(c)

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

    def _register_trade_contracts(self, trades: list, step: int) -> None:
        """成交入账：OFF→single_payment；ON→成交即 installment。"""
        from bank_econ_shared import borrower_origination_risk_q

        book = self.contract_book
        banks = self.banks
        cfg = self._schedule_cfg()
        log = getattr(self, "trade_schedule_log", None)
        blocked = getattr(self, "rollover_blocked_borrowers", set())
        policy = getattr(self, "rollover_borrow_policy", ROLLOVER_BORROW_COUPON_CLEARED)

        borrower_q_at_origination = {}
        for t in trades:
            j = int(t.borrower_idx)
            if j not in borrower_q_at_origination:
                borrower_q_at_origination[j] = float(
                    borrower_origination_risk_q(
                        banks[j],
                        system=self,
                        bank_idx=j,
                        car_threshold=float(
                            getattr(self, "car_cutoff", 0.08)
                        ),
                    )
                )

        for t in trades:
            if t.borrower_idx in blocked and str(policy).lower() == ROLLOVER_BORROW_BLOCK_ALL:
                continue
            _, dec = book.add_from_trade_with_schedule(t, banks[t.borrower_idx], cfg)
            record_relationship_trade(
                self, int(t.lender_idx), int(t.borrower_idx), float(t.amount)
            )
            touch_counterparty(self, int(t.lender_idx), int(t.borrower_idx), int(step))
            if log is not None:
                log.append({
                    "step": int(step),
                    "lender": int(t.lender_idx),
                    "borrower": int(t.borrower_idx),
                    "amount": float(t.amount),
                    "trade_rate": float(t.rate),
                    "schedule_type": dec["schedule_type"],
                    "reason": dec.get("reason", ""),
                    "tenor": dec.get("tenor"),
                    "coupon_rate": dec.get("coupon_rate"),
                    "settlement_rate": dec.get("settlement_rate"),
                    "maturity_in_periods": dec.get("maturity_in_periods"),
                })
            banks[t.lender_idx]["liquid_assets"] = float(
                banks[t.lender_idx].get("liquid_assets", 0.0)
            ) - t.amount
            banks[t.borrower_idx]["liquid_assets"] = float(
                banks[t.borrower_idx].get("liquid_assets", 0.0)
            ) + t.amount
            self.borrowed_cash[t.borrower_idx] += t.amount
            self.borrowed_origination_risk_sum[t.borrower_idx] += (
                float(t.amount)
                * float(borrower_q_at_origination[int(t.borrower_idx)])
            )
            from bank_econ_shared import accrue_funded_trade
            accrue_funded_trade(
                self,
                float(t.amount),
                float(borrower_q_at_origination[int(t.borrower_idx)]),
            )

    def _settle_interbank_installment_period(self, step: int) -> list[int]:
        """分期/单期：全市场 due_flows 合并后一次 EN；未付规则见 interbank_installment_rollover。"""
        book = self.contract_book
        n = self.num_banks
        corridor = getattr(self, "central_corridor", None) or CentralBankCorridor(
            deposit_rate=max(0.0, self.base_rate - DAILY_CB_DEPOSIT_SPREAD),
            lending_rate=self.base_rate + DAILY_CB_LENDING_SPREAD,
            base_rate=self.base_rate,
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
            book,
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
        self.rollover_active_borrowers = active_installment_rollover_borrowers(book, step)
        # Active installment borrowers are passed to the borrowing policy.
        # coupon_cleared blocks only those that did not clear today's coupon;
        # it does not turn arrears into a default.
        self.rollover_blocked_borrowers = (
            set(self.rollover_active_borrowers)
            if bool(getattr(self, "rollover_enabled", False))
            else set()
        )
        self.last_en_unpaid = float(getattr(settle_result, "unpaid_amount", 0.0))
        self.last_defaulted_banks = [int(i) for i in settle_result.failed if 0 < int(i) < len(self.banks)]
        return settle_result.failed

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
        if hasattr(self, "exposure_matrix"):
            self.exposure_matrix = aggregate_contracts_to_exposure_matrix_at_step(
                book, n, int(step)
            )
            np.fill_diagonal(self.exposure_matrix, 0.0)
            # Keep tombstone nodes off the live network
            for i, b in enumerate(banks):
                if b.get("absorbing_default") or b.get("balance_sheet_frozen"):
                    self.exposure_matrix[i, :] = 0.0
                    self.exposure_matrix[:, i] = 0.0
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
        # Formal model: never inject investable liquidity before trade / EN.
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
        # Excel export deferred to end of simulate_step (after disbursed fields fill).


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
        After EN settlement: refresh books, queue *liquidity* support for t+1 only.
        Capital (solvency) support is applied once after projects, before equity cascade.
        """
        queued = {"liquidity": 0.0, "solvency": 0.0, "n_liq": 0, "n_sol": 0}
        if not getattr(self, "policy_support_enabled", True):
            return queued
        if not getattr(self, "policy_enabled", True):
            return queued

        pending = list(getattr(self, "pending_policy_support", []) or [])
        pending = [x for x in pending if int(x.get("queued_step", -1)) != int(step)]
        # Drop any legacy deferred solvency entries.
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
            # Use accounting equity here so a mildly negative bank receives
            # enough to cross both zero equity and the target capital level.
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

        # 注意：以下会修改银行状态；若本函数在同一 step 内被对同一 i 多次调用，负债会被重复放大
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
                  float(b.get('liquid_assets', 0.0)) / (float(b.get('current_liabilities', 0.0)) * float(b.get('outflow_rate', 0.4)) + 1e-9))
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
        4) 刷新账面，排队下期流动性支持
        5) 市场/项目/冲击
        6) 每日撮合（DEN：局部 GNN-RFQ）
        7) 再刷新监管指标
        8) 最终违约判定（LGD 核销）
        9) 计算 SR_t 供下一期政策使用
        """
        try:
            step = int(step)
            self.current_step = step
            self.last_interbank_writeoff = 0.0
            self.last_estate_transfer_discount = 0.0
            from bank_econ_shared import reset_screening_loss
            reset_screening_loss(self, totals=False, step=True)
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
            n = self.num_banks
            book = self.contract_book
            banks = self.banks
            if getattr(self, "exposure_matrix", None) is not None:
                self.prev_exposure_matrix = self.exposure_matrix.copy()

            # 3) all due interbank claims → one EN matrix → one clearing (no pre-pay)
            # Capital support is deferred until after projects (not pre-EN).
            self._refresh_regulatory_metrics(step)
            failed = self._settle_interbank_installment_period(step)
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

            for _b in banks:
                if not _b.get("is_active", True):
                    _b.pop("pending_endowment", None)
                    continue
                if "pending_endowment" in _b:
                    _b["liquid_assets"] += _b.pop("pending_endowment")

            self.market_duration += 1
            if self.market_duration >= self.market_duration_limit:
                self.prev_market_environment = self.market_environment
                self.market_environment = "bull" if self.rng_environment.random() < 0.6 else "bear"
                self.market_duration = 0
                self.market_duration_limit = int(self.rng_environment.integers(2, (5) + 1))

            if self.market_environment == "bull":
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
            loan_spreads = self.rng_environment.uniform(loan_lo, loan_hi, size=n)
            inv_spreads = self.rng_environment.uniform(
                DAILY_INVESTMENT_SPREAD_STEP[0], DAILY_INVESTMENT_SPREAD_STEP[1], size=n
            )
            outflow_draws = self.rng_environment.uniform(out_lo, out_hi, size=n)

            for i, bank in enumerate(banks):
                if not bank.get("is_active", True):
                    continue
                bank["market_volatility"] = market_volatility
                bank["loan_interest_rate"] = self.base_rate + float(loan_spreads[i])
                bank["investment_interest_rate"] = self.long_term_rate + float(inv_spreads[i])
                bank.setdefault("risk_appetite", 0.5)
                bank.setdefault("hurdle_rate", DAILY_HURDLE_RATE)
                bank.setdefault("pending_endowment", 0.0)
                apply_deposit_flow(bank, DAILY_LIABILITY_GROWTH)
                adj = self._adjust_market_liquidity_shock(i, market_adjustment)
                apply_deposit_flow(bank, adj)
                bank["outflow_rate"] = (
                    0.2 if bank["type"] == "central" else float(outflow_draws[i])
                )
                if bank["liquid_assets"] < bank["current_liabilities"] * bank["outflow_rate"]:
                    bank["risk_appetite"] *= 0.9

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
                    banks,
                    multiplier=DAILY_COMMON_LIQUIDITY_OUTFLOW_MULTIPLIER,
                )
                # Fixed N-slot draws: failed banks keep their RNG slot.
                sensitivities = self.rng_environment.uniform(0.8, 1.2, size=n)
                for i, bank in enumerate(banks):
                    if not bank.get("is_active", True):
                        continue
                    bank["outflow_rate"] = min(
                        0.95,
                        float(bank.get("outflow_rate", 0.4))
                        * DAILY_COMMON_LIQUIDITY_OUTFLOW_MULTIPLIER
                        * float(sensitivities[i]),
                    )
                # Credit stress persists a few days so defaults arrive as a
                # staircase, not a same-day cash haircut.
                self.project_pd_stress_multiplier = DAILY_COMMON_PROJECT_PD_MULTIPLIER
                self.project_pd_stress_remaining = int(DAILY_COMMON_PROJECT_PD_STRESS_DAYS)
                self.last_common_liquidity_shock = True
            if self.market_environment == "bear" and self.rng_environment.random() < 0.1:
                for bank in banks:
                    if not bank.get("is_active", True):
                        continue
                    # Flight-to-deposit inflow: ΔA=ΔL, so it cannot lift CAR
                    # by gifting cash into core capital.
                    apply_deposit_flow(bank, 0.01)
            if (not getattr(self, "one_shot_default_done", False)) and (int(step) == 0):
                if self.rng_environment.random() < 0.03:
                    fail_bank = int(self.rng_environment.integers(1, (n - 1) + 1))
                    self._resolve_bank_default(fail_bank, step, reason="one_shot")
                self.one_shot_default_done = True

            # Refresh accounting after shocks/liability growth before matching.
            self._refresh_regulatory_metrics(step)

            # 6) daily matching
            if getattr(self, "free_market", False):
                self.roles = self.assign_roles_balanced(frac_lenders=0.5)
            else:
                self.roles = (
                    self.assign_roles_by_risk(
                        car_cutoff=getattr(self, "car_cutoff", 0.08),
                        lcr_cutoff=getattr(self, "lcr_cutoff", DAILY_INTERBANK_ROLE_LCR_CUTOFF),
                    )
                    if hasattr(self, "assign_roles_by_risk")
                    else self.assign_roles()
                )

            intentions = collect_intentions(
                banks, n, self.roles, self.reserve_buffer,
                self.base_rate,
                lcr_target=float(getattr(self, "interbank_lcr_target", DAILY_INTERBANK_INTENTION_LCR_TARGET)),
                step=int(step),
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
            intentions = self._filter_intentions_for_matching(intentions)
            rfq = getattr(self, "rfq_market", None) or RFQMarket(
                max_rounds=int(getattr(self, "rfq_max_rounds", DAILY_RFQ_MAX_ROUNDS)),
                min_trade_size=float(getattr(self, "rfq_min_trade_size", DAILY_RFQ_MIN_TRADE_SIZE)),
            )
            trades = rfq.run(
                intentions,
                banks,
                system=self,
                step=step,
                B_max=float(getattr(self, "B", DEFAULT_MATCH_B)),
                K=int(getattr(self, "rfq_k", DAILY_RFQ_K)),
                borrower_risk_markup_spread=float(
                    getattr(self, "rfq_borrower_risk_markup", 0.0)
                ),
            )
            lenders_int = [x for x in intentions if x.role == "lender"]
            borrowers_int = [x for x in intentions if x.role == "borrower"]
            total_supply = float(sum(x.quantity for x in lenders_int))
            total_demand = float(sum(x.quantity for x in borrowers_int))
            actual_lent = float(sum(t.amount for t in trades))
            queried = getattr(self, "rfq_unique_lenders_queried", {}) or {}
            rollover_map = getattr(self, "rfq_local_rollover", {}) or {}
            unique_n = [len(s) for s in queried.values()] if queried else []
            discovered_n = []
            for j, names in queried.items():
                roll = {int(x) for x in (rollover_map.get(int(j), []) or [])}
                discovered_n.append(len({int(x) for x in names} - roll))
            local_capacity_summary = _rfq_local_capacity_summary(self)
            if not hasattr(self, "rfq_history") or self.rfq_history is None:
                self.rfq_history = []
            self.rfq_history.append({
                "step": step,
                "total_supply": total_supply,
                "total_demand": total_demand,
                "actual_lent": actual_lent,
                "funding_ratio": actual_lent / (total_demand + 1e-9),
                "unmet_demand": max(0.0, total_demand - actual_lent),
                "num_trades": len(trades),
                "avg_rate": float(np.mean([t.rate for t in trades])) if trades else np.nan,
                "rfq_reject_counts": dict(getattr(self, "rfq_reject_counts", {}) or {}),
                "rfq_max_unique_lenders_queried": int(max(unique_n) if unique_n else 0),
                "rfq_max_unique_discovered_queried": int(max(discovered_n) if discovered_n else 0),
                "rfq_local_capacity": dict(local_capacity_summary),
            })
            self.last_match_stats = {
                "total_volume": float(actual_lent),
                "num_trades": int(len(trades)),
                "total_demand": float(total_demand),
                "total_supply": float(total_supply),
                "unmet_demand_rate": float(max(0.0, total_demand - actual_lent) / (total_demand + 1e-9)),
                "rfq_reject_counts": dict(getattr(self, "rfq_reject_counts", {}) or {}),
                "rfq_max_unique_lenders_queried": int(max(unique_n) if unique_n else 0),
                "rfq_max_unique_discovered_queried": int(max(discovered_n) if discovered_n else 0),
                "rfq_local_capacity": dict(local_capacity_summary),
            }
            from bank_econ_shared import note_screening_loss
            note_screening_loss(self, "funding_gap", max(0.0, total_demand - actual_lent))
            if trades:
                total_amt = sum(t.amount for t in trades)
                if total_amt > 1e-9:
                    self.last_avg_rate = sum(t.amount * t.rate for t in trades) / total_amt
                else:
                    self.last_avg_rate = sum(t.rate for t in trades) / len(trades)
            else:
                self.last_avg_rate = None
            self._register_trade_contracts(trades, step)
            self.exposure_matrix = aggregate_contracts_to_exposure_matrix_at_step(book, n, step)
            np.fill_diagonal(self.exposure_matrix, 0.0)

            for i in range(n):
                if self.roles[i] == +1 and banks[i].get("is_active", True):
                    frac = self._lender_invest_frac(i, 0.05)
                    if frac > 1e-6:
                        self.invest_free_cash_into_projects(i, invest_frac=frac)

            for i in range(n):
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
            for b in banks:
                if b.get("absorbing_default") or b.get("balance_sheet_frozen"):
                    continue
                b["capital_ratio_history"].append(float(b.get("solvency_ratio", 0.0)))
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
            risk = self.calculate_systemic_risk() if hasattr(self, "calculate_systemic_risk") else 0.0
            self._record_systemic_risk(risk)
            if getattr(self, "record_history", False):
                self.simulation_history.append({
                    "step": step,
                    "systemic_risk": risk,
                    "raw_systemic_risk": float(getattr(self, "last_raw_systemic_risk", risk)),
                    "collapse_index": float(getattr(self, "last_collapse_index", risk)),
                    "policy_note": getattr(self, "last_policy_note", ""),
                    "exposure_matrix": self.exposure_matrix.copy(),
                    "bank_states": [deepcopy(b) for b in banks],
                    **self._history_extra_fields(),
                })
            if self.all_default_step is None:
                alive_noncentral = [k for k in range(1, n) if banks[k].get("is_active", True)]
                if len(alive_noncentral) == 0:
                    self.all_default_step = int(step)
                    print(f"[ALL DEFAULT] step={self.all_default_step} (all non-central banks defaulted)")
            self._update_network_stability(step, risk)
            self.maybe_save_network_snapshot(step, risk, tag="rfq", edge_quantile=0.0)
            # Freeze CAR/LCR for next-step public grades / teacher (no look-ahead).
            for b in banks:
                b["lag_car"] = float(b.get("capital_adequacy_ratio", 0.0) or 0.0)
                b["lag_lcr"] = float(b.get("liquidity_coverage_ratio", 0.0) or 0.0)
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

    def simulate_step_decentralized(self, step: int):
        """已废弃：请使用 simulate_step()。"""
        import warnings

        warnings.warn(
            "simulate_step_decentralized() 已废弃，请改用 simulate_step()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.simulate_step(step)

    def _deterministic_rate_based_matching(
        self, lenders, borrowers, supply, demand, B, deg_init=None
    ):
        """
        利率优先的撮合（利率優先マッチング）。
        """
        n = self.num_banks
        eps = 1e-8

        supply = np.array(supply, dtype=float).copy()
        demand = np.array(demand, dtype=float).copy()

        if deg_init is None:
            deg = np.zeros(n, dtype=int)
        else:
            deg = np.array(deg_init, dtype=int).copy()

        pairs = []
        for li, i in enumerate(lenders):
            rate_i = float(self.banks[i].get("loan_interest_rate", DAILY_BULL_BASE_RATE))
            for bj, j in enumerate(borrowers):
                if i == j:
                    continue
                pairs.append((rate_i, i, j, li, bj))

        pairs.sort(key=lambda x: x[0])



        # ===== all-or-nothing borrower deterministic matching =====
        # 按 borrower 聚合候选 lenders（利率从低到高）
        pairs_by_b = {bj: [] for bj in range(len(borrowers))}
        for rate, i, j, li, bj in pairs:
            pairs_by_b[bj].append((rate, i, li))

        plan = []
        for bj, j in enumerate(borrowers):
            need = float(demand[bj])
            if need <= eps:
                continue

            if deg[j] >= self.max_degree:
                continue

            cand = pairs_by_b.get(bj, [])
            cand.sort(key=lambda x: x[0])  # 利率低优先

            remaining = need
            tmp_alloc = []  # (i, j, amt, li)

            for _, i, li in cand:
                if supply[li] <= eps:
                    continue
                if deg[i] >= self.max_degree:
                    continue
                if deg[j] + len(tmp_alloc) >= self.max_degree:
                    break

                amt = min(float(B), float(supply[li]), remaining)
                if amt <= eps:
                    continue

                tmp_alloc.append((i, j, float(amt), li))
                remaining -= amt

                if remaining <= eps:
                    break

            # 满额才提交；否则该 borrower 不成交（z_j=0）
            if remaining <= eps:
                for i, j2, amt, li in tmp_alloc:
                    plan.append((i, j2, float(amt)))
                    supply[li] -= amt
                    deg[i] += 1

                deg[j] += len(tmp_alloc)
                demand[bj] = 0.0
            else:
                continue

        return plan


def _gnn_make_plan_from_intentions(self, intentions: list[Intention], step: int):
    """
    方案A：去中心化撮合直接用论文Step3口径：
    feasible: r_min[i] <= r_max[j]
    surplus:  r_max[j] - r_min[i]
    weight:   w_ij = a_ij * surplus
    all-or-nothing borrower：凑不满需求则该borrower整单取消（z_j=0）
    """
    step = int(step)
    n = int(self.num_banks)
    eps = 1e-8
    B_base = float(getattr(self, "B", DEFAULT_MATCH_B))  # 单笔上限，和你RFQMarket run里用的B_max一致

    lenders, borrowers, supply, demand, r_min, r_max = _intentions_to_arrays(intentions)

    if (len(lenders) == 0) or (len(borrowers) == 0):
        return [], r_min, r_max

    # 初始度数（本期内控制 max_degree）
    deg = np.zeros(n, dtype=int)

    # --- build feasible pairs ---
    feasible_pairs = []
    surplus = []
    for i in lenders:
        for j in borrowers:
            if i == j:
                continue
            if float(r_min[i]) <= float(r_max[j]):
                feasible_pairs.append((int(i), int(j)))
                surplus.append(float(r_max[j]) - float(r_min[i]))

    if not feasible_pairs:
        return [], r_min, r_max

    # --- compute a_ij from matcher ---
    ctx = getattr(self, "gnn_context", None)
    matcher = None if ctx is None else ctx.get("matcher", None)
    device = None if ctx is None else ctx.get("device", None)

    if matcher is None:
        allow_fallback = bool(getattr(self, "allow_rate_only_fallback", False))
        require_gnn = bool(getattr(self, "require_gnn", True))
        if require_gnn or not allow_fallback:
            raise RuntimeError(
                "DEN matching requires a GNN matcher; rate-only is disabled for formal "
                "experiments. Set allow_rate_only_fallback=True only for debugging."
            )
        # fallback：deterministic matching（调试用）
        plan = self._deterministic_rate_based_matching(
            lenders=lenders,
            borrowers=borrowers,
            supply=supply,
            demand=demand,
            B=B_base,
            deg_init=deg,
        )
        return plan, r_min, r_max

    # 每家 borrower 在局部子图上打分（禁止全市场 to_pyg_graph）
    aij = np.zeros(len(feasible_pairs), dtype=float)
    K_local = int(getattr(self, "rfq_k", DAILY_RFQ_K))
    pairs_by_borrower: dict[int, list[int]] = {}
    for k, (i, j) in enumerate(feasible_pairs):
        pairs_by_borrower.setdefault(int(j), []).append(int(k))

    local_ok: dict[int, set[int]] = {}
    for j, ks in pairs_by_borrower.items():
        lender_opts = [int(feasible_pairs[k][0]) for k in ks]
        pool, hist, roll = build_local_lender_pool(lender_opts, int(j), self)
        local_ok[int(j)] = {int(x) for x in pool}
        cand, hist = discover_local_rfq_candidates(
            lender_opts, int(j), K_local, self, pool=pool, rollover=roll,
        )
        if not cand:
            continue
        cand_set = set(cand)
        local_graph, id_map = build_borrower_local_graph(
            self, int(j), cand, hist, use_prev=True, mask_private=True
        )
        scored_pairs = []
        scored_ks = []
        pair_feat_rows = []
        for k in ks:
            i = int(feasible_pairs[k][0])
            if i not in cand_set or i not in id_map or int(j) not in id_map:
                continue
            scored_pairs.append((int(id_map[i]), int(id_map[j])))
            scored_ks.append(k)
            pair_feat_rows.append([
                pair_bilateral_exposure_ratio(self, self.banks, int(i), int(j), proposed_amt=0.0)
            ])
        if not scored_pairs:
            continue
        scores = matcher.score_pairs(
            local_graph, scored_pairs, device=device, pair_features=pair_feat_rows
        )
        for t, k in enumerate(scored_ks):
            aij[k] = float(scores[t])

    surplus = np.asarray(surplus, dtype=float)
    w = aij * surplus  # 论文对齐：w_ij = a_ij * surplus

    idx_L = {i: k for k, i in enumerate(lenders)}
    idx_B = {j: k for k, j in enumerate(borrowers)}
    supply_left = np.array(supply, dtype=float).copy()
    demand_left = np.array(demand, dtype=float).copy()

    # 1) feasible pair 按 borrower 聚合：bj -> list[(score, lender_id)]
    pairs_by_b = {idx_B[j]: [] for j in borrowers}
    for k, (i, j) in enumerate(feasible_pairs):
        if int(i) not in local_ok.get(int(j), set()):
            continue
        bj = idx_B[j]
        pairs_by_b[bj].append((float(w[k]), int(i)))

    # 2) borrower 优先级：先处理“最有希望被凑满”的 borrower
    def borrower_priority(j_id: int) -> float:
        bj = idx_B[j_id]
        if not pairs_by_b[bj]:
            return -1e18
        return max(sc for sc, _ in pairs_by_b[bj])

    borrower_order = sorted(borrowers, key=borrower_priority, reverse=True)

    plan = []
    for j in borrower_order:
        bj = idx_B[j]
        need = float(demand_left[bj])
        if need <= eps:
            continue

        if deg[j] >= int(self.max_degree):
            continue

        cand = pairs_by_b[bj]
        cand.sort(key=lambda x: x[0], reverse=True)  # w_ij 降序

        remaining = need
        tmp_alloc = []  # (i, j, amt, li)

        for _, i in cand:
            li = idx_L[i]

            if supply_left[li] <= eps:
                continue
            if deg[i] >= int(self.max_degree):
                continue
            if deg[j] + len(tmp_alloc) >= int(self.max_degree):
                break

            amt = min(B_base, float(supply_left[li]), remaining)
            if amt <= eps:
                continue

            tmp_alloc.append((i, j, float(amt), li))
            remaining -= amt
            if remaining <= eps:
                break

        # 3) 满额才提交；否则 z_j=0（整单取消）
        if remaining <= eps:
            for i, j2, amt, li in tmp_alloc:
                plan.append((i, j2, float(amt)))
                supply_left[li] -= amt
                deg[i] += 1
            deg[j] += len(tmp_alloc)
            demand_left[bj] = 0.0

    return plan, r_min, r_max


def _sparse_bipartite_update(self, roles: np.ndarray) -> None:
    """
    重新生成稀疏双边同业网络：
    - 去掉 rollover：每期不保留存量网络
    - 每期 exposure_matrix 清零，只由当期撮合生成
    """
    B_base = float(getattr(self, "B", DEFAULT_MATCH_B))
    n = self.num_banks
    eps = 1e-8

    # ===== 去掉 rollover：每期清零网络（no carry-over）=====
    self.exposure_matrix = np.zeros((n, n), dtype=float)
    np.fill_diagonal(self.exposure_matrix, 0.0)

    # 既然不保留存量网络，初始度数全为 0
    deg_init = np.zeros(n, dtype=int)

    lenders = [i for i in range(n) if roles[i] == +1 and self.banks[i].get("is_active", True)]
    borrowers = [i for i in range(n) if roles[i] == -1 and self.banks[i].get("is_active", True)]
    step_match = int(getattr(self, "current_step", 0))
    blocked_rb = getattr(self, "rollover_blocked_borrowers", None)
    borrow_policy = getattr(self, "rollover_borrow_policy", ROLLOVER_BORROW_COUPON_CLEARED)
    borrowers, _ = filter_borrowers_for_rollover_block(
        borrowers,
        self.contract_book,
        step_match,
        precomputed_blocked=blocked_rb,
        borrow_policy=borrow_policy,
    )

    if len(lenders) == 0 or len(borrowers) == 0:
        print(f"[debug] No matching: lenders={len(lenders)}, borrowers={len(borrowers)}")
        return
    # ===== 诊断：角色分布 =====
    print(f"[diag] lenders={len(lenders)} borrowers={len(borrowers)}")
    # ===== 1) 供给 side =====
    supply = []
    LCR_TARGET = 1.0
    ALPHA_STRESS_LENDER = 1.0

    # 去掉“基于存量负债的 blocked”（因为你不保留存量网络了）
    for i in lenders:
        liq = float(self.banks[i]["liquid_assets"])
        lia = float(self.banks[i]["current_liabilities"])
        outflow_target = lia * float(self.banks[i].get("outflow_rate", 0.4))

        req = float(self.reserve_buffer[i] * lia)
        lcr_buffer = LCR_TARGET * ALPHA_STRESS_LENDER * outflow_target
        target_liq_lender = max(req, lcr_buffer)

        avail = max(0.0, liq - target_liq_lender)

        phi = float(self.banks[i].get("risk_appetite", 0.5))
        avail *= (0.6 + 0.4 * phi)

        supply.append(float(avail))

    # ===== 2) 需求 side =====
    demand_raw = []
    ALPHA_STRESS_BORROWER = 1.0
    K_EXPAND_BULL = 0.10
    K_EXPAND_BEAR = 0.06

    for j in borrowers:
        liq = float(self.banks[j]["liquid_assets"])
        lia = float(self.banks[j]["current_liabilities"])

        req = float(self.reserve_buffer[j] * lia)
        outflow_target = lia * float(self.banks[j].get("outflow_rate", 0.4))
        target_liq = max(req, LCR_TARGET * ALPHA_STRESS_BORROWER * outflow_target)

        # (a) 缺口需求
        gap_j = max(0.0, target_liq - liq)

        # (b) 扩张需求
        phi = float(self.banks[j].get("risk_appetite", 0.5))
        exp_proj = float(self.banks[j].get("investment_interest_rate", self.long_term_rate))
        loan_rt  = float(self.banks[j].get("loan_interest_rate", self.base_rate))
        spread_pos = max(0.0, exp_proj - loan_rt)

        K = K_EXPAND_BEAR if self.market_environment == "bear" else K_EXPAND_BULL
        extra_need = K * lia * phi * (spread_pos / (loan_rt + 1e-9))

        phi_b = float(self.banks[j].get("risk_appetite", 0.5))
        need_liq_j = min(gap_j + 0.08 * lia * phi_b, 0.5 * lia)
        from interbank_installment_rollover import compute_project_investment_borrow_cap

        need_inv_j = compute_project_investment_borrow_cap(
            self.banks[j],
            float(self.base_rate),
            last_avg_rate=getattr(self, "last_avg_rate", None),
        )
        need_raw = rollover_borrow_quantity(
            j,
            self.banks[j],
            float(self.base_rate),
            need_liq_j,
            need_inv_j,
            rollover_blocked=blocked_rb,
            coupon_cleared=getattr(self, "rollover_coupon_cleared_borrowers", None),
            coupon_due=getattr(self, "rollover_coupon_due_borrowers", None),
            borrow_policy=borrow_policy,
            last_avg_rate=getattr(self, "last_avg_rate", None),
        )

        # ===== borrower-specific cap =====
        size_cap_j = 0.5 * lia
        B_j = min(
            B_base,
            size_cap_j,
            max(300.0, 0.5 * gap_j)
        )

        need = min(need_raw, self.max_degree * B_j)

        # 保存 cap（debug/后用）
        self.banks[j].setdefault("_B_cap", B_j)

        demand_raw.append(float(need))

    # ===== totals: 循环结束后统一计算（关键）=====
    total_supply = float(np.sum(np.asarray(supply, dtype=float))) if len(supply) else 0.0
    total_demand = float(np.sum(np.asarray(demand_raw, dtype=float))) if len(demand_raw) else 0.0

    # ===== scale: 只算一次，再统一缩放 demand =====
    if total_demand > eps:
        scale = min(1.0, (total_supply + 1e-9) / (total_demand + 1e-9))
        scale = max(scale, 0.05) 
    else:
        scale = 0.0

    demand = [d * scale for d in demand_raw]
    # ===== DIAG: liquidity gap check (why borrowers few / market idle) =====
    liq_arr = np.array([float(self.banks[k]["liquid_assets"]) for k in range(n)], dtype=float)
    lia_arr = np.array([float(self.banks[k]["current_liabilities"]) for k in range(n)], dtype=float)
    out_arr = np.array([float(self.banks[k].get("outflow_rate", 0.4)) for k in range(n)], dtype=float)
    res_arr = np.array([float(self.reserve_buffer[k]) for k in range(n)], dtype=float)

    req_arr = res_arr * lia_arr
    target_arr = np.maximum(req_arr, LCR_TARGET * ALPHA_STRESS_BORROWER * (lia_arr * out_arr))
    gap_arr = np.maximum(0.0, target_arr - liq_arr)

    active_mask = np.array([bool(self.banks[k].get("is_active", True)) for k in range(n)])
    gap_active = gap_arr[active_mask]
    print(
        f"[diag-gap] active_gap>0={int((gap_active > 1e-6).sum())}/{int(active_mask.sum())} | "
        f"gap min/mean/max={gap_active.min():.2f}/{gap_active.mean():.2f}/{gap_active.max():.2f}"
    )

    # 进一步把“roles里的borrower”分解成：缺口需求 vs 扩张需求
    need_gap_list = []
    extra_need_list = []
    for j in borrowers:
        liq = float(self.banks[j]["liquid_assets"])
        lia = float(self.banks[j]["current_liabilities"])
        req = float(self.reserve_buffer[j] * lia)
        outflow_target = lia * float(self.banks[j].get("outflow_rate", 0.4))
        target_liq = max(req, LCR_TARGET * ALPHA_STRESS_BORROWER * outflow_target)
        need_gap_list.append(max(0.0, target_liq - liq))

        phi = float(self.banks[j].get("risk_appetite", 0.5))
        exp_proj = float(self.banks[j].get("investment_interest_rate", self.long_term_rate))
        loan_rt  = float(self.banks[j].get("loan_interest_rate", self.base_rate))
        spread_pos = max(0.0, exp_proj - loan_rt)
        K = K_EXPAND_BEAR if self.market_environment == "bear" else K_EXPAND_BULL
        extra_need_list.append(K * lia * phi * (spread_pos / (loan_rt + 1e-9)))

    if len(borrowers) > 0:
        print(
            f"[diag-need] borrowers={len(borrowers)} | "
            f"gap(min/mean/max)={np.min(need_gap_list):.2f}/{np.mean(need_gap_list):.2f}/{np.max(need_gap_list):.2f} | "
            f"extra(min/mean/max)={np.min(extra_need_list):.2f}/{np.mean(extra_need_list):.2f}/{np.max(extra_need_list):.2f}"
        )

    # ===== 3) 过滤无效 borrower（PATCH: 加 demand floor，避免被 eps 全过滤） =====
    borrowers_eff, demand_eff = [], []

    MIN_DEMAND_ABS = 50.0      # 你可以调 50~200
    # 也可以做相对下限（可选）：MIN_DEMAND_REL = 0.002  # 0.2% liabilities
    # 一个安全的 floor：不超过“平均供给的一半”，防止 floor 过大把供给吃爆
    avg_supply_per_b = (total_supply / max(1, len(borrowers))) if len(borrowers) else 0.0
    min_floor = float(min(MIN_DEMAND_ABS, 0.5 * avg_supply_per_b))

    for j, d in zip(borrowers, demand):
        d = float(d)
        # 如果是 borrower 但 demand 很小，就给一个 floor
        # 注意：这里不要用 eps 过滤掉，否则又回到你原来的问题
        if d <= eps:
            d = min_floor

        if d > 0.0:
            borrowers_eff.append(j)
            demand_eff.append(d)

    borrowers, demand = borrowers_eff, demand_eff

    # 如果仍然 0，基本就是 total_supply=0 或 borrowers 为空（结构性无交易）
    if len(borrowers) == 0:
        print("[debug] No effective demand (after floor). Market idle this step.")
        return

    print(f"[diag] supply min/mean/max = {np.min(supply):.2f}/{np.mean(supply):.2f}/{np.max(supply):.2f}")
    print(f"[diag] demand  min/mean/max = {np.min(demand):.2f}/{np.mean(demand):.2f}/{np.max(demand):.2f}")

    # 注意：这里建议用 “scaled 后的 demand” 来判定，而不是 total_demand(=raw demand)
    total_demand_eff = float(np.sum(np.asarray(demand, dtype=float)))
    if total_supply <= eps or total_demand_eff <= eps:
        print(f"[debug] No matching: total_supply={total_supply:.2f}, total_demand_eff={total_demand_eff:.2f}")
        return

    # ===== 4) 撮合方式：论文对齐版（GNN 生成 a_ij）=====

    ctx = getattr(self, "gnn_context", None)
    B_base = float(getattr(self, "B", DEFAULT_MATCH_B))

    if ctx is not None and ctx.get("matcher", None) is not None:
        matcher = ctx["matcher"]
        device  = ctx.get("device", "cpu")

        # 4.1 定义 reservation rates（你可按论文 Step2 改，这里给一个可运行的默认）
        # lender 的最低可接受利率
        r_min = {i: float(self.banks[i].get("loan_interest_rate", DAILY_BULL_BASE_RATE)) for i in lenders}
        # borrower 的最高可接受利率（默认：当前 loan_rate + 一个容忍利差）
        rmax_spread = float(ctx.get("rmax_spread", DAILY_RFQ_QUOTE_SPREAD))
        r_max = {j: float(self.banks[j].get("loan_interest_rate", DAILY_BULL_BASE_RATE)) + rmax_spread for j in borrowers}

        # 4.2 构造 feasible pairs & surplus
        feasible_pairs = []
        surplus = []
        for i in lenders:
            for j in borrowers:
                if i == j:
                    continue
                if deg_init[i] >= self.max_degree or deg_init[j] >= self.max_degree:
                    continue
                # feasibility: r_min <= r_max
                if r_min[i] <= r_max[j]:
                    feasible_pairs.append((i, j))
                    surplus.append(r_max[j] - r_min[i])  # s_ij >= 0

        if not feasible_pairs:
            plan = []
        else:
            # Local-subgraph scoring per borrower (no full-market graph).
            aij = np.zeros(len(feasible_pairs), dtype=float)
            K_local = int(getattr(self, "rfq_k", DAILY_RFQ_K))
            pairs_by_borrower: dict[int, list[int]] = {}
            for k, (i, j) in enumerate(feasible_pairs):
                pairs_by_borrower.setdefault(int(j), []).append(int(k))
            for j, ks in pairs_by_borrower.items():
                lender_opts = [int(feasible_pairs[k][0]) for k in ks]
                cand, hist = discover_local_rfq_candidates(lender_opts, int(j), K_local, self)
                if not cand:
                    continue
                cand_set = set(cand)
                local_nodes = {int(j)} | {int(x) for x in hist} | cand_set
                local_graph, id_map = self.to_local_pyg_graph(local_nodes, use_prev=True)
                scored_pairs, scored_ks, pair_feat_rows = [], [], []
                for k in ks:
                    i = int(feasible_pairs[k][0])
                    if i not in cand_set or i not in id_map or int(j) not in id_map:
                        continue
                    scored_pairs.append((int(id_map[i]), int(id_map[j])))
                    scored_ks.append(k)
                    pair_feat_rows.append([
                        pair_bilateral_exposure_ratio(
                            self, self.banks, int(i), int(j), proposed_amt=0.0
                        )
                    ])
                if not scored_pairs:
                    continue
                scores = matcher.score_pairs(
                    local_graph, scored_pairs, device=device, pair_features=pair_feat_rows
                )
                for t, k in enumerate(scored_ks):
                    aij[k] = float(scores[t])

            surplus = np.asarray(surplus, dtype=float)
            # 论文对齐：权重 = a_ij * surplus
            w = aij * surplus

            beta_amt = float(ctx.get("beta_amt", 0.05))

            idx_L = {i: k for k, i in enumerate(lenders)}
            idx_B = {j: k for k, j in enumerate(borrowers)}
            supply_left = np.array(supply, dtype=float).copy()
            demand_left = np.array(demand, dtype=float).copy()
            deg = np.array(deg_init, dtype=int).copy()

            # ===== all-or-nothing borrower matching (replace ranked greedy) =====

            # 1) 把 feasible pair 按 borrower 聚合：bj -> list[(score, lender_id)]
            pairs_by_b = {idx_B[j]: [] for j in borrowers}
            for k, (i, j) in enumerate(feasible_pairs):
                bj = idx_B[j]
                # 论文对齐：w = aij * surplus（你上面已算好）
                pairs_by_b[bj].append((float(w[k]), i))

            # 2) borrower 排序：优先处理“最有希望被凑满”的 borrower
            def borrower_priority(j_id: int) -> float:
                bj = idx_B[j_id]
                if not pairs_by_b[bj]:
                    return -1e18
                return max(sc for sc, _ in pairs_by_b[bj])

            borrower_order = sorted(borrowers, key=borrower_priority, reverse=True)

            plan = []
            for j in borrower_order:
                bj = idx_B[j]
                need = float(demand_left[bj])
                if need <= eps:
                    continue

                # borrower 已满度数则跳过
                if deg[j] >= self.max_degree:
                    continue

                cand = pairs_by_b[bj]
                cand.sort(key=lambda x: x[0], reverse=True)  # w_ij 降序

                remaining = need
                tmp_alloc = []  # (i, j, amt, li)

                for _, i in cand:
                    li = idx_L[i]

                    # lender 约束
                    if supply_left[li] <= eps:
                        continue
                    if deg[i] >= self.max_degree:
                        continue

                    # borrower 度数：该 borrower 可能需要多条边凑满
                    if deg[j] + len(tmp_alloc) >= self.max_degree:
                        break

                    amt = min(B_base, float(supply_left[li]), remaining)
                    if amt <= eps:
                        continue

                    tmp_alloc.append((i, j, float(amt), li))
                    remaining -= amt

                    if remaining <= eps:
                        break

                # 3) 满额才提交；否则 z_j=0（回滚，不成交）
                if remaining <= eps:
                    for i, j2, amt, li in tmp_alloc:
                        plan.append((i, j2, float(amt)))
                        supply_left[li] -= amt
                        deg[i] += 1

                    deg[j] += len(tmp_alloc)
                    demand_left[bj] = 0.0
                else:
                    continue

            # ===== end all-or-nothing borrower matching =====

    else:
        # fallback：原 deterministic
        plan = self._deterministic_rate_based_matching(
            lenders=lenders, borrowers=borrowers,
            supply=supply, demand=demand,
            B=B_base, deg_init=deg_init
        )

    # ===== 5) 落地 plan=====
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
    rollover_blocked = getattr(self, "rollover_blocked_borrowers", set())

    for lender_idx, borrower_idx, amount in plan:
        amt = float(amount)
        if amt <= eps:
            continue
        # rollover_blocked 仅用于 need_liq 反借新还旧；整笔禁借仅在 block_all
        if (
            rollover_blocked
            and borrower_idx in rollover_blocked
            and str(getattr(self, "rollover_borrow_policy", ROLLOVER_BORROW_COUPON_CLEARED)).lower()
            == ROLLOVER_BORROW_BLOCK_ALL
        ):
            continue

        if deg[lender_idx] >= self.max_degree or deg[borrower_idx] >= self.max_degree:
            continue

        # ===== all-or-nothing: 不满额的 borrower 直接整单取消 =====
        if borrower_idx not in good_borrowers:
            continue

        try:
            li = lenders.index(lender_idx)
        except ValueError:
            continue

        # borrower 剩余需求
        need_rem = remaining_need.get(borrower_idx, 0.0)
        if need_rem <= eps:
            continue

        # 这里不要再用 demand[bj] 作为 cap
        amt = min(amt, supply[li], need_rem)
        if amt <= eps:
            continue

        self.exposure_matrix[borrower_idx, lender_idx] -= amt
        self.exposure_matrix[lender_idx, borrower_idx] += amt

        self.banks[lender_idx]["liquid_assets"] -= amt
        self.banks[borrower_idx]["liquid_assets"] = float(
            self.banks[borrower_idx].get("liquid_assets", 0.0)
        ) + amt
        self.borrowed_cash[borrower_idx] += amt

        supply[li] -= amt
        remaining_need[borrower_idx] -= amt

        deg[lender_idx] += 1
        deg[borrower_idx] += 1

        edge_count += 1
        actual_lent += amt


    np.fill_diagonal(self.exposure_matrix, 0.0)

    print(
        f"[debug] Matching finished: edges={edge_count}, "
        f"lenders={len(lenders)}, borrowers={len(borrowers)}, "
        f"total_supply={total_supply:.2f}, total_demand={total_demand:.2f}, "
        f"actual_lent={actual_lent:.2f}, B_eff={float(getattr(self, 'B', DEFAULT_MATCH_B)):.2f}"
    )
    # ===== snapshot + plot (pre-clearing) =====
    A_new = self.exposure_matrix.copy()

    # step 变量：用你外部循环传进来的更好；临时没有就用 self.step / self.t / self.current_step 兜底
    step = int(getattr(self, "step", getattr(self, "t", getattr(self, "current_step", 0))))

    # 确保容器存在
    if not hasattr(self, "exposure_hist"):
        self.exposure_hist = {}
    self.exposure_hist[step] = A_new



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
    把当前状态转成 PyG Data（全图；仅用于 SR 可视化/旧工具，正式 DEN RFQ 不用）。
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


def to_local_pyg_graph(
    self,
    node_ids,
    use_prev: bool = True,
    observer_idx: int | None = None,
    mask_private: bool = True,
):
    """
    借款人有限信息局部子图：
    仅包含 ``node_ids`` 中的银行节点，以及这些节点之间可观察的历史正边。
    对非观察者节点遮蔽私人资产负债表特征。
    返回 (Data, global_id -> local_id)。
    """
    from interbank_matcher_shared import PRIVATE_FEATURE_INDICES

    n_banks = len(self.banks)
    nodes = sorted({int(i) for i in node_ids if 0 <= int(i) < n_banks})
    if not nodes:
        empty = Data(
            x=torch.empty((0, 15), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1), dtype=torch.float),
        )
        return empty, {}

    id_map = {g: loc for loc, g in enumerate(nodes)}
    env = {
        "market_environment": getattr(self, "market_environment", "bull"),
        "base_rate": getattr(self, "base_rate", DAILY_BULL_BASE_RATE),
        "long_term_rate": getattr(
            self, "long_term_rate", DAILY_BULL_BASE_RATE + DAILY_LONG_RATE_SPREAD_BULL[0]
        ),
    }
    feats = [_bank_to_feature_vec_15(self.banks[g], env) for g in nodes]
    if mask_private and observer_idx is not None:
        obs = int(observer_idx)
        theta = float(getattr(self, "car_cutoff", 0.08))
        for loc, g in enumerate(nodes):
            if int(g) == obs:
                continue
            row = list(feats[loc])
            for fi in PRIVATE_FEATURE_INDICES:
                if 0 <= int(fi) < len(row):
                    row[int(fi)] = 0.0
            # Public / limited-info overlays (not full private levels):
            # 3 underpay intensity, 4 late ratio, 7 CAR grade, 8 LCR grade,
            # 9 bilateral exposure ratio vs observer (lender→borrower if g is lender).
            under, late = public_repayment_stats(self, int(g))
            row[3] = float(under)
            row[4] = float(late)
            # Lagged / pre-trade grades only (avoid look-ahead from post-trade CAR/LCR).
            b = self.banks[int(g)]
            car_src = float(b.get("lag_car", b.get("capital_adequacy_ratio", 0.0)) or 0.0)
            lcr_src = float(b.get("lag_lcr", b.get("liquidity_coverage_ratio", 0.0)) or 0.0)
            row[7] = public_car_risk_grade(
                {"capital_adequacy_ratio": car_src}, car_threshold=theta
            )
            row[8] = public_lcr_risk_grade({"liquidity_coverage_ratio": lcr_src})
            row[9] = 0.0  # bilateral exposure is a pair decoder feature
            feats[loc] = row
    x = torch.tensor(feats, dtype=torch.float)

    L_src = None
    if use_prev:
        L_src = getattr(self, "prev_exposure_matrix", None)
    if L_src is None:
        L_src = getattr(self, "exposure_matrix", None)

    src, dst, w = [], [], []
    if L_src is not None:
        L = np.asarray(L_src, dtype=float)
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                if i >= L.shape[0] or j >= L.shape[1]:
                    continue
                val = float(L[i, j])
                if val > 1e-9:
                    src.append(id_map[i])
                    dst.append(id_map[j])
                    w.append(val)

    if not src:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(w, dtype=torch.float).view(-1, 1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), id_map

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

    # 结算后清零同业网络（no rollover 的正确实现）
    self.exposure_matrix[:] = 0.0
    np.fill_diagonal(self.exposure_matrix, 0.0)

def build_candidate_graphs_for_pairs(base_graph, pairs, amounts):
    cand = []
    old_scale = float(getattr(base_graph, "edge_scale", 1.0))
    old_scale = max(old_scale, 1e-9)

    ei = base_graph.edge_index
    ea = getattr(base_graph, "edge_attr", None)
    if ea is None or ea.numel() == 0:
        ea = base_graph.x.new_zeros((0, 1))
    else:
        ea = base_graph.edge_attr.clone()

    for (i, j), w in zip(pairs, amounts):
        w = float(w)
        new_scale = max(old_scale, w, 1e-9)

        g = Data(
            x=base_graph.x.clone(),
            edge_index=ei.clone(),
            edge_attr=ea.clone()
        )

        if g.edge_attr.numel() > 0 and new_scale != old_scale:
            g.edge_attr = g.edge_attr * (old_scale / new_scale)

        ei_add = torch.tensor([[i], [j]], dtype=torch.long, device=base_graph.x.device)
        ea_add = torch.tensor([[w / new_scale]], dtype=torch.float, device=base_graph.x.device)

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
    """借入现金已在撮合成交时入账；此处仅按用途：项目部分从 liquid 转投资，最后清零 borrowed_cash。（对齐 Decentralized 的闭环记账逻辑）"""
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
    # else: keep frozen project book / amount untouched
def visualize_network(
    self,
    step,
    risk,
    tag: str = "",
    save: bool = True,
    show_first: bool = True,
    edge_quantile: float = 0.0,
    seed: int = DEFAULT_RANDOM_SEED,
    edge_lw_min: float = 0.4,
    edge_lw_max: float = 2.0,
):
    """
    交互式银行网络图（悬停查看信息）。边线宽在 [edge_lw_min, edge_lw_max] 内可控，不会爆粗。
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

    # 只画一遍：边线宽可控，log 压缩后严格落在 [edge_lw_min, edge_lw_max]，不会爆粗
    edge_lines = {}
    edge_lw = {}
    wmax = float(np.max(abs_ws)) if abs_ws.size else 1.0
    for (u, v) in strong_edges:
        (x0, y0), (x1, y1) = pos[u], pos[v]
        w = abs(G[u][v]["weight"])
        wn = np.log1p(float(w)) / (np.log1p(wmax) + 1e-9)
        wn = float(np.clip(wn, 0.0, 1.0))
        lw = float(np.clip(edge_lw_min + (edge_lw_max - edge_lw_min) * wn, edge_lw_min, edge_lw_max))
        ln = ax.plot(
            [x0, x1], [y0, y1],
            color="gray", alpha=0.50,
            lw=lw, zorder=1
        )[0]
        edge_lines[(u, v)] = ln
        edge_lw[(u, v)] = lw

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
                    ln.set_linewidth(edge_lw.get((u, v), 0.6))
            for ln in bg_lines:
                ln.set_alpha(0.02)
            fig.canvas.draw_idle()

        @cursor.connect("remove")
        def _on_remove(sel):
            for (u, v), ln in edge_lines.items():
                ln.set_alpha(0.50)
                ln.set_linewidth(edge_lw.get((u, v), 1.2))
            for ln in bg_lines:
                ln.set_alpha(0.25)
            fig.canvas.draw_idle()

    if show_first:
        plt.show()
        plt.pause(0.2)
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
        purpose: str = "matcher",
        oversample_risk: bool | None = None,
        seed_root: int = 42,
        seed_namespace: str = "MATCHER_TRAIN",
    ):
        """
        purpose:
          matcher — RFQ choice labels only (default; no risk oversampling)
          risk    — optional systemic-risk trajectories (may oversample high/low SR)
        """
        self.num_simulations = num_simulations
        self.num_timesteps = num_timesteps
        self.data_file = data_file
        self.expected_features = 15
        self.purpose = str(purpose or "matcher").strip().lower()
        self.seed_root = int(seed_root)
        self.seed_namespace = str(seed_namespace).strip().upper()
        if oversample_risk is None:
            oversample_risk = self.purpose == "risk"
        self.oversample_risk = bool(oversample_risk)
        need = int(min_samples) if min_samples is not None else int(num_simulations)
        if (not force_regenerate) and (not should_regenerate_dataset(
            self.data_file, force=False, min_samples=need
        )):
            self.data = load_json_dataset(self.data_file)
            print(
                f"[BankContagionDataset] load cache {self.data_file} "
                f"(n={len(self.data)}, purpose={self.purpose}, no append)"
            )
            return
        # Regenerate from scratch — never append onto a stale/partial cache.
        self.data = []
        self._generate_data()
        save_json_dataset(
            self.data_file,
            self.data,
            meta={
                "purpose": self.purpose,
                "oversample_risk": bool(self.oversample_risk),
                "seed_root": self.seed_root,
                "seed_namespace": self.seed_namespace,
            },
        )
        print(
            f"[BankContagionDataset] wrote {self.data_file} "
            f"(n={len(self.data)}, purpose={self.purpose}, "
            f"oversample_risk={self.oversample_risk}, force={force_regenerate})"
        )

    def _generate_data(self):
        """
        Bootstrap local RFQ trajectories for matcher training (v6).

        Matcher path: exactly ``num_simulations`` trajectories — no high/low SR
        rejection sampling (that loop can burn thousands of 5-step runs).

        Risk path (``purpose='risk'`` / ``oversample_risk=True``): optional
        extreme/low oversampling for a separate GNN-LSTM risk model.
        """
        print(
            "[BankContagionDataset] generating v6 competition RFQ-event labels "
            f"(schema={DATASET_SCHEMA_VERSION}, label={LABEL_DEFINITION_V6}, "
            f"purpose={self.purpose}, oversample_risk={self.oversample_risk})"
        )
        from interbank_matcher_shared import make_seed_stream
        seeds = make_seed_stream(
            self.seed_namespace, self.seed_root, self.num_simulations + 4000
        )

        def make_simulator(seed: int):
            sim = BankNetworkSimulator(
                num_banks=30, max_steps=self.num_timesteps, seed=int(seed)
            )
            sim._save_network_snapshot = False
            sim.export_policy_logs = False
            sim.require_gnn = True
            sim.allow_rate_only_fallback = False
            sim.collect_rfq_events = True
            sim.rfq_teacher_mode = "v6"
            sim.relationship_lending_enabled = True
            sim.relationship_history_window = 10
            sim.gnn_context = {
                "matcher": CompetitionTeacherMatcher(sim),
                "device": "cpu",
            }
            return sim

        def make_snapshot(simulator):
            g = simulator.to_pyg_graph(use_prev=False)
            events = list(getattr(simulator, "step_rfq_events", []) or [])
            simulator.step_rfq_events = []
            return {
                "node_features": g.x.detach().cpu().numpy().astype(float).tolist(),
                "edge_index":   g.edge_index.detach().cpu().numpy().astype(int).tolist(),
                "edge_attr":    g.edge_attr.detach().cpu().numpy().astype(float).tolist(),
                "rfq_events": events,
            }

        seed_cursor = 0

        def run_one_trajectory(label):
            nonlocal seed_cursor
            simulator = make_simulator(seeds[seed_cursor])
            trajectory_seed = int(seeds[seed_cursor])
            seed_cursor += 1
            try:
                simulator.initialize_network()
                simulator.step_rfq_events = []
                simulator.counterparty_touch_step = {}
            except Exception as e:
                print(f"[{label}] Error in initialize_network: {e}")
                raise

            sequence = []
            final_risk = 0.0
            for step in range(self.num_timesteps):
                try:
                    simulator.step_rfq_events = []
                    final_risk = simulator.simulate_step(step)
                except Exception as e:
                    print(f"[{label}] Error in simulate_step(step={step}): {e}")
                    raise

                try:
                    snap = make_snapshot(simulator)
                except Exception as e:
                    print(f"[{label}] Error in make_snapshot at step={step}: {e}")
                    print(f"    num_banks={simulator.num_banks}, "
                          f"exposure_matrix.shape={simulator.exposure_matrix.shape}")
                    raise

                snap["systemic_risk"] = float(final_risk)
                sequence.append(snap)

            return sequence, float(final_risk), trajectory_seed

        for _ in range(self.num_simulations):
            seq, final_risk, seed = run_one_trajectory("normal")
            self.data.append({"sequence": seq, "risk": float(final_risk), "seed": seed})

        if not self.oversample_risk:
            print(
                f"[BankContagionDataset] matcher mode: skipped high/low SR oversampling "
                f"(kept {len(self.data)} trajectories)"
            )
            return

        # Optional: only for a separate risk-model dataset, not matcher training.
        extreme_threshold = 0.8
        count_extreme, tries_extreme = 0, 0
        max_tries = 2000

        while count_extreme < 200 and tries_extreme < max_tries:
            tries_extreme += 1
            seq, final_risk, seed = run_one_trajectory("extreme")
            if final_risk > extreme_threshold:
                self.data.append({"sequence": seq, "risk": float(final_risk), "seed": seed})
                count_extreme += 1

        normal_threshold = 0.2
        count_normal, tries_normal = 0, 0

        while count_normal < 200 and tries_normal < max_tries:
            tries_normal += 1
            seq, final_risk, seed = run_one_trajectory("low")
            if final_risk < normal_threshold:
                self.data.append({"sequence": seq, "risk": float(final_risk), "seed": seed})
                count_normal += 1

        print(
            f"[BankContagionDataset] risk oversample done: "
            f"extreme={count_extreme}/{tries_extreme}, "
            f"low={count_normal}/{tries_normal}"
        )

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
        for t, data in enumerate(seq_raw):
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
    Structure-aware pair scorer: GCN over local exposure graph, then edge MLP on
    [h_i, h_j, |h_i-h_j|, h_i⊙h_j, pair_features].
    pair_features[0] = bilateral gross exposure / lender core ∈ [0,1].
    """
    def __init__(self, in_dim=15, hid=64, pair_dim: int = 1):
        super().__init__()
        from interbank_matcher_shared import PAIR_FEATURE_DIM

        self.pair_dim = int(pair_dim if pair_dim is not None else PAIR_FEATURE_DIM)
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hid * 4 + self.pair_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
        )

    def node_embed(self, g: Data):
        x = g.x
        edge_index = g.edge_index
        edge_weight = None
        if getattr(g, "edge_attr", None) is not None and g.edge_attr.numel() > 0:
            edge_weight = torch.log1p(
                torch.clamp(g.edge_attr.view(-1), min=0.0)
            )
            edge_weight = edge_weight / (edge_weight.mean() + 1e-9)
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h = F.relu(self.conv2(h, edge_index, edge_weight=edge_weight))
        return h

    def _pair_feat_tensor(self, pairs, pair_features, device):
        m = len(pairs)
        if pair_features is None:
            return torch.zeros((m, self.pair_dim), dtype=torch.float, device=device)
        pf = torch.as_tensor(pair_features, dtype=torch.float, device=device)
        if pf.ndim == 1:
            pf = pf.view(-1, 1)
        if pf.size(0) != m:
            raise ValueError(f"pair_features rows {pf.size(0)} != n_pairs {m}")
        if pf.size(1) < self.pair_dim:
            pad = torch.zeros((m, self.pair_dim - pf.size(1)), dtype=torch.float, device=device)
            pf = torch.cat([pf, pad], dim=1)
        elif pf.size(1) > self.pair_dim:
            pf = pf[:, : self.pair_dim]
        return torch.clamp(pf, 0.0, 1.0)

    @torch.no_grad()
    def score_pairs(self, g: Data, pairs, device="cpu", pair_features=None):
        from interbank_matcher_shared import sanitize_matcher_node_tensor

        self.eval()
        g = g.to(device)
        g.x = sanitize_matcher_node_tensor(g.x)
        h = self.node_embed(g)
        ii = torch.tensor([i for i, _ in pairs], dtype=torch.long, device=device)
        jj = torch.tensor([j for _, j in pairs], dtype=torch.long, device=device)
        hi, hj = h[ii], h[jj]
        pf = self._pair_feat_tensor(pairs, pair_features, device)
        z = torch.cat([hi, hj, torch.abs(hi - hj), hi * hj, pf], dim=1)
        return torch.sigmoid(self.edge_mlp(z).view(-1)).cpu().numpy()


# Backward-compatible alias (old name was a bank-only MLP; do not use for DEN-GNN claims).
BankPairMatcher = GNNPairMatcher

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
    model = GNNLSTMModel(input_dim=15, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    window_dataset = GraphWindowDataset(
        dataset, seq_len=seq_len, stride=1, drop_short=True
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
        train_loss = total_loss / len(train_loader)

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
        test_loss = test_loss / len(test_loader)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return model


def train_risk_lstm_from_dataset(
    *,
    num_simulations: int = 1000,
    num_timesteps: int = 5,
    data_file: Path | str | None = None,
    force_regenerate: bool = False,
    num_epochs: int = 10,
    batch_size: int = 32,
    checkpoint: Path | str | None = None,
):
    """
    Separate entry for the GNN-LSTM systemic-risk model.

    Not used by DEN matcher training. Builds an optional risk-oversampled
    dataset and saves ``gnn_lstm_model_shared.pth``.
    """
    import torch

    path = Path(data_file) if data_file is not None else (
        MODEL_DIR / "bank_contagion_data_risk_lstm.json"
    )
    ckpt = Path(checkpoint) if checkpoint is not None else SHARED_GNN_LSTM_PATH
    dataset = BankContagionDataset(
        num_simulations=int(num_simulations),
        num_timesteps=int(num_timesteps),
        data_file=str(path),
        force_regenerate=bool(force_regenerate),
        purpose="risk",
        oversample_risk=True,
    )
    model = train_model(dataset, num_epochs=int(num_epochs), batch_size=int(batch_size))
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print(f"[risk-lstm] trained/saved {ckpt} (dataset={path}, n={len(dataset)})")
    return model


def _neighbors_from_pyg(g, node: int) -> set[int]:
    """Undirected 1-hop neighbors of ``node`` in a PyG graph (global ids)."""
    out: set[int] = set()
    ei = getattr(g, "edge_index", None)
    if ei is None or ei.numel() == 0:
        return out
    n = int(node)
    for k in range(ei.size(1)):
        u = int(ei[0, k].item())
        v = int(ei[1, k].item())
        if u == n:
            out.add(v)
        elif v == n:
            out.add(u)
    return out


def _extract_local_pyg_subgraph(full_g: Data, node_ids) -> tuple[Data, dict[int, int]]:
    """Induce a local subgraph on ``node_ids`` (global indices in full_g)."""
    nodes = sorted({int(i) for i in node_ids if 0 <= int(i) < int(full_g.x.size(0))})
    if not nodes:
        empty = Data(
            x=torch.empty((0, full_g.x.size(1)), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1), dtype=torch.float),
        )
        return empty, {}
    id_map = {g: loc for loc, g in enumerate(nodes)}
    x = full_g.x[nodes].clone()
    src, dst, w = [], [], []
    ei = getattr(full_g, "edge_index", None)
    ea = getattr(full_g, "edge_attr", None)
    if ei is not None and ei.numel() > 0:
        node_set = set(nodes)
        for k in range(ei.size(1)):
            u = int(ei[0, k].item())
            v = int(ei[1, k].item())
            if u in node_set and v in node_set and u != v:
                src.append(id_map[u])
                dst.append(id_map[v])
                if ea is not None and ea.numel() > 0:
                    w.append(float(ea[k].view(-1)[0].item()))
                else:
                    w.append(1.0)
    if not src:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(w, dtype=torch.float).view(-1, 1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), id_map


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


def _iter_rfq_training_events(dataset):
    """
    Yield (input_graph, rfq_event, trajectory_seed). v6 uses the exact RFQ-time
    local graph snapshot; stale datasets fall back to the prior full graph.
    """
    raw = getattr(dataset, "data", None)
    if raw is None and hasattr(dataset, "base_dataset"):
        raw = getattr(dataset.base_dataset, "data", None)
    if not isinstance(raw, list):
        # Fallback: GraphWindow / list-of-graphs without RFQ metadata.
        return
    for item in raw:
        seq = item.get("sequence") if isinstance(item, dict) else None
        if not isinstance(seq, list) or len(seq) < 2:
            continue
        for t in range(1, len(seq)):
            prev = seq[t - 1]
            cur = seq[t]
            events = cur.get("rfq_events") or []
            if not events:
                continue
            x = torch.tensor(prev["node_features"], dtype=torch.float)
            edge_index = torch.tensor(prev["edge_index"], dtype=torch.long)
            edge_attr = torch.tensor(prev["edge_attr"], dtype=torch.float)
            g_prev = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            for ev in events:
                local = ev.get("local_graph") if isinstance(ev, dict) else None
                if isinstance(local, dict) and local.get("node_features") is not None:
                    g_in = Data(
                        x=torch.tensor(local["node_features"], dtype=torch.float),
                        edge_index=torch.tensor(local["edge_index"], dtype=torch.long),
                        edge_attr=torch.tensor(local["edge_attr"], dtype=torch.float),
                    )
                else:
                    g_in = g_prev
                yield g_in, ev, int(item.get("seed", -1))


def train_matcher_from_dataset(dataset, epochs=3, lr=1e-3, neg_ratio=1.0, batch_graphs=64):
    """
    Train GNNPairMatcher on exact RFQ-time v6 events.

    Positive = lenders accepted in that RFQ; negative = same candidate set
    that were seen but not accepted. Never sample arbitrary directed pairs
    and never treat outstanding rollover edges as positives.
    Pair decoder receives bilateral exposure ratio from event.pair_feats.
    """
    from interbank_matcher_shared import (
        PRIVATE_FEATURE_INDICES, PAIR_FEATURE_DIM, sanitize_matcher_node_tensor,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = GNNPairMatcher(in_dim=15, hid=64, pair_dim=PAIR_FEATURE_DIM).to(device)
    opt = torch.optim.Adam(matcher.parameters(), lr=float(lr), weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()

    # (graph, positive-pairs, negative-pairs, pair-features, teacher scores, seed)
    local_units = []
    for g_prev, ev, trajectory_seed in _iter_rfq_training_events(dataset) or []:
        if "repay_feats" not in ev or "local_graph" not in ev or "teacher_scores" not in ev:
            raise RuntimeError(
                "Stale pre-v6 dataset: RFQ-time graph/teacher scores missing; regenerate v6."
            )
        borrower = int(ev.get("borrower", -1))
        candidates = [int(x) for x in (ev.get("candidates") or [])]
        accepted = {int(x) for x in (ev.get("accepted") or [])}
        hist = [int(x) for x in (ev.get("hist") or [])]
        raw_pf = ev.get("pair_feats") or {}
        raw_scores = ev.get("teacher_score_unit") or {}
        score_by_global = {
            int(k): float(v) for k, v in raw_scores.items()
        } if isinstance(raw_scores, dict) else {}
        pair_feat_by_global: dict[int, list[float]] = {}
        if isinstance(raw_pf, dict):
            for k, vals in raw_pf.items():
                try:
                    pair_feat_by_global[int(k)] = [float(v) for v in (vals or [0.0])]
                except (TypeError, ValueError):
                    continue
        raw_repay = ev.get("repay_feats") or {}
        if not isinstance(raw_repay, dict):
            raise RuntimeError(
                "Stale pre-v6 dataset: repay_feats must be a dict; regenerate dataset."
            )
        repay_by_global: dict[int, tuple[float, float]] = {}
        for k, vals in raw_repay.items():
            try:
                row = list(vals or [0.0, 0.0])
                under = float(row[0]) if len(row) > 0 else 0.0
                late = float(row[1]) if len(row) > 1 else 0.0
                repay_by_global[int(k)] = (
                    float(np.clip(under, 0.0, 1.0)),
                    float(np.clip(late, 0.0, 1.0)),
                )
            except (TypeError, ValueError):
                continue
        if borrower < 0 or not candidates:
            continue
        local_payload = ev.get("local_graph") or {}
        candidate_local = local_payload.get("candidate_local_ids") or {}
        if local_payload and candidate_local:
            local_g = g_prev
            id_map = {
                int(k): int(v) for k, v in candidate_local.items()
            }
            id_map[borrower] = int(local_payload.get("borrower_local_id", -1))
        else:
            nodes = {borrower} | set(hist) | set(candidates)
            local_g, id_map = _extract_local_pyg_subgraph(g_prev, nodes)
        # Legacy fallback masking only; v6 graph is already inference-identical.
        if not local_payload and local_g.x is not None and local_g.x.numel() > 0 and borrower in id_map:
            x = local_g.x.clone()
            thr = 0.08
            for g_id, loc in id_map.items():
                if int(g_id) == borrower:
                    continue
                loc = int(loc)
                car_raw = float(x[loc, 7].item()) if x.size(1) > 7 else 0.0
                lcr_raw = float(x[loc, 8].item()) if x.size(1) > 8 else 0.0
                for fi in PRIVATE_FEATURE_INDICES:
                    if 0 <= int(fi) < x.size(1):
                        x[loc, int(fi)] = 0.0
                # Public grades reconstructed from cached ratio features.
                if car_raw >= 1.5 * thr:
                    car_g = 0.0
                elif car_raw >= thr:
                    car_g = 0.33
                elif car_raw >= 0.5 * thr:
                    car_g = 0.66
                else:
                    car_g = 1.0
                if lcr_raw >= 1.2:
                    lcr_g = 0.0
                elif lcr_raw >= 1.0:
                    lcr_g = 0.33
                elif lcr_raw >= 0.8:
                    lcr_g = 0.66
                else:
                    lcr_g = 1.0
                under, late = repay_by_global.get(int(g_id), (0.0, 0.0))
                x[loc, 3] = float(under)
                x[loc, 4] = float(late)
                x[loc, 7] = float(car_g)
                x[loc, 8] = float(lcr_g)
                x[loc, 9] = 0.0  # bilateral exposure is pair decoder feature
            local_g.x = x
        pos_locals = []
        neg_locals = []
        pair_feat_by_local: dict[int, list[float]] = {}
        score_by_local: dict[int, float] = {}
        if borrower not in id_map:
            continue
        bj = int(id_map[borrower])
        for i in candidates:
            if i not in id_map:
                continue
            li = int(id_map[i])
            pair = (li, bj)
            pf = pair_feat_by_global.get(int(i))
            if pf is None:
                pf = [0.0] * int(PAIR_FEATURE_DIM)
            pair_feat_by_local[li] = list(pf)
            score_by_local[li] = float(score_by_global.get(int(i), 0.0))
            if i in accepted:
                pos_locals.append(pair)
            else:
                neg_locals.append(pair)
        if not pos_locals:
            # Preserve all-negative RFQs: they are informative rejection events.
            if neg_locals:
                local_units.append(
                    (local_g, [], list(neg_locals), pair_feat_by_local,
                     score_by_local, trajectory_seed)
                )
            continue
        local_units.append(
            (local_g, list(pos_locals), list(neg_locals), pair_feat_by_local,
             score_by_local, trajectory_seed)
        )

    if not local_units:
        raise RuntimeError(
            "train_matcher_from_dataset: no RFQ-event training units. "
            "Regenerate bank_contagion_data_local_v6.json with collect_rfq_events."
        )

    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    def _is_validation_unit(unit) -> bool:
        return int.from_bytes(
            hashlib.sha256(str(int(unit[5])).encode("ascii")).digest()[:4], "little"
        ) % 5 == 0

    validation_units = [u for u in local_units if _is_validation_unit(u)]
    train_units = [u for u in local_units if not _is_validation_unit(u)]
    if not train_units or not validation_units:
        cut = max(1, int(0.8 * len(local_units)))
        train_units, validation_units = local_units[:cut], local_units[cut:]
    for ep in range(epochs):
        random.shuffle(train_units)
        total = 0.0
        cnt = 0

        for s in range(0, len(train_units), batch_graphs):
            batch = train_units[s:s + batch_graphs]
            loss_acc = 0.0
            opt.zero_grad()

            for local_g, pos_pool, neg_pool, pf_by_local, score_by_local, _seed in batch:
                g = local_g.to(device)
                g.x = sanitize_matcher_node_tensor(g.x)
                N = int(g.x.size(0))
                if N < 2:
                    continue

                # Negatives: other candidates from the same RFQ only.
                neg_pairs = list(neg_pool)
                num_neg = max(1, int(float(neg_ratio) * max(1, len(pos_pool))))
                if len(neg_pairs) > num_neg:
                    pick = rng.choice(len(neg_pairs), size=num_neg, replace=False)
                    neg_pairs = [neg_pairs[int(k)] for k in pick]
                pairs = list(pos_pool) + neg_pairs
                if not pairs:
                    continue
                y = torch.tensor(
                    [1.0] * len(pos_pool) + [0.0] * len(neg_pairs),
                    dtype=torch.float,
                    device=device,
                )
                pair_rows = []
                for li, _bj in pairs:
                    row = list(pf_by_local.get(int(li), [0.0] * int(matcher.pair_dim)))
                    if len(row) < matcher.pair_dim:
                        row = row + [0.0] * (matcher.pair_dim - len(row))
                    pair_rows.append(row[: matcher.pair_dim])

                h = matcher.node_embed(g)
                ii = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
                jj = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
                hi, hj = h[ii], h[jj]
                pf = torch.as_tensor(pair_rows, dtype=torch.float, device=device)
                pf = torch.clamp(pf, 0.0, 1.0)
                feat = torch.cat([hi, hj, torch.abs(hi - hj), hi * hj, pf], dim=1)
                logit = matcher.edge_mlp(feat).view(-1)
                teacher_y = torch.tensor(
                    [float(score_by_local.get(int(li), 0.0)) for li, _ in pairs],
                    dtype=torch.float, device=device,
                )
                score_loss = F.smooth_l1_loss(torch.sigmoid(logit), teacher_y)
                rank_loss = logit.new_tensor(0.0)
                rank_terms = []
                for a_idx in range(len(pairs)):
                    for b_idx in range(a_idx + 1, len(pairs)):
                        delta = teacher_y[a_idx] - teacher_y[b_idx]
                        if torch.abs(delta).item() > 1e-6:
                            sign = torch.sign(delta)
                            rank_terms.append(F.softplus(-sign * (logit[a_idx] - logit[b_idx])))
                if rank_terms:
                    rank_loss = torch.stack(rank_terms).mean()
                loss = bce(logit, y) + 0.2 * score_loss + 0.3 * rank_loss
                loss.backward()
                loss_acc += float(loss.item())
                cnt += 1

            torch.nn.utils.clip_grad_norm_(matcher.parameters(), 1.0)
            opt.step()
            if cnt > 0:
                total += loss_acc

        avg = total / max(1, cnt)
        print(f"[matcher-v6-rfq] epoch {ep+1}/{epochs} loss={avg:.4f} (exact RFQ-time)")

    val_pred, val_label, val_teacher = [], [], []
    top1_hits = []
    matcher.eval()
    for local_g, pos_pool, neg_pool, pf_by_local, score_by_local, _seed in validation_units:
        pairs = list(pos_pool) + list(neg_pool)
        if not pairs:
            continue
        pair_rows = [pf_by_local.get(int(li), [0.0] * matcher.pair_dim) for li, _ in pairs]
        pred = matcher.score_pairs(
            local_g, pairs, device=device, pair_features=pair_rows
        ).astype(float)
        labels = [1.0] * len(pos_pool) + [0.0] * len(neg_pool)
        teacher = np.asarray(
            [score_by_local.get(int(li), 0.0) for li, _ in pairs], dtype=float
        )
        val_pred.extend(pred.tolist())
        val_label.extend(labels)
        val_teacher.extend(teacher.tolist())
        if len(pairs) > 1 and float(np.ptp(teacher)) > 1e-9:
            top1_hits.append(int(np.argmax(pred) == np.argmax(teacher)))

    vp = np.asarray(val_pred, dtype=float)
    vy = np.asarray(val_label, dtype=float)
    vt = np.asarray(val_teacher, dtype=float)
    if np.any(vy > 0.5) and np.any(vy <= 0.5):
        order = np.argsort(vp, kind="mergesort")
        ranks = np.empty(len(vp), dtype=float)
        ranks[order] = np.arange(1, len(vp) + 1, dtype=float)
        n_pos, n_neg = float(np.sum(vy > 0.5)), float(np.sum(vy <= 0.5))
        auc = float(
            (np.sum(ranks[vy > 0.5]) - n_pos * (n_pos + 1.0) / 2.0)
            / (n_pos * n_neg)
        )
    else:
        auc = float("nan")
    if vp.size and np.any(vy > 0.5):
        order = np.argsort(-vp)
        y_sorted = vy[order]
        precision = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1)
        average_precision = float(
            np.sum(precision * y_sorted) / max(1.0, float(np.sum(y_sorted)))
        )
    else:
        average_precision = float("nan")

    matcher.training_metrics = {
        "n_train_units": int(len(train_units)),
        "n_validation_units": int(len(validation_units)),
        "split": "trajectory_seed_sha256_mod5",
        "loss": "bce+0.2*smooth_l1_teacher+0.3*pairwise_rank",
        "all_negative_units": int(sum(not u[1] for u in local_units)),
        "validation_roc_auc": auc,
        "validation_pr_auc": average_precision,
        "validation_brier": float(np.mean((vp - vy) ** 2)) if vp.size else float("nan"),
        "validation_teacher_mae": float(np.mean(np.abs(vp - vt))) if vp.size else float("nan"),
        "validation_top1_teacher_accuracy": (
            float(np.mean(top1_hits)) if top1_hits else float("nan")
        ),
    }
    return matcher


def load_gnn_pair_matcher(path=None, device=None):
    """Load frozen local-subgraph GNNPairMatcher (v6 dict checkpoint)."""
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(path) if path is not None else GNN_PAIR_MATCHER_V6_LOCAL_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"GNN local matcher checkpoint missing: {path}. "
            f"Run with --matcher-mode train once to create gnn_pair_matcher_v6_local.pth "
            f"(do not reuse v2/v3/v4/v5 checkpoints)."
        )
    resolved = path.resolve()
    banned = {
        Path(GNN_PAIR_MATCHER_V2_PATH).resolve(),
        Path(GNN_PAIR_MATCHER_V3_LOCAL_PATH).resolve(),
        Path(GNN_PAIR_MATCHER_V4_LOCAL_PATH).resolve(),
        Path(GNN_PAIR_MATCHER_V5_LOCAL_PATH).resolve(),
    }
    if resolved in banned:
        raise RuntimeError(
            "Refusing deprecated matcher checkpoint for formal DEN. "
            "Train/load gnn_pair_matcher_v6_local.pth instead."
        )
    matcher = GNNPairMatcher(in_dim=15, hid=64)
    state, _meta = load_matcher_state_dict(path, device=device)
    matcher.load_state_dict(state)
    return matcher.to(device).eval()


def resolve_matcher(
    matcher_mode: str = "off",
    *,
    device=None,
    dataset=None,
    checkpoint: Path | None = None,
    mechanism: str = "decentralized",
):
    """
    matcher_mode:
      train — bootstrap RFQ-event dataset, train local matcher, save v6
      load  — load frozen gnn_pair_matcher_v6_local.pth
      off   — rate-only / no GNN (debug only)
    """
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = str(matcher_mode).strip().lower()
    ckpt = Path(checkpoint) if checkpoint is not None else GNN_PAIR_MATCHER_V6_LOCAL_PATH

    if mode == "off":
        return None, matcher_meta(
            matcher_mode="off", matcher=None, checkpoint=None, mechanism=mechanism
        )

    if mode == "train":
        data_path = SHARED_LOCAL_DATA_FILE
        if dataset is None:
            dataset = BankContagionDataset(
                num_simulations=1000,
                num_timesteps=5,
                data_file=str(data_path),
                # Always rebuild: --force-rerun alone does not refresh matcher data.
                force_regenerate=True,
                purpose="matcher",
                oversample_risk=False,
            )
        matcher = train_matcher_from_dataset(
            dataset, epochs=3, lr=1e-4, neg_ratio=1.0, batch_graphs=64
        )
        matcher = matcher.to(device).eval()
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        save_matcher_checkpoint(
            ckpt,
            matcher.state_dict(),
            data_path=data_path,
            train_config={
                "epochs": 3,
                "lr": 1e-4,
                "neg_ratio": 1.0,
                "label_definition": LABEL_DEFINITION_V6,
                "validation": dict(getattr(matcher, "training_metrics", {})),
                "purpose": "matcher",
                "oversample_risk": False,
            },
        )
        print(f"[matcher] trained local GNNPairMatcher v6 and saved {ckpt}")
        return matcher, matcher_meta(
            matcher_mode="train",
            matcher=matcher,
            checkpoint=ckpt,
            mechanism=mechanism,
        )

    if mode == "load":
        matcher = load_gnn_pair_matcher(ckpt, device=device)
        print(f"[matcher] loaded local GNNPairMatcher v6 from {ckpt}")
        return matcher, matcher_meta(
            matcher_mode="load",
            matcher=matcher,
            checkpoint=ckpt,
            mechanism=mechanism,
        )

    raise ValueError(f"Unknown matcher_mode={matcher_mode!r}; use train|load|off")

def predict_and_regulate(model, matcher, simulator, num_steps=5, seq_len=5, draw_net=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 风险预测模型：仍用于预测SR/出建议
    model = model.to(device).eval()

    # 撮合 matcher：用于 Step3 生成 a_ij（如果 matcher 是 torch.nn.Module）
    if matcher is not None and hasattr(matcher, "to"):
        matcher = matcher.to(device).eval()

    simulator.initialize_network()
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

    errors = [abs(r - p) for r, p in zip(risks, pred_risks) if p is not None]
    if errors:
        print("\nNormal Test Prediction Error Statistics:")
        print(f"Mean Error: {np.mean(errors):.4f}")
        print(f"Std Error: {np.std(errors):.4f}")

    return risks, pred_risks


# ========= 下面保持你原始结构（到你粘贴处为止） =========
# 最多运行 4000 步；商业银行只剩 STOP_ALIVE_THRESHOLD 家时早停，以其 collapse_step 为存活时间。
DEFAULT_SIM_T_CAP = 4000
MAX_PLOT_STEPS = 4000
STOP_ALIVE_THRESHOLD = 1
MAX_NETWORK_SNAPSHOT_STEP = 200  # 网络快照只画早中期；轨迹图用真实早停长度，不拉到 T_cap

# ----- 共享仿真录制：一次（或一批 seed）跑完，所有 sweep 图只重算 SR 分量 -----
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
    """固定 policy 路径下录制的多 seed 轨迹；改 W / θ_measure 只重算不打分仿真。"""
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
    """θ sweep spread≈0 时：打印 CAR 分布是否落在 sweep 区间内。"""
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
    """batch 模式：sweep 的 baseline θ 曲线应与 score_batch_mean_components 完全一致。"""
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
    """供 _components_from_state 读取单步快照。"""

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
    require_gnn: bool = True,
    allow_rate_only_fallback: bool = False,
) -> RecordedRun:
    """Record one run; early-stop when remaining commercial banks <= stop_alive_threshold."""
    T = min(int(T), MAX_PLOT_STEPS)
    sim = BankNetworkSimulator(num_banks=N, max_steps=T, B=B, sigma=sigma, seed=int(seed))
    configure_simulation_features(
        sim,
        rollover_enabled=rollover_enabled,
        policy_support_enabled=policy_support_enabled,
    )
    sim.require_gnn = bool(require_gnn)
    sim.allow_rate_only_fallback = bool(allow_rate_only_fallback)
    sim._save_network_snapshot = False
    sim.export_policy_logs = False
    sim.verbose_matching = False
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
    require_gnn: bool = True,
    allow_rate_only_fallback: bool = False,
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
                require_gnn=require_gnn,
                allow_rate_only_fallback=allow_rate_only_fallback,
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
  若传入 batch：仅在固定 policy 轨迹上重算 W / θ_measure（不再为每个网格点重跑仿真）。
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
    T_steps    = min(DEFAULT_SIM_T_CAP, MAX_PLOT_STEPS)

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
        if len(sr) == 0:
            return np.nan, np.nan
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
        sim._save_network_snapshot = False
        sim.export_policy_logs = False
        sim.car_cutoff = float(theta_policy)
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
    edge_lw_min: float = 0.4,
    edge_lw_max: float = 2.0,
    rollover_enabled: bool = True,
    policy_support_enabled: bool = True,
):
    """合并原 generate_network_snapshots 与 generate_matcher_snapshots。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sim = BankNetworkSimulator(max_steps=max(steps) + 1)
    configure_simulation_features(
        sim,
        rollover_enabled=rollover_enabled,
        policy_support_enabled=policy_support_enabled,
    )
    sim.require_gnn = matcher is not None
    sim.allow_rate_only_fallback = matcher is None
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
                edge_lw_min=edge_lw_min, edge_lw_max=edge_lw_max,
            )
            path = FIG_DIR / f"network_{tag}_step{snap['step']}.png"
            print(f"Saved: {path}")
            saved.append(path)
    finally:
        sim.banks = current_banks
        sim.exposure_matrix = current_exposure
    return saved


def generate_gnn_panel(
    steps=(10, 20, 30, 40, 50),
    tag="gnnbase",
    matcher=None,
    device=None,
    show=True,
    edge_lw_min: float = 0.4,
    edge_lw_max: float = 2.0,
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
        edge_lw_min=edge_lw_min,
        edge_lw_max=edge_lw_max,
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



def _components_from_state(sim, weights=(0.5, 0.3, 0.2), theta=0.08):
    return _systemic_risk_from_banks(
        sim.banks, weights=weights, car_threshold=float(theta)
    )


def _nanmean_variable_length(series_list):
    """对提前停止导致的不同长度轨迹按实际长度做均值。"""
    non_empty = [np.asarray(x, dtype=float) for x in series_list if len(x) > 0]
    if not non_empty:
        return np.asarray([], dtype=float)
    max_len = max(len(x) for x in non_empty)
    mat = np.full((len(non_empty), max_len), np.nan, dtype=float)
    for i, arr in enumerate(non_empty):
        mat[i, :len(arr)] = arr
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
    画基准情景的 SR & 组成（FR/CBS/CGR）随时间的轨迹图。T 限制在 MAX_PLOT_STEPS 以内。
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
    生成情景对比折线图 + t@0.5 竖线。T 限制在 MAX_PLOT_STEPS 以内。
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
        if len(sr) == 0:
            continue
        xs = np.arange(1, len(sr) + 1)
        line, = ax.plot(xs, sr, ls=ls, marker='o', ms=4,
                        label=f"{name} (W={w}|θ={th})")

        t05 = _first_cross_time(sr, 0.5)
        if np.isfinite(t05):
            ax.axvline(x=t05, color=line.get_color(), linestyle=':', alpha=0.6)
            ax.text(t05, 0.5, "t₀․₅", color=line.get_color(),
                    ha='left', va='bottom', fontsize=9, alpha=0.8)

        ax.annotate(f"{name}\nStep={len(sr)}, SR={sr[-1]:.3f}",
                    xy=(xs[-1], sr[-1]), xytext=(8, 8), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                    fontsize=9)

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

    batch = _resolve_plot_batch(batch) if use_shared_batch else batch
    if batch is not None and float(theta_policy) != float(batch.theta_policy):
        batch = None
    if batch is not None:
        return score_batch_mean_components(
            batch, weights=weights, theta_measure=float(theta_measure)
        )

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
        sim._save_network_snapshot = False
        sim.export_policy_logs = False
        sim.car_cutoff = float(theta_policy)
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
    theta_policy_fixed=None,   # None: policy θ 跟随 th；否则 policy θ 固定
    theta_min=0.08,
    theta_max=0.15,
    n_theta=8,
    track="sr",                # 'sr'/'cbs'/'fr'/'cgr'
    baseline_theta=0.08,       # ★要高亮的 baseline θ
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
    # If batch is absent and policy follows θ, each θ re-simulates with nsim (not max(50,nsim)).
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
        print("[theta-sweep] policy-follow: each θ re-runs simulation (car_cutoff = θ)")
    elif batch is not None:
        print(
            f"[plot-batch] measure-only sweep on shared batch, "
            f"policy θ={batch.theta_policy:.2f}"
        )
        theta_policy_fixed = float(batch.theta_policy)

    # ===== sweep curves =====
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
        "sr":  "Systemic Risk (SR)",
        "cbs": "CBS (active CAR<θ / active)",
        "fr":  "FR (Failure Rate)",
        "cgr": "CGR (gap / required)"
    }[track]
    ttl = {"sr": "SR", "cbs": "CBS", "fr": "FR", "cgr": "CGR"}[track]

    base_idx = int(np.argmin(np.abs(theta_grid - baseline_theta)))
    y_ref = curves[base_idx] if len(curves[base_idx]) > 0 else None

    # ===== plot =====
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

    # sweep lines (highlight baseline inside loop)
    for i, (th, y) in enumerate(zip(theta_grid, curves)):
        if len(y) == 0:
            continue
        xs = np.arange(1, len(y) + 1)
        is_base = np.isclose(th, baseline_theta, atol=1e-12)
        color = cmap(norm(th))
        ls = linestyles[i % len(linestyles)]

        lw = 3.0 if is_base else 1.8
        z  = 10  if is_base else 2
        a  = 1.0 if is_base else 0.92

        line, = ax.plot(xs, y, lw=lw, alpha=a, color=color, linestyle=ls, zorder=z)

        if is_base:
            line.set_path_effects([
                pe.Stroke(linewidth=6.5, foreground="white", alpha=0.95),
                pe.Normal()
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
    out = FIG_DIR / (
        f"{output_prefix}_{track}_theta{theta_min:.2f}-{theta_max:.2f}_n{n_theta}"
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
    """纯测度敏感性：固定系统演化阈值，只改变 SR 计算口径的 θ。"""
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
    """政策情景：θ 同时改变 car_cutoff（角色分配/网络）与 SR 测度口径；每个 θ 重跑仿真。"""
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
    for t in range(T):
        sim.simulate_step(t)
        sr = sim.calculate_systemic_risk()
        sr_list.append(sr)
        if sim.network_stable_step is not None:
            break
    end = time.perf_counter()

    elapsed = end - start
    steps_run = max(1, len(sr_list))
    print(f"[measure] banks={num_banks}, T={T}, steps_run={len(sr_list)}")
    print(f"  Total time: {elapsed:.4f} seconds")
    print(f"  Per step : {elapsed / steps_run:.6f} seconds/step")

    return sr_list, elapsed

def _init_network_snapshot_schedule(self, block_size=50, start_step=1, seed=42):
    """
    Build schedule of steps to plot: one random step per block of block_size, starting at start_step.
    Blocks: [1..50], [51..100], [101..150], ...
    暂时：不安排 >= MAX_NETWORK_SNAPSHOT_STEP 的步数。
    """
    rng = random.Random(seed)
    last_step = min(self.max_steps - 1, MAX_NETWORK_SNAPSHOT_STEP - 1)
    self._net_snapshot_steps = set()
    s = start_step
    while s <= last_step:
        block_end = min(s + block_size - 1, last_step)
        if s <= block_end:
            step = rng.randint(s, block_end)
            self._net_snapshot_steps.add(step)
        s += block_size


def maybe_save_network_snapshot(self, step, risk, tag="rfq", edge_quantile=0.0):
    """
    Save network graph if (A) step is in scheduled random-per-block steps, or
    (B) step is the network-stable step.
    Only runs when sim._save_network_snapshot=True (data gen 等设为 False 跳过).
    """
    try:
        if not getattr(self, "_save_network_snapshot", False):
            return
        # --- hard stop: do not plot beyond max_steps (e.g., 200) ---
        if step >= int(getattr(self, "max_steps", 10**9)):
            return
        # --- 暂时：所有 network 图在 1000 step 停止 ---
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
BankNetworkSimulator.to_local_pyg_graph = to_local_pyg_graph
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
            matcher_mode="off" if matcher is None else "load",
            matcher=matcher,
            checkpoint=(GNN_PAIR_MATCHER_V6_LOCAL_PATH if matcher is not None else None),
            mechanism="decentralized",
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
    matcher_mode: str = "load",
    train_models: bool | None = None,
    T: int = DEFAULT_SIM_T_CAP,
    nsim: int = 20,
    B: float = DEFAULT_MATCH_B,
    fig_dir: Path | None = None,
    network_steps=(10, 20, 30, 40, 50, 100, 150, 200),
    network_tag: str = "decentralized",
    show: bool = False,
    rollover_enabled: bool = True,
    policy_support_enabled: bool = True,
    stop_alive_threshold: int | None = STOP_ALIVE_THRESHOLD,
    allow_den_rate_only: bool = False,
    seed0: int = DEFAULT_RANDOM_SEED,
) -> dict:
    """标准结果图：network panel、θ sweep、w1 sweep、baseline trajectory。"""
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configure_figure_output(fig_dir)

    # Legacy: train_models=True → matcher_mode train; False alone does not force off.
    if train_models is True and str(matcher_mode).lower() == "off":
        matcher_mode = "train"

    mode = str(matcher_mode).strip().lower()
    if mode == "off" and not allow_den_rate_only:
        raise RuntimeError(
            "Formal DEN must use matcher_mode='load' or 'train'. "
            "Pass allow_den_rate_only=True only for debugging rate-only RFQ."
        )

    matcher_info = matcher_meta(
        matcher_mode=str(matcher_mode),
        matcher=None,
        checkpoint=None,
        mechanism="decentralized",
    )
    if matcher is None and mode != "off":
        matcher, matcher_info = resolve_matcher(
            matcher_mode,
            device=device,
            dataset=dataset,
            mechanism="decentralized",
        )
    elif matcher is not None:
        matcher_info = matcher_meta(
            matcher_mode=str(matcher_mode) if mode != "off" else "load",
            matcher=matcher,
            checkpoint=GNN_PAIR_MATCHER_V6_LOCAL_PATH,
            mechanism="decentralized",
        )

    # Matcher training must not auto-train the unrelated GNN-LSTM risk model.
    # Call train_risk_lstm_from_dataset(...) separately if needed.

    require_gnn = mode in ("train", "load") or not allow_den_rate_only
    if require_gnn and matcher is None:
        raise RuntimeError(
            "DEN-GNN requested, but no trained local GNN matcher was loaded. "
            "Run with --matcher-mode train once to create gnn_pair_matcher_v6_local.pth"
        )

    network_panel = generate_gnn_panel(
        steps=network_steps,
        tag=network_tag,
        matcher=matcher,
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
        matcher=matcher,
        device=device,
        rollover_enabled=rollover_enabled,
        policy_support_enabled=policy_support_enabled,
        stop_alive_threshold=stop_alive_threshold,
        require_gnn=require_gnn,
        allow_rate_only_fallback=bool(allow_den_rate_only),
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
        matcher=matcher,
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
        "matcher": matcher,
        "matcher_info": matcher_info,
        "device": device,
    }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Decentralized RFQ 标准结果图")
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
        choices=["train", "load", "off"],
        default="load",
        help="train=训练局部 GNNPairMatcher v6；load=加载 gnn_pair_matcher_v6_local（默认）；off=rate-only（仅调试）",
    )
    parser.add_argument(
        "--train-risk-lstm",
        action="store_true",
        help="单独训练 GNN-LSTM 风险模型（与 matcher 无关；默认不跑）",
    )
    parser.add_argument(
        "--allow-den-rate-only",
        action="store_true",
        help="调试：允许 --matcher-mode off（正式论文结果禁止）",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="(已废弃) 请改用 --matcher-mode load；不再静默切到 rate-only",
    )
    parser.add_argument("--no-rollover", action="store_true", help="关闭同业 rollover 分期续借")
    parser.add_argument("--no-policy-support", action="store_true", help="关闭央行 liquidity/capital support 注入")
    parser.add_argument(
        "--stop-alive-threshold",
        type=int,
        default=STOP_ALIVE_THRESHOLD,
        help="剩余商业银行数 <= 该阈值时早停（主实验 4；稳健性 3；央行不计）",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("错误：未安装 PyTorch。请运行：")
        print(f"  {sys.executable} -m pip install torch torch-geometric")
        raise SystemExit(1)

    matcher_mode = str(args.matcher_mode)
    if args.no_train:
        print("[warn] --no-train 已废弃，正式 DEN 仍使用 --matcher-mode（默认 load）")

    t_all = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.train_risk_lstm:
        train_risk_lstm_from_dataset(force_regenerate=False)
        print(f"[time] risk-lstm only: {time.perf_counter() - t_all:.2f}s")
        raise SystemExit(0)

    run_standard_figure_pipeline(
        device=device,
        matcher_mode=matcher_mode,
        T=args.T,
        nsim=args.nsim,
        B=args.B,
        fig_dir=args.fig_dir,
        network_tag="decentralized",
        show=args.show,
        rollover_enabled=not args.no_rollover,
        policy_support_enabled=not args.no_policy_support,
        stop_alive_threshold=args.stop_alive_threshold,
        allow_den_rate_only=bool(args.allow_den_rate_only),
        seed0=int(args.seed0),
    )
    print(f"[time] TOTAL: {time.perf_counter() - t_all:.2f}s")
