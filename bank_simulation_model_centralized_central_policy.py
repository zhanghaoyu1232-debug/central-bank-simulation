# ===== Runtime setup (paste from line 1) =====
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"networkx backend defined more than once",
    category=RuntimeWarning,
    module=r"networkx\.utils\.backends",
)

from pathlib import Path
OUTPUT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = OUTPUT_ROOT / "输入"
OUTPUT_DIR = OUTPUT_ROOT / "输出"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# 统一所有图片的输出目录
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MAX_PLOT_STEPS = 1000
MAX_NETWORK_SNAPSHOT_STEP = 1000

import matplotlib
try:
    matplotlib.use("TkAgg")   # 有桌面环境
except Exception:
    matplotlib.use("Agg")     # 服务器/无界面
import matplotlib.pyplot as plt
plt.ioff() # 关闭交互
#plt.show = lambda *args, **kwargs: None # 禁止弹窗
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
from torch.utils.data import DataLoader
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
    ScheduleConfig,
    compute_project_investment_borrow_cap,
    effective_notional,
    filter_borrowers_for_rollover_block,
    rollover_borrow_quantity,
    schedule_config_from_mapping,
    settle_interbank_period,
)

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
DAILY_PROJECT_PD_DEFAULT = 0.00005
DAILY_PROJECT_PD_RANGE = (0.00002, 0.00010)
DAILY_PROJECT_MATURITY_DAYS = (1, 21)  # rng.integers high is exclusive; yields 1-20 steps
DAILY_PROJECT_SHOCK_MEAN_BULL = -0.00002
DAILY_PROJECT_SHOCK_STD_BULL = 0.00016
DAILY_PROJECT_SHOCK_MEAN_BEAR = -0.00012
DAILY_PROJECT_SHOCK_STD_BEAR = 0.00018
DAILY_PROJECT_REALIZED_CLIP = (-0.00045, 0.00055)
DAILY_PROJECT_RETURN_DEFAULT = 0.00008
DAILY_PROJECT_RISK_DEFAULT = 0.00016
DAILY_INTERBANK_ROLE_LCR_CUTOFF = 0.85
DAILY_INTERBANK_CONTRACT_MATURITY = 10
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
DAILY_ROLLOVER_SPREAD_SHORT = 0.00002
DAILY_ROLLOVER_SPREAD_LONG = 0.00008
DAILY_ROLLOVER_SPREAD = 0.00005
DAILY_HURDLE_RATE = 0.00012
DAILY_PENALTY_RATE_CEILING = 0.00035
DAILY_LIABILITY_GROWTH = 0.00002
DAILY_MARKET_ADJUSTMENT_BULL = (0.0002, 0.0008)
DAILY_MARKET_ADJUSTMENT_BEAR = (-0.0010, -0.0002)

@dataclass
class ProjectLoan:
    principal: float
    rate: float           # 每期利率
    maturity: int         # 期数
    age: int = 0
    pd: float = DAILY_PROJECT_PD_DEFAULT      # 每工作日违约概率
    lgd: float = 0.4      # 违约损失率


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
    schedule_type: str = "bullet"
    remaining_principal: float = 0.0
    coupon_rate: float = 0.0
    tenor_total: int = 1
    periods_paid: int = 0
    settlement_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.schedule_type == "installment" and self.remaining_principal <= 0.0:
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
        self, t: Trade, borrower: dict, cfg: ScheduleConfig
    ) -> tuple[Contract, dict]:
        from interbank_installment_rollover import choose_trade_schedule

        rollover_mode = str(getattr(cfg, "rollover_mode", "installment")).lower()
        if rollover_mode in ("off", "none", "false", "0", "disabled", "bullet"):
            dec = {
                "schedule_type": "bullet",
                "reason": "rollover_disabled",
                "tenor": None,
                "coupon_rate": None,
                "settlement_rate": float(t.rate),
                "maturity_in_periods": int(getattr(cfg, "bullet_maturity_periods", 1)),
            }
        else:
            dec = choose_trade_schedule(float(t.amount), float(t.rate), borrower, cfg)
        step = int(t.step_executed)
        if dec["schedule_type"] == "installment":
            tenor = int(dec["tenor"])
            c = Contract(
                contract_id=self._new_id(),
                lender_idx=t.lender_idx,
                borrower_idx=t.borrower_idx,
                principal=float(t.amount),
                rate=float(dec["settlement_rate"]),
                created_step=step,
                maturity_step=step + tenor,
                schedule_type="installment",
                remaining_principal=float(t.amount),
                coupon_rate=float(dec["coupon_rate"]),
                tenor_total=tenor,
                periods_paid=0,
                settlement_rate=float(dec["settlement_rate"]),
            )
        else:
            mat = int(dec["maturity_in_periods"])
            c = Contract(
                contract_id=self._new_id(),
                lender_idx=t.lender_idx,
                borrower_idx=t.borrower_idx,
                principal=float(t.amount),
                rate=float(dec["settlement_rate"]),
                created_step=step,
                maturity_step=step + mat,
                schedule_type="bullet",
                settlement_rate=float(dec["settlement_rate"]),
            )
        self.add_contract(c)
        return c, dec

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
):
    """统一设置实验开关，供模型脚本和 compare 脚本复用。"""
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
    return sim


def aggregate_contracts_to_exposure_matrix_at_step(book: ContractBook, n: int, current_step: int) -> np.ndarray:
    L = np.zeros((n, n), dtype=float)
    for c in book.contracts:
        if int(c.maturity_step) > int(current_step):
            P = effective_notional(c)
            L[c.lender_idx, c.borrower_idx] += P
            L[c.borrower_idx, c.lender_idx] -= P
    return L


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
    L: np.ndarray, banks: list, n: int, use_core: bool = False
) -> np.ndarray:
    """True 表示该银行在清算前流动性不足以覆盖应付。"""
    e = np.zeros(n, dtype=float)
    for i in range(min(n, len(banks))):
        b = banks[i]
        e[i] = (b["core_capital"] + b["liquid_assets"]) if use_core else b["liquid_assets"]
    p_bar = np.maximum(-L, 0.0).sum(axis=1)
    p_bar[p_bar < 1e-12] = 0.0
    shortfall = (p_bar > 0) & (e < p_bar * 0.999)
    return shortfall


def run_en_clearing_and_recovery(
    L: np.ndarray,
    banks: list,
    n: int,
    use_core: bool = False,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, list[int]]:
    """Eisenberg–Noe 清算；更新 liquid_assets，返回支付向量 p 与违约名单。"""
    eps = 1e-9
    Lbar = np.maximum(-L, 0.0)
    np.fill_diagonal(Lbar, 0.0)
    p_bar = Lbar.sum(axis=1)
    if p_bar.sum() <= eps:
        return np.zeros(n), []
    Pi = np.divide(Lbar, p_bar[:, None], out=np.zeros_like(Lbar), where=(p_bar[:, None] > 0))
    e = np.zeros(n, dtype=float)
    for i in range(min(n, len(banks))):
        b = banks[i]
        e[i] = (b["core_capital"] + b["liquid_assets"]) if use_core else b["liquid_assets"]
    p = p_bar.copy()
    for _ in range(max_iter):
        p_new = np.minimum(p_bar, Pi.T @ p + e)
        if np.max(np.abs(p_new - p)) < tol:
            break
        p = p_new
    failed = [i for i in range(n) if p[i] < p_bar[i] - 1e-6]
    recv = Pi.T @ p
    for i in range(min(n, len(banks))):
        banks[i]["liquid_assets"] = float(max(0.0, e[i] - p[i] + recv[i]))
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
        banks[i]["interbank_assets"] = float(assets[i])
        banks[i]["interbank_liabilities"] = float(liabilities[i])


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

    def _is_central(b):
        return (b.get("type") == "central") or (b.get("name") == "CentralBank")

    noncentral = [b for b in banks if not _is_central(b)]
    active_nc = [b for b in noncentral if b.get("is_active", True)]
    n_nc = max(1, len(noncentral))
    fr = sum(1 for b in noncentral if not b.get("is_active", True)) / n_nc
    low_cap_cnt = sum(1 for b in active_nc if float(b.get("capital_adequacy_ratio", 0.0)) < car_threshold)
    cbs = low_cap_cnt / max(1, len(active_nc))
    gap_num, cap_den = 0.0, 0.0
    for b in noncentral:
        ib = float(b.get("interbank_assets", 0.0))
        pa = float(b.get("investment", {}).get("projects", {}).get("amount", 0.0))
        rwa = 0.5 * ib + 1.0 * pa
        required = car_threshold * rwa
        actual = float(b.get("core_capital", 0.0))
        gap_num += max(0.0, required - actual)
        cap_den += (actual + float(b.get("total_assets", 0.0)))
    cgr = gap_num / (gap_num + cap_den + 1e-9)
    w1, w2, w3 = weights
    sr = w1 * fr + w2 * cbs + w3 * cgr
    return float(np.clip(sr, 0.0, 1.0))


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

MODEL_PATH = MODEL_DIR / "gnn_lstm_model.pth"        # 模型
DATA_FILE  = MODEL_DIR / "bank_contagion_data.json" # 数据集
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
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["bank_id"])
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / f"{prefix}_initial_bank_data.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


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
    broad_injection_ratio: float
    facility_spread: float
    note: str = ""


class BankNetworkSimulator:
    def __init__(self, num_banks=30, max_steps=5, B=1200, sigma=0.3, free_market=False, seed: int | None = DEFAULT_RANDOM_SEED):
        self.seed = DEFAULT_RANDOM_SEED if seed is None else int(seed)
        set_random_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
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
        self.clear_tol = 1e-3
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
    def _safe_car_value(self, core_capital: float, interbank_assets: float, projects_amt: float) -> float:
        """
        Paper-aligned CAR:
        RWA = 0.5 * interbank_assets + 1.0 * projects_amt
        CAR = K / (RWA + eps)
        """
        eps = 1e-9
        rwa = 0.5 * float(interbank_assets) + 1.0 * float(projects_amt)
        if rwa <= 0.0:
            return 0.0
        return float(core_capital) / (rwa + eps)

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
            "bullet_maturity_periods": getattr(self, "interbank_contract_maturity", 1),
            "bullet_max_principal": getattr(self, "bullet_max_principal", 250_000.0),
            "installment_min_principal": getattr(self, "installment_min_principal", 400_000.0),
            "lcr_installment_cutoff": getattr(self, "lcr_installment_cutoff", 1.0),
            "rollover_min_tenor": getattr(self, "rollover_min_tenor", 20),
            "rollover_max_tenor": getattr(self, "rollover_max_tenor", 120),
            "rollover_ref_small": getattr(self, "rollover_ref_small", 50_000.0),
            "rollover_ref_large": getattr(self, "rollover_ref_large", 2_000_000.0),
            "rollover_spread_short": getattr(self, "rollover_spread_short", DAILY_ROLLOVER_SPREAD_SHORT),
            "rollover_spread_long": getattr(self, "rollover_spread_long", DAILY_ROLLOVER_SPREAD_LONG),
            "rollover_mode": getattr(self, "rollover_mode", "installment"),
        })

    def _settle_interbank_installment_period(self, step: int) -> list[int]:
        """分期 rollover（20–120 个工作日、先息后本）；bullet 到期后可续借为分期。"""
        step = int(step)
        n = self.num_banks
        corridor = getattr(self, "central_corridor", None) or CentralBankCorridor(
            deposit_rate=max(0.0, float(self.base_rate) - DAILY_CB_DEPOSIT_SPREAD),
            lending_rate=float(self.base_rate) + DAILY_CB_LENDING_SPREAD,
            base_rate=float(self.base_rate),
        )

        def _issue(i: int, amt: float, st: int) -> None:
            self._issue_central_bank_liquidity_support(
                i, amt, st, rate=corridor.lending_rate, tenor=1, kind="settlement_backstop",
            )

        settle_result = settle_interbank_period(
            self.contract_book,
            self.banks,
            n,
            step,
            cfg=self._schedule_cfg(),
            liquidity_default_candidates=liquidity_default_candidates,
            run_en_clearing_and_recovery=run_en_clearing_and_recovery,
            issue_liquidity_support=_issue,
            corridor_lending_rate=float(corridor.lending_rate),
            use_core=False,
            clear_max_iter=int(getattr(self, "clear_max_iter", 100)),
            clear_tol=float(getattr(self, "clear_tol", 1e-6)),
            verbose_rollover=bool(getattr(self, "verbose_rollover", True)),
        )
        from interbank_installment_rollover import (
            active_installment_rollover_borrowers,
            log_rollover_borrow_policy_note,
        )

        self.rollover_coupon_due_borrowers = set(settle_result.coupon_due_borrowers)
        self.rollover_coupon_cleared_borrowers = set(settle_result.coupon_cleared_borrowers)
        self.rollover_blocked_borrowers = active_installment_rollover_borrowers(
            self.contract_book, step
        )
        log_rollover_borrow_policy_note(
            step,
            getattr(self, "rollover_borrow_policy", ROLLOVER_BORROW_COUPON_CLEARED),
            verbose=bool(getattr(self, "verbose_rollover", True)),
        )
        failed = settle_result.failed

        for i in failed:
            if 0 < i < len(self.banks):
                self.banks[i]["is_active"] = False
                self.banks[i]["liquid_assets"] *= 0.8

        self._refresh_exposure_from_contract_book(step)
        return failed

    def _settle_due_interbank_contracts(self, step: int) -> list[int]:
        """到期/分期结算（兼容旧调用名）。"""
        return self._settle_interbank_installment_period(step)


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
        self.borrowed_cash[:] = 0.0
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

            # 负债绝对值
            deposits            = core * dep_ratio
            interbank_borrowing = core * ib_ratio
            wholesale_funding   = core * wf_ratio
            lia = deposits + interbank_borrowing + wholesale_funding

            # ===== FIX: 负债不是“全额现金” =====
            # 只把一小部分留在 liquid_assets，其余默认已经配置成非流动资产（项目/贷款等）
            # 建议 0.10~0.30 之间调参；越小 => demand 越大、supply 越不离谱
            LIQ_FROM_LIA_FRAC = 0.10

            liq += LIQ_FROM_LIA_FRAC * lia

            # 把剩余资金当作“已投出去的资产”（你模型里最接近的是 projects）
            # 这样资产负债表不会凭空少一大块
            pre_alloc = (1.0 - LIQ_FROM_LIA_FRAC) * lia

            # 初始同业资产置 0（边稍后生成）
            interbank_assets0 = 0.0

            # 先把预配置资金计入 projects（后面你本来就用 projects_amt 做 CAR / SR）
            # 注意：下面 projects_amt 你后面会再用 invest 覆盖/叠加，所以这里用 +=
            projects_amt = 0.0
            projects_amt += pre_alloc


            # 投资决策：仅项目
            assets = liq + interbank_assets0
            if inv_ret > loan_rt:
                if t == 'central':
                    invest = assets * random.uniform(0.05, 0.10)
                elif t == 'shadow':
                    invest = assets * random.uniform(0.20, 0.30) * (1 + risk_app)
                else:
                    invest = assets * random.uniform(0.10, 0.20) * (1 + 0.5 * risk_app)
                invest *= (1.1 if self.market_environment == 'bull' else 0.7)
            else:
                invest = 0.0
            projects_amt += invest


            # 监管约束（RWA: 同业50% + 项目100%）
            rwa_init = 0.5 * interbank_assets0 + 1.0 * projects_amt
            leverage = (liq + interbank_assets0 + projects_amt) / (core + 1e-9)
            if core / (rwa_init + 1e-9) < 0.08 or leverage > 10:
                projects_amt *= 0.6

            total_assets = core + liq + interbank_assets0 + projects_amt
            # ===== project_book init  =====
            self.project_book[i].clear()

            if projects_amt > 1e-8:
                initial_project_maturity = int(self.rng.integers(*DAILY_PROJECT_MATURITY_DAYS))
                initial_project_age = int((i * 3) % max(1, initial_project_maturity))
                if i == 0:
                    self.project_book[i].append(ProjectLoan(
                        principal=float(projects_amt),
                        rate=0.0,
                        maturity=initial_project_maturity,
                        age=initial_project_age,
                        pd=0.0,
                        lgd=0.0,
                    ))
                else:
                    self.project_book[i].append(ProjectLoan(
                        principal=float(projects_amt),
                        rate=float(self.long_term_rate + DAILY_PROJECT_SPREAD),
                        maturity=initial_project_maturity,
                        age=initial_project_age,
                        pd=DAILY_PROJECT_PD_DEFAULT,
                        lgd=0.4,
                    ))
            else:
                # projects_amt 很小就不建项目；保持为空即可
                pass

            solv_ratio = (core + liq) / (lia + 1e-9)
            bank = {
                "id": i,
                "name": self.bank_names[i],
                "type": t,
                "is_active": True,
                "core_capital": core,
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
                "solvency_ratio": solv_ratio,
                "capital_ratio_history": [solv_ratio],
                "risk_appetite": risk_app,
                "market_volatility": vol,
                "proj_mu": float(inv_ret),
                "proj_sigma": float(
                    self.rng.uniform(0.00010, 0.00020)
                    if self.market_environment == "bull"
                    else self.rng.uniform(0.00014, 0.00026)
                ),
                "loan_interest_rate": loan_rt,
                "investment_interest_rate": self.long_term_rate + random.uniform(*DAILY_INVESTMENT_SPREAD_INIT),
                "outflow_rate": 0.2 if t == "central" else random.uniform(0.3, 0.5),
                "investment": {"projects": {"amount": projects_amt, "risk_weight": 1.0}},
                "capital_adequacy_ratio": self._safe_car_value(core, interbank_assets0, projects_amt),
                "liquidity_coverage_ratio": liq / (lia * (0.2 if t == "central" else 0.4) + 1e-9),
                "leverage_ratio": core / (liq + interbank_assets0 + 1e-9),
                "pending_endowment": 0.0,
                "hurdle_rate": DAILY_HURDLE_RATE,
                "defaulted": False,
                "chi_role": 0,
                "reservation_rate": loan_rt,
                "demand": 0.0,
                "supply": 0.0,
            }
            self.banks.append(bank)

        # —— 同业矩阵置零 -> 分配角色 -> 撮合生成边 —— 
        self.exposure_matrix = np.zeros((self.num_banks, self.num_banks), dtype=float)
        self.contract_book = ContractBook()
        self.current_step = 0
        roles0 = (
            self.assign_roles_by_risk(
                car_cutoff=getattr(self, "car_cutoff", 0.08),
                lcr_cutoff=getattr(self, "lcr_cutoff", DAILY_INTERBANK_ROLE_LCR_CUTOFF),
            )
            if hasattr(self, 'assign_roles_by_risk') else self.assign_roles()
        )

        self._sparse_bipartite_update(roles0)
        self.roles = roles0

        # —— 生成边后回填同业科目与监管指标 —— #
        L = np.asarray(self.exposure_matrix, dtype=float)
        for idx, b in enumerate(self.banks):
            ib_assets = float(np.maximum(L[idx], 0.0).sum())       # 对外拆出
            ib_liabs  = float(np.maximum(-L[idx], 0.0).sum())   # 对外拆入
            b['interbank_assets']      = ib_assets
            b['interbank_liabilities'] = ib_liabs
            rwa = 0.5 * ib_assets + 1.0 * b['investment']['projects']['amount']
            b['capital_adequacy_ratio'] = b['core_capital'] / (rwa + 1e-9)
            b['leverage_ratio'] = b['core_capital'] / (b['liquid_assets'] + b['interbank_assets'] + 1e-9)
            b['total_assets'] = (
                b['core_capital'] + b['liquid_assets'] + b['interbank_assets']
                + b['investment']['projects']['amount']
            )

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
        self.policy_enabled = True
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
        self.policy_lcr_target = 1.00
        self.policy_car_floor = 0.06
        self.cb_loan_tenor = 2
        self.cb_penalty_spread = DAILY_CB_PENALTY_SPREAD
        self.cb_max_support_share = 0.10
        self.cb_broad_support_share = 0.01
        self.cb_total_budget = 5000.0
        self.cb_remaining_budget = self.cb_total_budget
        self.cb_step_budget = 500.0
        self.cb_step_budget_remaining = self.cb_step_budget
        self.cb_total_injected = 0.0
        self.solvency_support_enabled = True
        self.solvency_support_car_trigger = 0.08
        self.solvency_support_car_floor = 0.03
        self.solvency_support_equity_floor = -0.02
        self.solvency_support_spread = DAILY_SOLVENCY_SUPPORT_SPREAD
        self.solvency_support_max_share = 0.08
        self.solvency_support_tenor = 4
        self.solvency_support_budget = 3000.0
        self.solvency_support_remaining_budget = self.solvency_support_budget
        self.solvency_support_step_budget = 250.0
        self.solvency_support_step_remaining = self.solvency_support_step_budget
        self.policy_support_book = []
        self.policy_support_book_outstanding_limit = self.cb_total_budget
        self.policy_support_cooldown_steps = 4
        self.policy_support_last_step_by_bank: dict[int, int] = {}
        self.policy_history = []
        self.policy_event_log = []
        self.export_policy_logs = getattr(self, "export_policy_logs", True)
        self.last_policy_note = "policy_init"
        self.initial_state_export_prefix = "centralized_central_policy"
        self.rollover_blocked_borrowers: set[int] = set()
        self.rollover_borrow_policy = ROLLOVER_BORROW_COUPON_CLEARED
        self.rollover_coupon_cleared_borrowers: set[int] = set()
        self.rollover_coupon_due_borrowers: set[int] = set()
        self.rollover_mode = "installment"
        self.schedule_selection = "auto"
        self.bullet_max_principal = 250_000.0
        self.installment_min_principal = 400_000.0
        self.lcr_installment_cutoff = 1.0
        self.rollover_spread_short = DAILY_ROLLOVER_SPREAD_SHORT
        self.rollover_spread_long = DAILY_ROLLOVER_SPREAD_LONG
        self.rollover_spread = DAILY_ROLLOVER_SPREAD
        self.rollover_min_tenor = 20
        self.rollover_max_tenor = 120
        self.rollover_ref_small = 50_000.0
        self.rollover_ref_large = 2_000_000.0
        self.interbank_contract_maturity = DAILY_INTERBANK_CONTRACT_MATURITY
        self.verbose_matching = getattr(self, "verbose_matching", True)
        self.verbose_rollover = getattr(self, "verbose_rollover", True)
        self.trade_schedule_log: list[dict] = []
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
        """
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

        base0 = getattr(self, "initial_base_rate", DAILY_BULL_BASE_RATE)
        delta *= DAILY_POLICY_RATE_MAX_STEP_CHANGE
        self.base_rate = float(np.clip(base0 + delta, DAILY_POLICY_RATE_FLOOR, DAILY_POLICY_RATE_CEILING))

    def _bank_lcr(self, bank) -> float:
        liq = float(bank.get("liquid_assets", 0.0))
        lia = float(bank.get("current_liabilities", 0.0))
        outflow = lia * float(bank.get("outflow_rate", 0.4))
        return liq / (outflow + 1e-9)

    def _bank_equity(self, bank) -> float:
        return float(bank.get("core_capital", 0.0)) + float(bank.get("liquid_assets", 0.0)) - float(bank.get("current_liabilities", 0.0))

    def _is_liquidity_support_target(self, bank) -> bool:
        if not bank.get("is_active", True):
            return False
        lia = float(bank.get("current_liabilities", 0.0))
        liq = float(bank.get("liquid_assets", 0.0))
        outflow = lia * float(bank.get("outflow_rate", 0.4))
        required_liq = float(getattr(self, "policy_lcr_target", 1.0)) * outflow
        if required_liq - liq <= 1e-9 and self._bank_lcr(bank) >= self.policy_lcr_target:
            return False
        if self._bank_equity(bank) <= 0.0:
            return False
        if float(bank.get("capital_adequacy_ratio", 0.0)) < self.policy_car_floor:
            return False
        return True

    def _is_capital_support_target(self, bank) -> bool:
        if not bank.get("is_active", True):
            return False
        lia = float(bank.get("current_liabilities", 0.0))
        if lia <= 1e-9:
            return False
        car = float(bank.get("capital_adequacy_ratio", 0.0))
        if not (self.solvency_support_car_floor < car < self.solvency_support_car_trigger):
            return False
        equity_ratio = self._bank_equity(bank) / (lia + 1e-9)
        if equity_ratio < self.solvency_support_equity_floor:
            return False
        return True

    def _run_central_bank_policy_cycle(self, step: int) -> None:
        policy_obs = self._observe_central_bank_conditions()
        policy_action = self._decide_central_bank_policy(policy_obs)
        self._apply_central_bank_policy(policy_action, int(step))

    def _record_systemic_risk(self, risk: float) -> None:
        """保存上一仿真阶段算出的 SR，供下一步央行规则使用。"""
        self.last_systemic_risk = float(np.clip(float(risk), 0.0, 1.0))

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
                broad_injection_ratio=0.015,
                facility_spread=DAILY_FACILITY_SPREAD_CRISIS,
                note="crisis_easing",
            )
        if sr >= sr_defensive:
            return CentralBankPolicyAction(
                policy_rate=max(self.policy_rate_floor, base_target - DAILY_POLICY_EASING_DEFENSIVE),
                reserve_requirement=max(0.0075, reserve_target - 0.005),
                liquidity_support_ratio=0.12,
                broad_injection_ratio=0.005,
                facility_spread=DAILY_FACILITY_SPREAD_DEFENSIVE,
                note="defensive_easing",
            )
        return CentralBankPolicyAction(
            policy_rate=float(np.clip(base_target, self.policy_rate_floor, self.policy_rate_ceiling)),
            reserve_requirement=reserve_target,
            liquidity_support_ratio=0.04,
            broad_injection_ratio=0.0,
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
            payment = min(float(bank.get("liquid_assets", 0.0)), due)
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
        last_support_step = self.policy_support_last_step_by_bank.get(int(bank_idx))
        if last_support_step is not None:
            cooldown = int(getattr(self, "policy_support_cooldown_steps", 0))
            if int(step) - int(last_support_step) < cooldown:
                return 0.0
        per_bank_cap = per_bank_cap_share * float(bank.get("current_liabilities", 0.0))
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
        self.policy_support_last_step_by_bank[int(bank_idx)] = int(step)
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
        solvency_support_total = 0.0
        solvency_supported_banks = 0
        facility_rate = self.base_rate + float(action.facility_spread)
        for i in range(1, len(self.banks)):
            bank = self.banks[i]
            if not self._is_liquidity_support_target(bank):
                continue
            lia = float(bank.get("current_liabilities", 0.0))
            liq = float(bank.get("liquid_assets", 0.0))
            outflow = lia * float(bank.get("outflow_rate", 0.4))
            required_liq = max(reserve_target * lia, outflow)
            gap = max(0.0, required_liq - liq)
            if gap <= 1e-9:
                if action.broad_injection_ratio > 0.0 and self._bank_lcr(bank) < 1.15:
                    gap = action.broad_injection_ratio * lia
                else:
                    continue
            if self._bank_equity(bank) <= 0.0 or float(bank.get("capital_adequacy_ratio", 0.0)) < self.policy_car_floor:
                continue
            cap = max(self.cb_max_support_share, float(action.liquidity_support_ratio)) * lia
            amount = min(max(gap * 1.05, action.broad_injection_ratio * lia), cap)
            injected = self._issue_central_bank_liquidity_support(
                i,
                amount,
                step,
                rate=facility_rate,
                tenor=self.cb_loan_tenor,
                kind="liquidity_window",
            )
            if injected > 0.0:
                support_total += injected
                supported_banks += 1

        if getattr(self, "solvency_support_enabled", False):
            support_rate = self.base_rate + float(self.solvency_support_spread)
            for i in range(1, len(self.banks)):
                bank = self.banks[i]
                if not self._is_capital_support_target(bank):
                    continue
                lia = float(bank.get("current_liabilities", 0.0))
                ib = float(bank.get("interbank_assets", 0.0))
                pa = float(bank["investment"]["projects"]["amount"])
                rwa = 0.5 * ib + 1.0 * pa
                capital_gap = max(0.0, self.solvency_support_car_trigger * rwa - float(bank.get("core_capital", 0.0)))
                if capital_gap <= 1e-9:
                    continue
                amount = min(capital_gap, self.solvency_support_max_share * lia)
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

        self.last_policy_note = action.note
        self.policy_history.append({
            "step": int(step),
            "systemic_risk": float(getattr(self, "last_systemic_risk", 0.0)),
            "policy_rate": self.base_rate,
            "reserve_requirement": reserve_target,
            "support_total": support_total,
            "supported_banks": supported_banks,
            "solvency_support_total": solvency_support_total,
            "solvency_supported_banks": solvency_supported_banks,
            "remaining_budget": float(self.cb_remaining_budget),
            "step_budget_remaining": float(self.cb_step_budget_remaining),
            "solvency_remaining_budget": float(self.solvency_support_remaining_budget),
            "solvency_step_remaining": float(self.solvency_support_step_remaining),
            "note": action.note,
        })
        if getattr(self, "export_policy_logs", True):
            export_policy_logs_excel(
                self.policy_history,
                self.policy_event_log,
                POLICY_LOG_DIR,
                self.initial_state_export_prefix,
            )

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

            car  = float(b.get("capital_adequacy_ratio", 0.0))
            lcr  = float(b.get("liquidity_coverage_ratio", 1.0))
            solv = float(b.get("solvency_ratio", 1.0))

            need = 0.0

            # CAR 低：不适合放贷
            car_low = (car < car_cutoff)

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

            if need >= 0.55:
                roles[i] = -1
            else:
                if car_low:
                    roles[i] = -1 if (lcr < lcr_cutoff or solv < 1.0 or liq < target_liq) else 0
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
            roles[strongest] = +1


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
                if r != 0 and np.isfinite(avail_liq[r]) and avail_liq[r] > 1e-8:
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

    def simulate_step(self, step):
        try:
            self.current_step = int(step)
            if getattr(self, "exposure_matrix", None) is not None:
                self.prev_exposure_matrix = self.exposure_matrix.copy()
            self._reset_policy_step_budget()
            self._run_central_bank_policy_cycle(step)
            self._settle_central_bank_loans(step)
            self._settle_due_interbank_contracts(step)

            # A) 利率/环境
            if hasattr(self, "adjust_base_rate"):
                self.adjust_base_rate()

            # 结转上期挂账
            for _b in self.banks:
                if 'pending_endowment' in _b:
                    _b['liquid_assets'] += _b.pop('pending_endowment')

            # 市场环境演化
            self.market_duration += 1
            if self.market_duration >= self.market_duration_limit:
                self.prev_market_environment = self.market_environment
                self.market_environment = 'bull' if random.random() < 0.6 else 'bear'
                self.market_duration = 0
                self.market_duration_limit = random.randint(2, 5)

            # 环境对利率/波动的影响
            if self.market_environment == 'bull':
                self.base_rate = max(DAILY_POLICY_RATE_FLOOR, self.base_rate + random.uniform(-0.00001, 0.00001))
                self.long_term_rate = self.base_rate + random.uniform(*DAILY_LONG_RATE_SPREAD_BULL)
                market_volatility = random.uniform(10, 20) / 50
                market_adjustment = random.uniform(*DAILY_MARKET_ADJUSTMENT_BULL)
            else:
                self.base_rate = min(DAILY_POLICY_RATE_CEILING, self.base_rate + random.uniform(0.0, 0.000015))
                self.long_term_rate = self.base_rate + random.uniform(*DAILY_LONG_RATE_SPREAD_BEAR)
                market_volatility = random.uniform(30, 50) / 50
                market_adjustment = random.uniform(*DAILY_MARKET_ADJUSTMENT_BEAR)

            # B) 银行逐家处理（仅项目投资）
            for i, bank in enumerate(self.banks):
                bank['market_volatility'] = market_volatility
                bank['loan_interest_rate'] = (
                    self.base_rate + random.uniform(*DAILY_LOAN_SPREAD_BULL)
                    if self.market_environment == 'bull'
                    else self.base_rate + random.uniform(*DAILY_LOAN_SPREAD_BEAR)
                )
                bank['investment_interest_rate'] = self.long_term_rate + random.uniform(*DAILY_INVESTMENT_SPREAD_STEP)

                bank.setdefault('risk_appetite', 0.5)
                bank.setdefault('hurdle_rate', DAILY_HURDLE_RATE)
                bank.setdefault('pending_endowment', 0.0)

                

                bank['current_liabilities'] += DAILY_LIABILITY_GROWTH * bank['current_liabilities']
                bank['liquid_assets'] *= (1 + market_adjustment)

                bank['outflow_rate'] = (
                    0.2 if bank['type'] == 'central'
                    else (
                        random.uniform(0.4, 0.6) if self.market_environment == 'bear'
                        else random.uniform(0.3, 0.5)
                    )
                )

                if bank['liquid_assets'] < bank['current_liabilities'] * bank['outflow_rate']:
                    bank['risk_appetite'] *= 0.9

            # C) 外生冲击 / 政策注入 / 随机失败
            if random.random() < 0.05:
                for i, bank in enumerate(self.banks):
                    bank['liquid_assets'] *= 0.9
                    for loan in self.project_book[i]:
                        loan.pd = min(1.0, loan.pd * 1.2)

            if self.market_environment == 'bear' and random.random() < 0.1:
                for bank in self.banks:
                    bank['liquid_assets'] += 0.01 * bank['current_liabilities']

            if (not getattr(self, "one_shot_default_done", False)) and (int(step) == 0):
                if random.random() < 0.05:
                    fail_bank = random.randint(1, self.num_banks - 1)
                    self.banks[fail_bank]['is_active'] = False
                    self.banks[fail_bank]['liquid_assets'] *= 0.85
                self.one_shot_default_done = True

            # E) 分配角色
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

            # F) 同业撮合（带现金结算与准备金约束）
            self._sparse_bipartite_update(self.roles)
            for i in range(self.num_banks):
                if self.roles[i] == +1 and self.banks[i].get("is_active", True):
                    self.invest_free_cash_into_projects(i, invest_frac=0.05)

            # G) 借入资金落地到项目/准备金 & 更新项目台账
            for i in range(self.num_banks):
                self.allocate_borrowed_to_projects(i)
                self.update_project_book(i)

            # H) 指标更新：同业科目与 ContractBook 同步后再算 CAR/LCR 等
            n_b = self.num_banks
            update_bank_states_from_contract_book(
                self.banks, self.contract_book, n_b, int(step)
            )
            for idx, b in enumerate(self.banks):
                ib_assets = float(b.get("interbank_assets", 0.0))
                ib_liabs = float(b.get("interbank_liabilities", 0.0))

                cap_ratio = (b['core_capital'] + b['liquid_assets']) / (b['current_liabilities'] + 1e-9)
                b['solvency_ratio'] = cap_ratio

                proj_amt = b['investment']['projects']['amount']
                b['capital_adequacy_ratio'] = self._safe_car_value(
                    b['core_capital'], ib_assets, proj_amt
                )

                b['liquidity_coverage_ratio'] = b['liquid_assets'] / (
                    b['current_liabilities'] * b['outflow_rate'] + 1e-9
                )
                b['leverage_ratio'] = b['core_capital'] / (
                    b['liquid_assets'] + b['interbank_assets'] + 1e-9
                )
                b['capital_ratio_history'].append(cap_ratio)

            # D) 偿付能力违约（在合约簿口径更新之后，避免误判）
            for i in range(self.num_banks):
                if self.bank_types[i] == 'central':
                    continue
                b = self.banks[i]
                equity = self._bank_equity(b)
                if equity < 0:
                    b['is_active'] = False
                    b['liquid_assets'] *= 0.8

            # I) 风险度量 &（可选）记录
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
                    'policy_note': getattr(self, "last_policy_note", ""),
                    'exposure_matrix': self.exposure_matrix.copy(),
                    'bank_states': [deepcopy(b) for b in self.banks],
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
            return risk

        except Exception as e:
            print(f"Error in simulate_step: {e}")
            raise

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

def _sparse_bipartite_update(self, roles: np.ndarray) -> None:
    """
    重新生成稀疏双边同业网络：
    - exposure_matrix 是 ContractBook 中未到期合约的当期快照
    - 新成交写入 ContractBook，再重新聚合 exposure_matrix
    """
    B_base = float(getattr(self, "B", 3000.0))
    n = self.num_banks
    eps = 1e-8

    step = int(getattr(self, "current_step", 0))
    self._refresh_exposure_from_contract_book(step)

    # 根据未到期合约快照初始化度数
    deg_init = np.count_nonzero(np.abs(self.exposure_matrix) > eps, axis=1).astype(int)

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

    if len(lenders) == 0 or len(borrowers) == 0:
        print(f"[debug] No matching: lenders={len(lenders)}, borrowers={len(borrowers)}")
        return
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

    # ===== 4) 撮合方式：论文对齐版（GNN 生成 a_ij）；全图 + 排序配给 =====
    # matcher 每步输入为 to_pyg_graph(use_prev=True) 的全体银行图；成交在 supply/demand/度数约束下按权重排序分配。

    ctx = getattr(self, "gnn_context", None)
    B_base = float(getattr(self, "B", 3000.0))

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
            # 每步全图：全体银行节点 + 上期 exposure 正边，再对可行 (i,j) 批量 score_pairs
            base_graph = self.to_pyg_graph(use_prev=True)
            aij = matcher.score_pairs(base_graph, feasible_pairs, device=device)  # (K,)

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
        if (
            borrower_idx in rollover_blocked
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

        self.banks[lender_idx]["liquid_assets"] -= amt
        self.banks[borrower_idx]["liquid_assets"] = float(
            self.banks[borrower_idx].get("liquid_assets", 0.0)
        ) + amt
        self.borrowed_cash[borrower_idx] += amt
        rate = float(self.banks[lender_idx].get("loan_interest_rate", getattr(self, "base_rate", DAILY_BULL_BASE_RATE)))
        trade = Trade(
            lender_idx=int(lender_idx),
            borrower_idx=int(borrower_idx),
            amount=float(amt),
            rate=rate,
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

    print(
        f"[debug] Matching finished: edges={edge_count}, "
        f"lenders={len(lenders)}, borrowers={len(borrowers)}, "
        f"total_supply={total_supply:.2f}, total_demand={total_demand:.2f}, "
        f"actual_lent={actual_lent:.2f}, B_eff={float(getattr(self,'B',300.0)):.2f}"
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
    B   = getattr(self, "B", 300.0)
    n = self.exposure_matrix.shape[0]
    rng = getattr(self, "rng", np.random.default_rng(DEFAULT_RANDOM_SEED))
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

def calculate_systemic_risk(self, weights=(0.5, 0.3, 0.2), car_threshold=0.08,return_parts=False):
    """
    SR_t = w1 * FR_t + w2 * CBS_t + w3 * CGR_t
      FR_t  : Failure Rate (exclude central bank)
      CBS_t : Low-CAR share among active non-central banks
      CGR_t : Capital gap ratio among non-central banks
    """
    banks = self.banks

    # ---- exclude central bank (robust: by type/name) ----
    def _is_central(b):
        return (b.get("type") == "central") or (b.get("name") == "CentralBank")

    noncentral = [b for b in banks if not _is_central(b)]
    active_nc  = [b for b in noncentral if b.get("is_active", True)]

    n_nc = len(noncentral)

    # FR (non-central only)
    fr = sum(1 for b in noncentral if not b.get("is_active", True)) / max(1, n_nc)

    # CBS (active non-central only)
    low_cap_cnt = 0.0
    for b in active_nc:
        ib = float(b.get("interbank_assets", 0.0))
        pa = float(b["investment"]["projects"]["amount"])
        car = float(b.get("capital_adequacy_ratio", 0.0))
        if car < car_threshold:
            low_cap_cnt += 1.0
    cbs = low_cap_cnt / max(1, len(active_nc))

    # CGR (non-central only)
    gap_num, cap_den = 0.0, 0.0
    for b in noncentral:
        ib = float(b.get("interbank_assets", 0.0))
        pa = float(b["investment"]["projects"]["amount"])
        rwa = 0.5 * ib + 1.0 * pa
        required = car_threshold * rwa
        actual = float(b["core_capital"])
        gap_num += max(0.0, required - actual)
        cap_den += (actual + float(b.get("total_assets", 0.0)))

    cgr = gap_num / (gap_num + cap_den + 1e-9)

    w1, w2, w3 = weights
    sr = w1 * fr + w2 * cbs + w3 * cgr
    sr = float(np.clip(sr, 0.0, 1.0)) 
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
    if invest_frac is None:
        invest_frac = 0.10 if getattr(self, "free_market", False) else 0.30

    bank = self.banks[i]
    liq = float(bank['liquid_assets'])
    avail = max(0.0, liq - self.reserve_buffer[i] * float(bank['current_liabilities']))
    invest = min(avail, liq * float(invest_frac))
    if invest <= 1e-8:
        return

    bank['liquid_assets'] -= invest

    rng = getattr(self, "rng", np.random.default_rng(DEFAULT_RANDOM_SEED))
    loan = ProjectLoan(
        principal=float(invest),
        rate=float(self.long_term_rate + DAILY_PROJECT_SPREAD),
        maturity=int(rng.integers(*DAILY_PROJECT_MATURITY_DAYS)),
        pd=float(rng.uniform(*DAILY_PROJECT_PD_RANGE)),
        lgd=float(rng.uniform(0.30, 0.60)),
    )
    self.project_book[i].append(loan)
    bank['investment']['projects']['amount'] += invest



def allocate_borrowed_to_projects(self, i, spread: float = DAILY_PROJECT_SPREAD):
    """借入现金已入账；这里只把项目份额从 liquid 转入项目台账。"""
    budget = float(self.borrowed_cash[i])
    if budget <= 1e-8:
        return

    target_proj = budget * self.project_min_share
    num = max(1, int(target_proj // 5e5))
    per = target_proj / num if num > 0 else 0.0

    self.banks[i]["liquid_assets"] = float(self.banks[i].get("liquid_assets", 0.0)) - float(target_proj)
    rng = getattr(self, "rng", np.random.default_rng(DEFAULT_RANDOM_SEED))
    for _ in range(num):
        loan = ProjectLoan(
            principal=float(per),
            rate=float(self.long_term_rate + spread),
            maturity=int(rng.integers(*DAILY_PROJECT_MATURITY_DAYS)),
            pd=float(rng.uniform(*DAILY_PROJECT_PD_RANGE)),
            lgd=float(rng.uniform(0.30, 0.60)),
        )
        self.project_book[i].append(loan)
        self.banks[i]['investment']['projects']['amount'] += float(per)

    self.borrowed_cash[i] = 0.0

def _daily_project_shock_params(market_environment: str) -> tuple[float, float]:
    if market_environment == "bull":
        return DAILY_PROJECT_SHOCK_MEAN_BULL, DAILY_PROJECT_SHOCK_STD_BULL
    return DAILY_PROJECT_SHOCK_MEAN_BEAR, DAILY_PROJECT_SHOCK_STD_BEAR


def update_project_book(self, i):
    """
    项目台账：每期计息 / 负收益 / 到期回款 / 违约扣损。
    """
    bank = self.banks[i]

    mu_shock, sigma_shock = _daily_project_shock_params(self.market_environment)
    clip_lo, clip_hi = DAILY_PROJECT_REALIZED_CLIP

    new_book = []
    projects_amt = 0.0

    for loan in self.project_book[i]:
        if random.random() < loan.pd:
            loss = loan.principal * loan.lgd
            bank['liquid_assets'] -= loss
            bank['core_capital'] = max(0.0, bank['core_capital'] - 0.5 * loss)
            continue

        rng = getattr(self, "rng", np.random.default_rng(DEFAULT_RANDOM_SEED))
        shock = rng.normal(mu_shock, sigma_shock)
        realized_r = float(np.clip(loan.rate + shock, clip_lo, clip_hi))
        cashflow = loan.principal * realized_r

        bank['liquid_assets'] += cashflow

        if realized_r < 0.0:
            loss = -cashflow
            bank['core_capital'] = max(0.0, bank['core_capital'] - 0.3 * loss)

        loan.age += 1
        if loan.age >= loan.maturity:
            bank['liquid_assets'] += loan.principal
        else:
            new_book.append(loan)
            projects_amt += loan.principal

    self.project_book[i] = new_book
    bank['investment']['projects']['amount'] = float(projects_amt)

    solv = (bank['core_capital'] + bank['liquid_assets']) - bank['current_liabilities']
    if i != 0 and solv < 0 and bank.get('is_active', True):
        bank['is_active'] = False
        bank['liquid_assets'] *= 0.8
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
    def __init__(self, num_simulations=1000, num_timesteps=5,
                 data_file=str(DATA_FILE)):
        self.num_simulations = num_simulations
        self.num_timesteps = num_timesteps
        self.data_file = data_file
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = []
        self.expected_features = 15
        self._generate_data()
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f)

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


def train_model(dataset, num_epochs=50, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNLSTMModel(input_dim=15).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: x
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=lambda x: x
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            graphs = [g.to(device) for seq in batch for g in seq]
            optimizer.zero_grad()
            out = model(graphs)
            targets = torch.stack([seq[-1].y for seq in batch]).to(device)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                graphs = [g.to(device) for seq in batch for g in seq]
                out = model(graphs)
                targets = torch.stack([seq[-1].y for seq in batch]).to(device)
                test_loss += criterion(out, targets).item()
        test_loss = test_loss / len(test_loader)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return model


def train_matcher_from_dataset(dataset, epochs=3, lr=1e-3, neg_ratio=1.0, batch_graphs=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = GNNPairMatcher(in_dim=15, hid=64).to(device)
    opt = torch.optim.Adam(matcher.parameters(), lr=float(lr), weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()


    # 把 dataset 展平成图列表
    graphs = []
    for seq in dataset:
        for g in seq:
            graphs.append(g)

    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)

    for ep in range(epochs):
        random.shuffle(graphs)
        total = 0.0
        cnt = 0

        for s in range(0, len(graphs), batch_graphs):
            batch = graphs[s:s+batch_graphs]
            loss_acc = 0.0

            opt.zero_grad()

            for g in batch:
                g = g.to(device)
                g.x = torch.nan_to_num(g.x, nan=0.0, posinf=5.0, neginf=-5.0)
                g.x = torch.clamp(g.x, -5.0, 5.0)
                N = g.x.size(0)

                # 正边
                ei = g.edge_index
                pos_pairs = []
                if ei is not None and ei.numel() > 0:
                    for k in range(ei.size(1)):
                        u = int(ei[0, k].item())
                        v = int(ei[1, k].item())
                        if u != v:
                            pos_pairs.append((u, v))

                if not pos_pairs:
                    continue

                # 负边：随机采样不存在的边
                pos_set = set(pos_pairs)
                num_neg = int(len(pos_pairs) * neg_ratio)
                neg_pairs = []
                while len(neg_pairs) < num_neg:
                    u = int(rng.integers(0, N))
                    v = int(rng.integers(0, N))
                    if u == v:
                        continue
                    if (u, v) in pos_set:
                        continue
                    neg_pairs.append((u, v))

                pairs = pos_pairs + neg_pairs
                y = torch.tensor([1.0]*len(pos_pairs) + [0.0]*len(neg_pairs), dtype=torch.float, device=device)

                # forward
                h = matcher.node_embed(g)
                ii = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
                jj = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
                hi, hj = h[ii], h[jj]
                feat = torch.cat([hi, hj, torch.abs(hi-hj), hi*hj], dim=1)
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
        print(f"[matcher] epoch {ep+1}/{epochs} loss={avg:.4f}")

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


@dataclass
class RecordedRun:
    seed: int
    theta_policy: float
    snapshots: list[StepSnapshot]
    rollover_enabled: bool = True
    policy_support_enabled: bool = True


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
    B: float = 300.0,
    sigma: float = 0.3,
    matcher=None,
    device=None,
    stop_on_network_stable: bool = False,
    rollover_enabled: bool = True,
    policy_support_enabled: bool = True,
) -> RecordedRun:
    T = min(int(T), MAX_PLOT_STEPS)
    sim = BankNetworkSimulator(num_banks=N, max_steps=T, B=B, sigma=sigma, seed=int(seed))
    configure_simulation_features(
        sim,
        rollover_enabled=rollover_enabled,
        policy_support_enabled=policy_support_enabled,
    )
    sim._save_network_snapshot = False
    sim.export_policy_logs = False
    sim.car_cutoff = float(theta_policy)
    sim.initialize_network()

    snapshots: list[StepSnapshot] = []
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
        snapshots.append(
            StepSnapshot(
                step=int(s),
                banks=[deepcopy(b) for b in sim.banks],
                exposure_matrix=np.array(sim.exposure_matrix, dtype=float, copy=True),
                policy_support_total=float(last_policy.get("support_total", 0.0)),
                solvency_support_total=float(last_policy.get("solvency_support_total", 0.0)),
            )
        )
        if stop_on_network_stable and sim.network_stable_step is not None:
            break
    return RecordedRun(
        seed=int(seed),
        theta_policy=float(theta_policy),
        snapshots=snapshots,
        rollover_enabled=bool(rollover_enabled),
        policy_support_enabled=bool(policy_support_enabled),
    )


def run_simulation_plot_batch(
    T: int = 800,
    nsim: int = 20,
    *,
    theta_policy: float = 0.08,
    N: int = 30,
    B: float = 300.0,
    sigma: float = 0.3,
    seed0: int = DEFAULT_RANDOM_SEED,
    matcher=None,
    device=None,
    stop_on_network_stable: bool = False,
    rollover_enabled: bool = True,
    policy_support_enabled: bool = True,
) -> SimulationPlotBatch:
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
                rollover_enabled=rollover_enabled,
                policy_support_enabled=policy_support_enabled,
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
        f"avg_steps={np.mean([len(r.snapshots) for r in runs]):.1f}"
    )
    return batch


def score_recorded_run(
    run: RecordedRun,
    weights=(0.5, 0.3, 0.2),
    theta_measure: float = 0.08,
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
    return {
        "sr": np.asarray(sr_list, dtype=float),
        "fr": np.asarray(fr_list, dtype=float),
        "cbs": np.asarray(cbs_list, dtype=float),
        "cgr": np.asarray(cgr_list, dtype=float),
    }


def score_batch_mean_components(
    batch: SimulationPlotBatch,
    weights=(0.5, 0.3, 0.2),
    theta_measure: float = 0.08,
) -> dict[str, np.ndarray]:
    sr_mat, fr_mat, cbs_mat, cgr_mat = [], [], [], []
    for run in batch.runs:
        comp = score_recorded_run(run, weights=weights, theta_measure=float(theta_measure))
        sr_mat.append(comp["sr"])
        fr_mat.append(comp["fr"])
        cbs_mat.append(comp["cbs"])
        cgr_mat.append(comp["cgr"])
    return {
        "sr": _nanmean_variable_length(sr_mat),
        "fr": _nanmean_variable_length(fr_mat),
        "cbs": _nanmean_variable_length(cbs_mat),
        "cgr": _nanmean_variable_length(cgr_mat),
    }


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

    w1_grid    = np.round(np.linspace(0.10, 0.90, 9), 2)
    theta_grid = np.round(np.linspace(0.01, 0.50, 10), 2)
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

    def run_series(weights, theta_measure, theta_policy=0.08, N=30, T=T_steps, B=300, sigma=0.3):
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
            if _scenario_stable_for_sweep(sim, current_sr, window=50, risk_tol=0.01):
                break

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
    banks = sim.banks

    def _is_central(b):
        return (b.get("type") == "central") or (b.get("name") == "CentralBank")

    noncentral = [b for b in banks if not _is_central(b)]
    active_nc  = [b for b in noncentral if b.get("is_active", True)]
    n_nc = len(noncentral)

    # --- FR: non-central failure rate ---
    fr = sum(1 for b in noncentral if not b.get("is_active", True)) / max(1, n_nc)

    # --- CBS: share of active non-central banks with CAR < theta ---
    low_cap = 0.0
    for b in active_nc:
        car = float(b.get("capital_adequacy_ratio", 0.0))
        low_cap += 1.0 if car < theta else 0.0
    cbs = low_cap / max(1, len(active_nc))

    # --- CGR: bounded gap intensity in [0,1) ---
    gap_num = 0.0
    cap_den = 0.0
    for b in noncentral:
        ib = float(b.get("interbank_assets", 0.0))
        pa = float(b.get("investment", {}).get("projects", {}).get("amount", 0.0))
        rwa = 0.5 * ib + 1.0 * pa

        required = theta * rwa
        actual   = float(b.get("core_capital", 0.0))

        gap = max(0.0, required - actual)
        gap_num += gap

        total_assets = float(b.get("total_assets", 0.0))
        cap_den += max(0.0, actual) + max(0.0, total_assets)

    cgr = gap_num / (gap_num + cap_den + 1e-9)   # 关键：有界，不会爆炸

    w1, w2, w3 = weights
    sr = w1 * fr + w2 * cbs + w3 * cgr            # 线性定义，本身就在[0,1]

    return float(sr), float(fr), float(cbs), float(cgr)


def _nanmean_variable_length(series_list):
    non_empty = [np.asarray(x, dtype=float) for x in series_list if len(x) > 0]
    if not non_empty:
        return np.asarray([], dtype=float)
    max_len = max(len(x) for x in non_empty)
    mat = np.full((len(non_empty), max_len), np.nan, dtype=float)
    for i, arr in enumerate(non_empty):
        mat[i, : len(arr)] = arr
    return np.nanmean(mat, axis=0)


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
    T=800,
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
        comp = score_batch_mean_components(batch, weights=weights, theta_measure=float(theta))
        sr_list = list(comp["sr"])
        fr_list = list(comp["fr"])
        cbs_list = list(comp["cbs"])
        cgr_list = list(comp["cgr"])
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
            if _scenario_stable_for_sweep(sim, sr, window=50, risk_tol=0.01):
                break

    xs = np.arange(1, len(sr_list) + 1)

    plt.close('all')
    fig = plt.figure(figsize=(12, 7))
    ax  = plt.gca()

    ax.plot(xs, sr_list,  lw=2.2, marker='o', ms=4, label='SR (Systemic Risk)')
    ax.plot(xs, fr_list,  lw=1.8, marker='.', ms=3, label='FR (Failure Rate)')
    ax.plot(xs, cbs_list, lw=1.8, marker='.', ms=3, label='CBS (Low-CAR Share)')
    ax.plot(xs, cgr_list, lw=1.8, marker='.', ms=3, label='CGR (Capital Gap Ratio)')

    ax.set_title(f"Baseline Trajectory — SR and Components over Time\nW={weights}, θ={theta}")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value (0–1)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.4)
    ax.legend()

    out = FIG_DIR / "baseline_trajectory.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_scenario_comparison(T=800, batch: SimulationPlotBatch | None = None):
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
            if _scenario_stable_for_sweep(sim, current_sr, window=50, risk_tol=0.01):
                break
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
    ax.set_xlabel("Time Step")
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
    B=300,
    sigma=0.3,
    nsim=20,
    seed0=DEFAULT_RANDOM_SEED,
    batch: SimulationPlotBatch | None = None,
):
    """
    跑 nsim 次，返回每期 SR/FR/CBS/CGR 的均值轨迹（长度 T）。
    theta_measure: 只影响“评分口径”（CBS/CGR 的阈值）
    theta_policy : 影响仿真过程（角色分配/网络生成用的 car_cutoff）
    batch: 若提供则只重算分量，不重跑仿真。
    """
    import numpy as np

    T = min(int(T), MAX_PLOT_STEPS)
    batch = _resolve_plot_batch(batch)
    if batch is not None:
        if float(theta_policy) != float(batch.theta_policy):
            print(
                f"[plot-batch] score: use recorded policy={batch.theta_policy:.2f} "
                f"(requested policy={float(theta_policy):.2f} ignored)"
            )
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
    )
    cached = _MEAN_COMPONENTS_CACHE.get(key)
    if cached is not None:
        return {k: np.asarray(v, dtype=float).copy() for k, v in cached.items()}

    sr_mat, fr_mat, cbs_mat, cgr_mat = [], [], [], []
    rng = np.random.default_rng(seed0)

    for k in range(nsim):
        sim_seed = int(rng.integers(0, 2**32 - 1))
        sim = BankNetworkSimulator(num_banks=N, max_steps=T, B=B, sigma=sigma, seed=sim_seed)

        # policy θ：影响行为/网络
        sim.car_cutoff = float(theta_policy)
        sim._save_network_snapshot = False
        sim.export_policy_logs = False

        sim.initialize_network()

        sr_list, fr_list, cbs_list, cgr_list = [], [], [], []
        for s in range(T):
            sim.simulate_step(s)

            # 用你现成的分解函数，口径 θ = theta_measure
            sr, fr, cbs, cgr = _components_from_state(
                sim, weights=weights, theta=float(theta_measure)
            )
            sr_list.append(sr); fr_list.append(fr); cbs_list.append(cbs); cgr_list.append(cgr)
            if _scenario_stable_for_sweep(sim, sr, window=50, risk_tol=0.01):
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
    T=800,
    weights=(0.5, 0.3, 0.2),
    nsim=20,
    theta_policy_fixed=None,
    theta_min=0.04,
    theta_max=0.16,
    n_theta=10,
    track="sr",
    baseline_theta=0.08,
    title_prefix="CAR-threshold sweep",
    output_prefix="theta_sweep_lines",
    batch: SimulationPlotBatch | None = None,
    show_delta_panel: bool = True,
    annotate_ends: bool = True,
    print_spread: bool = True,
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
    batch = _resolve_plot_batch(batch)

    baseline_theta = float(baseline_theta)
    theta_grid = np.round(np.linspace(theta_min, theta_max, n_theta), 2)

    if batch is not None and theta_policy_fixed is None:
        print(
            "[plot-batch] shared batch: policy θ fixed at "
            f"{batch.theta_policy:.2f}; curves vary θ_measure only"
        )
        title_prefix = (
            f"{title_prefix} (shared sim, policy θ={batch.theta_policy:.2f})"
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
        )
        curves.append(np.asarray(comp[track], float))

    if print_spread:
        spread = _theta_sweep_pairwise_spread(curves)
        print(
            f"[theta-sweep] track={track} max pairwise spread={spread:.2e} "
            f"(≈0 表示曲线数值重合)"
        )

    ylab = {
        "sr": "Systemic Risk (SR)",
        "cbs": "CBS (Low-CAR Share)",
        "fr": "FR (Failure Rate)",
        "cgr": "CGR (Capital Gap Ratio)",
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

    ax.set_title(rf"{title_prefix} — {ttl}$_t$ for each $\theta$  $(\mathbf{{W}}={weights})$")
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
    T=800,
    weights=(0.5, 0.3, 0.2),
    nsim=20,
    theta_policy=0.08,
    theta_min=0.04,
    theta_max=0.16,
    n_theta=10,
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
    T=800,
    weights=(0.5, 0.3, 0.2),
    nsim=20,
    theta_policy=0.08,
    theta_min=0.04,
    theta_max=0.16,
    n_theta=10,
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
    T=800,
    weights=(0.5, 0.3, 0.2),
    nsim=20,
    theta_min=0.04,
    theta_max=0.16,
    n_theta=10,
    track="sr",
    baseline_theta=0.08,
    batch: SimulationPlotBatch | None = None,
):
    """θ 同时影响角色分配（政策）与风险测度。"""
    batch = _resolve_plot_batch(batch)
    if batch is not None:
        print(
            "[plot-batch] policy scenario plot uses shared batch "
            "(measurement-only θ sweep on fixed policy path)"
        )
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
        title_prefix="Policy-threshold scenario comparison",
        output_prefix="theta_policy_scenario_lines",
        batch=batch,
    )


def _run_series_mean(weights, theta_measure, theta_policy=0.08, T=50, N=30, B=300, sigma=0.3,
                     nsim=20, seed0=DEFAULT_RANDOM_SEED, track="sr"):
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
    )
    return np.asarray(comp[track], dtype=float)


def plot_weight_sweep_lines(
    T=800,
    theta=0.08,
    nsim=50,
    batch: SimulationPlotBatch | None = None,
):
    """
    固定 θ=0.08，遍历 w1∈{0.10,...,0.90}（保持 w2:w3=3:2 归一化），
    每个 w1 一条 SR(t) 曲线。颜色映射 = w1。
    输出：FIG_DIR / 'weight_sweep_lines.png'
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    T = min(int(T), MAX_PLOT_STEPS)
    batch = _resolve_plot_batch(batch)
    w1_grid = np.round(np.linspace(0.10, 0.90, 9), 2)

    # 先算出所有曲线
    sr_curves = []
    w1_base, w2_base, w3_base = 0.5, 0.3, 0.2
    den = (w2_base + w3_base)
    for w1 in w1_grid:
        delta = w1 - w1_base
        w2 = w2_base - delta * (w2_base / den)
        w3 = w3_base - delta * (w3_base / den)
        if batch is not None:
            comp = score_batch_mean_components(
                batch, weights=(w1, w2, w3), theta_measure=float(theta)
            )
            sr = comp["sr"]
        else:
            sr = _run_series_mean((w1, w2, w3), theta, T=T, nsim=nsim)
        sr_curves.append(np.asarray(sr, float))

    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 7))

    # 颜色按 w1 映射
    cmap = mpl.colormaps['plasma'].reversed()
    norm = mpl.colors.Normalize(vmin=float(w1_grid.min()), vmax=float(w1_grid.max()))

    for w1, sr in zip(w1_grid, sr_curves):
        xs = np.arange(1, len(sr) + 1)
        ax.plot(xs, sr, lw=1.6, alpha=0.95, color=cmap(norm(w1)))

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label(r"$w_1$ (failure weight)")

    ax.set_title(r"Failure-weight sweep — $SR_t$ for each $w_1$  $(\theta=0.08)$")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Systemic Risk (SR)")
    ax.grid(True, alpha=0.4)

    out = FIG_DIR / "weight_sweep_lines.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)
    print(f"Saved: {out}")


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
                show_first=True,
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
    theta_min: float = 0.04,
    theta_max: float = 0.16,
    n_theta: int = 10,
) -> Path:
    """导出对比脚本所需的 JSON（无需再 import 本模块）。"""
    fig_dir = Path(fig_dir)
    comp = score_batch_mean_components(
        plot_batch, weights=weights, theta_measure=float(theta)
    )
    baseline = {k: [float(x) for x in comp[k]] for k in ("sr", "fr", "cbs", "cgr")}
    support = policy_support_mean_series(plot_batch)

    theta_grid = np.round(np.linspace(theta_min, theta_max, n_theta), 2)
    sr_curves = []
    for th in theta_grid:
        c = _run_series_mean_components(
            weights=weights,
            theta_measure=float(th),
            theta_policy=float(plot_batch.theta_policy),
            T=plot_batch.T,
            nsim=len(plot_batch.runs),
            batch=plot_batch,
        )
        sr_curves.append([float(x) for x in c["sr"]])

    payload = {
        "features": {
            "rollover_enabled": bool(getattr(plot_batch, "rollover_enabled", True)),
            "policy_support_enabled": bool(getattr(plot_batch, "policy_support_enabled", True)),
        },
        "weights": list(weights),
        "theta": float(theta),
        "baseline": baseline,
        "policy_support": {
            key: [float(x) for x in values]
            for key, values in support.items()
        },
        "theta_sweep": {
            "theta_grid": [float(x) for x in theta_grid],
            "sr_curves": sr_curves,
        },
    }
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
    train_models: bool = True,
    T: int = 800,
    nsim: int = 20,
    fig_dir: Path | None = None,
    network_steps=(10, 20, 30, 40, 50, 100, 150, 200),
    network_tag: str = "centralized",
    show: bool = False,
    rollover_enabled: bool = True,
    policy_support_enabled: bool = True,
) -> dict:
    """标准结果图：network panel、θ sweep、baseline trajectory。"""
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configure_figure_output(fig_dir)

    if train_models or matcher is None:
        if dataset is None:
            dataset = BankContagionDataset(num_simulations=1000, num_timesteps=5)
        model = train_model(dataset, num_epochs=10, batch_size=32)
        model = model.to(device).eval()
        matcher = train_matcher_from_dataset(
            dataset, epochs=3, lr=1e-4, neg_ratio=1.0, batch_graphs=64
        )
        matcher = matcher.to(device).eval()

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
        B=300.0,
        sigma=0.3,
        seed0=DEFAULT_RANDOM_SEED,
        matcher=matcher,
        device=device,
        rollover_enabled=rollover_enabled,
        policy_support_enabled=policy_support_enabled,
    )
    set_plot_simulation_batch(plot_batch)

    theta_sweep = plot_theta_measure_sweep_lines(
        T=T, weights=(0.5, 0.3, 0.2), batch=plot_batch
    )
    baseline = plot_baseline_trajectory(
        T=T, weights=(0.5, 0.3, 0.2), theta=0.08, batch=plot_batch
    )
    compare_artifacts = export_compare_artifacts(plot_batch, FIG_DIR)

    return {
        "fig_dir": FIG_DIR,
        "network_panel": network_panel,
        "theta_sweep": theta_sweep,
        "baseline_trajectory": baseline,
        "compare_artifacts": compare_artifacts,
        "plot_batch": plot_batch,
        "matcher": matcher,
        "device": device,
    }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Centralized baseline 标准结果图")
    parser.add_argument("--fig-dir", type=Path, default=None, help="图输出目录")
    parser.add_argument("--show", action="store_true", help="显示 matplotlib 窗口")
    parser.add_argument("--T", type=int, default=800, help="仿真步数")
    parser.add_argument("--nsim", type=int, default=20, help="plot batch 轨迹数")
    parser.add_argument("--no-train", action="store_true", help="跳过 GNN/matcher 训练")
    parser.add_argument("--no-rollover", action="store_true", help="关闭同业 rollover 分期续借")
    parser.add_argument("--no-policy-support", action="store_true", help="关闭央行 liquidity/capital support 注入")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("错误：未安装 PyTorch。请运行：")
        print(f"  {sys.executable} -m pip install torch torch-geometric")
        raise SystemExit(1)

    t_all = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_standard_figure_pipeline(
        device=device,
        train_models=not args.no_train,
        T=args.T,
        nsim=args.nsim,
        fig_dir=args.fig_dir,
        network_tag="centralized",
        show=args.show,
        rollover_enabled=not args.no_rollover,
        policy_support_enabled=not args.no_policy_support,
    )
    print(f"[time] TOTAL: {time.perf_counter() - t_all:.2f}s")

