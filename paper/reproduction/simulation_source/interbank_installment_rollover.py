"""
同业贷款调度与清算（单次 EN）。

正确时序（simulate_step；Step1–4 映射不变）：
  1) 上期流动性救助到账
  2) 偿还到期央行支持贷款
  3) Step4 结算：全部到期并入同一个 EN，只清算一次
  4) 市场冲击与角色
  5) Step1–3 成交/意图/撮合（OFF 新隔夜到期在 t+1，不展期）
  6) 项目投放并每期结算回报（短袖同期收回本金+回报；锁定期本金不兑成现金还隔夜；亏损由核心垫一部分）

Rollover OFF — schedule_type = "single_payment"：
  当天借入、本期结束后一次还本付息 D = P(1+r)（T+1，不展期成 5–20）；
  EN 清算中完成实际结算（禁止先直接付款再跑 EN，也不先撮合再结算）；
  任何未付 → 当期硬违约，未付不顺延。

Rollover ON — 成交即 5–20 期 installment：
  D_t = arrears + scheduled_interest + scheduled_principal；
  已付部分经 EN 转账；未付整体并入 arrears_due；
  consecutive_misses 仅作逾期统计，**永不**因逾期次数触发硬违约
  （已废除旧规则「连续三期不足 → 硬违约」）；
  退出网络仅由负权益等偿付能力违约触发。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol

import numpy as np

DAILY_ROLLOVER_SPREAD_SHORT = -0.00005
DAILY_ROLLOVER_SPREAD_LONG = 0.0
DAILY_OPPORTUNITY_MARGIN = 0.00012
DAILY_PROJECT_RETURN_DEFAULT = 0.00008
DAILY_PROJECT_RISK_DEFAULT = 0.00016
DAILY_BORROW_SPREAD = 0.00005

DEFAULT_DEBT_BURDEN_KAPPA = 3.0
DEFAULT_IBL_CAP_ASSET_LAMBDA = 0.5

SCHEDULE_SINGLE_PAYMENT = "single_payment"
SCHEDULE_INSTALLMENT = "installment"
# Legacy alias accepted when reading old contracts / configs.
_LEGACY_SINGLE = frozenset({"bullet", "single", "single_payment", "overnight"})


def normalize_schedule_type(schedule_type: str | None) -> str:
    s = str(schedule_type or SCHEDULE_SINGLE_PAYMENT).strip().lower()
    if s in _LEGACY_SINGLE:
        return SCHEDULE_SINGLE_PAYMENT
    if s == SCHEDULE_INSTALLMENT:
        return SCHEDULE_INSTALLMENT
    return SCHEDULE_SINGLE_PAYMENT


def bank_equity_proxy(bank: dict) -> float:
    """Equity ≈ assets − liabilities; never trust stale total_assets cache."""
    from bank_econ_shared import assets_ex_equity, liabilities_total

    return float(assets_ex_equity(bank) - liabilities_total(bank))


def opportunity_borrow_phi(
    bank: dict,
    *,
    kappa: float = DEFAULT_DEBT_BURDEN_KAPPA,
    equity: float | None = None,
) -> float:
    ibl = max(0.0, float(bank.get("interbank_liabilities", 0.0)))
    eq = float(equity) if equity is not None else bank_equity_proxy(bank)
    eq = max(float(eq), 1e-9)
    k = max(1e-9, float(kappa))
    return float(max(0.0, min(1.0, 1.0 - ibl / (k * eq))))


def ibl_borrowing_room(
    bank: dict,
    *,
    asset_lambda: float = DEFAULT_IBL_CAP_ASSET_LAMBDA,
    extra_new_borrowing: float = 0.0,
) -> float:
    from bank_econ_shared import assets_ex_equity

    assets = float(assets_ex_equity(bank))
    lam = max(0.0, float(asset_lambda))
    cap = lam * max(0.0, assets)
    ibl = max(0.0, float(bank.get("interbank_liabilities", 0.0)))
    return float(max(0.0, cap - ibl - max(0.0, float(extra_new_borrowing))))


@dataclass
class SettleInterbankResult:
    """同业结算结果（供借款政策使用）。"""
    failed: list[int]
    coupon_due_borrowers: set[int] = field(default_factory=set)
    coupon_cleared_borrowers: set[int] = field(default_factory=set)
    unpaid_amount: float = 0.0


class ContractLike(Protocol):
    contract_id: str
    lender_idx: int
    borrower_idx: int
    principal: float
    rate: float
    created_step: int
    maturity_step: int
    schedule_type: str
    remaining_principal: float
    coupon_rate: float
    tenor_total: int
    periods_paid: int


@dataclass
class ScheduleConfig:
    """成交与 rollover 的调度参数。"""
    schedule_selection: str = "auto"  # installment | single_payment | auto
    single_payment_maturity_periods: int = 1
    # Kept for config compatibility; OFF/ON no longer auto-split by size.
    bullet_maturity_periods: int = 1
    bullet_max_principal: float = 150.0
    installment_min_principal: float = 200.0
    lcr_installment_cutoff: float = 1.0
    min_tenor: int = 5
    max_tenor: int = 20
    ref_small: float = 50.0
    ref_large: float = 2000.0
    spread_short: float = DAILY_ROLLOVER_SPREAD_SHORT
    spread_long: float = DAILY_ROLLOVER_SPREAD_LONG
    rollover_mode: str = "installment"
    rollover_extension_periods: int = 1
    soft_principal_deferral: bool = True  # unused under new rules; kept for mapping
    allow_early_full_repay: bool = False
    early_repay_lcr_buffer: float = 0.85
    # installment_miss_limit removed: consecutive underpayment never hard-defaults.


def tenor_from_principal(
    principal: float,
    *,
    min_tenor: int = 5,
    max_tenor: int = 20,
    ref_small: float = 50.0,
    ref_large: float = 2000.0,
) -> int:
    p = max(0.0, float(principal))
    if p <= ref_small:
        return int(min_tenor)
    if p >= ref_large:
        return int(max_tenor)
    frac = (p - ref_small) / max(ref_large - ref_small, 1e-9)
    t = min_tenor + frac * (max_tenor - min_tenor)
    return int(np.clip(round(t), min_tenor, max_tenor))


def coupon_rate_by_tenor(
    settlement_rate: float,
    tenor: int,
    *,
    min_tenor: int = 5,
    max_tenor: int = 20,
    spread_short: float = DAILY_ROLLOVER_SPREAD_SHORT,
    spread_long: float = DAILY_ROLLOVER_SPREAD_LONG,
) -> float:
    t = int(tenor)
    r0 = float(settlement_rate)
    if max_tenor <= min_tenor:
        return r0 + float(spread_long)
    frac = (t - min_tenor) / float(max_tenor - min_tenor)
    frac = float(np.clip(frac, 0.0, 1.0))
    spread = float(spread_short) + frac * (float(spread_long) - float(spread_short))
    return r0 + spread


def rollover_coupon_rate(
    settlement_rate: float,
    tenor: int,
    cfg: ScheduleConfig,
) -> float:
    return coupon_rate_by_tenor(
        settlement_rate,
        tenor,
        min_tenor=cfg.min_tenor,
        max_tenor=cfg.max_tenor,
        spread_short=cfg.spread_short,
        spread_long=cfg.spread_long,
    )


def _borrower_lcr(borrower: dict) -> float:
    liq = float(borrower.get("liquid_assets", 0.0))
    lia = float(borrower.get("current_liabilities", 1.0))
    outflow = float(borrower.get("outflow_rate", 0.4))
    return liq / (lia * outflow + 1e-9)


def choose_trade_schedule(
    principal: float,
    settlement_rate: float,
    borrower: dict,
    cfg: ScheduleConfig,
) -> dict[str, Any]:
    """
    OFF → single_payment（当天借，本期结束后一次还本付息，不展期）。
    ON  → installment（成交即 5–20 期，无事后转换）。
    """
    P = max(0.0, float(principal))
    r_settle = float(settlement_rate)
    # Explicit installment / single_payment; "auto" follows rollover_mode.
    mode = str(cfg.rollover_mode).lower()
    rollover_on = mode not in ("", "off", "none", "false", "0", "disabled")

    raw_sel = str(cfg.schedule_selection).lower()
    if raw_sel in ("installment",):
        want_installment = True
    elif raw_sel in _LEGACY_SINGLE or raw_sel == "single_payment":
        want_installment = False
    else:
        want_installment = rollover_on

    if want_installment:
        tenor = tenor_from_principal(
            P,
            min_tenor=cfg.min_tenor,
            max_tenor=cfg.max_tenor,
            ref_small=cfg.ref_small,
            ref_large=cfg.ref_large,
        )
        cr = rollover_coupon_rate(r_settle, tenor, cfg)
        return {
            "schedule_type": SCHEDULE_INSTALLMENT,
            "reason": "rollover_on_installment_at_trade",
            "tenor": int(tenor),
            "coupon_rate": float(cr),
            "settlement_rate": r_settle,
            "maturity_in_periods": None,
        }

    maturity = max(
        1,
        int(
            getattr(cfg, "single_payment_maturity_periods", None)
            or getattr(cfg, "bullet_maturity_periods", 1)
            or 1
        ),
    )
    return {
        "schedule_type": SCHEDULE_SINGLE_PAYMENT,
        "reason": "rollover_off_single_payment",
        "tenor": None,
        "coupon_rate": None,
        "settlement_rate": r_settle,
        "maturity_in_periods": int(maturity),
    }


ROLLOVER_BORROW_BLOCK_ALL = "block_all"
ROLLOVER_BORROW_PROJECT_ONLY = "project_only"
ROLLOVER_BORROW_COUPON_CLEARED = "coupon_cleared"


def compute_project_investment_borrow_cap(
    borrower: dict,
    base_rate: float,
    *,
    opportunity_borrow: bool = True,
    opp_margin: float = DAILY_OPPORTUNITY_MARGIN,
    opp_risk_lambda: float = 0.5,
    opp_borrow_scale: float = 0.5,
    last_avg_rate: float | None = None,
    lia_cap_frac: float = 0.5,
) -> float:
    if not opportunity_borrow:
        return 0.0
    lia = float(borrower.get("current_liabilities", 0.0))
    roi = float(borrower.get("proj_mu", DAILY_PROJECT_RETURN_DEFAULT))
    pen = opp_risk_lambda * float(borrower.get("proj_sigma", DAILY_PROJECT_RISK_DEFAULT))
    r_hat = float(last_avg_rate) if last_avg_rate is not None else base_rate + float(
        borrower.get("borrow_spread", DAILY_BORROW_SPREAD)
    )
    if (roi - r_hat) <= (opp_margin + pen):
        return 0.0
    need_inv = opp_borrow_scale * lia
    return float(min(need_inv, lia_cap_frac * lia))


def rollover_borrow_quantity(
    bank_idx: int,
    borrower: dict,
    base_rate: float,
    need_liq: float,
    need_inv: float,
    *,
    rollover_blocked: set[int] | None,
    coupon_cleared: set[int] | None,
    coupon_due: set[int] | None,
    borrow_policy: str = ROLLOVER_BORROW_COUPON_CLEARED,
    opportunity_borrow: bool = True,
    opp_margin: float = DAILY_OPPORTUNITY_MARGIN,
    opp_risk_lambda: float = 0.5,
    opp_borrow_scale: float = 0.5,
    last_avg_rate: float | None = None,
    debt_burden_kappa: float = DEFAULT_DEBT_BURDEN_KAPPA,
    ibl_cap_asset_lambda: float = DEFAULT_IBL_CAP_ASSET_LAMBDA,
    equity: float | None = None,
) -> float:
    _ = (
        base_rate, opportunity_borrow, opp_margin,
        opp_risk_lambda, opp_borrow_scale, last_avg_rate,
    )
    idx = int(bank_idx)
    active_rollover = idx in (rollover_blocked or set())
    policy = str(borrow_policy or ROLLOVER_BORROW_COUPON_CLEARED).lower()
    has_coupon_due = idx in (coupon_due or set())
    coupon_is_cleared = idx in (coupon_cleared or set())

    # An active installment borrower may not use fresh interbank debt to hide
    # an unpaid current coupon.  This is a borrowing-control rule only: misses
    # remain arrears statistics and never become a hard default by themselves.
    if active_rollover and policy == ROLLOVER_BORROW_BLOCK_ALL:
        return 0.0
    if (
        active_rollover
        and policy == ROLLOVER_BORROW_COUPON_CLEARED
        and has_coupon_due
        and not coupon_is_cleared
    ):
        return 0.0
    liq = float(max(0.0, need_liq))
    inv = float(max(0.0, need_inv))
    phi = opportunity_borrow_phi(
        borrower, kappa=float(debt_burden_kappa), equity=equity
    )
    if active_rollover and policy == ROLLOVER_BORROW_PROJECT_ONLY:
        raw = phi * inv
    else:
        raw = liq + phi * inv
    room = ibl_borrowing_room(
        borrower, asset_lambda=float(ibl_cap_asset_lambda)
    )
    return float(max(0.0, min(raw, room)))


def outstanding_principal(c: ContractLike) -> float:
    """Scheduled remaining principal only (excludes arrears_due)."""
    st = normalize_schedule_type(getattr(c, "schedule_type", SCHEDULE_SINGLE_PAYMENT))
    if st == SCHEDULE_INSTALLMENT:
        return max(0.0, float(getattr(c, "remaining_principal", 0.0) or 0.0))
    return max(0.0, float(getattr(c, "principal", 0.0) or 0.0))


def total_claim(c: ContractLike) -> float:
    """
    Full creditor claim = outstanding principal + unpaid arrears.
    Use for IB exposure, CAR/LCR books, and default resolution claims.
    Do **not** use for scheduled I_t / P_t (those use outstanding_principal only).
    """
    principal = outstanding_principal(c)
    arrears = max(0.0, float(getattr(c, "arrears_due", 0.0) or 0.0))
    return float(principal + arrears)


# Backward-compatible alias used across DEN/CEN / resolution hooks.
effective_notional = total_claim


def active_installment_rollover_borrowers(book: Any, step: int) -> set[int]:
    _ = int(step)
    out: set[int] = set()
    for c in getattr(book, "contracts", []):
        bj = int(getattr(c, "borrower_idx", -1))
        if bj <= 0:
            continue
        if normalize_schedule_type(getattr(c, "schedule_type", "")) != SCHEDULE_INSTALLMENT:
            continue
        if effective_notional(c) <= 1e-8:
            continue
        out.add(bj)
    return out


def log_rollover_status(
    step: int, book: Any, *, phase: str = "post_settle", verbose: bool = True,
) -> set[int]:
    blocked = active_installment_rollover_borrowers(book, step)
    idx = sorted(blocked)
    if not verbose:
        return blocked
    print(
        f"[rollover] step={int(step)} phase={phase} "
        f"banks_in_rollover={len(idx)} bank_indices={idx}"
    )
    return blocked


def log_rollover_borrow_policy_note(step: int, policy: str, *, verbose: bool = True) -> None:
    if not verbose:
        return
    pol = str(policy).lower()
    if pol == ROLLOVER_BORROW_PROJECT_ONLY:
        print(
            f"[rollover] step={int(step)} borrow_policy=project_only: "
            "liquidity-gap borrow blocked; project-opportunity borrow still allowed"
        )
    elif pol == ROLLOVER_BORROW_COUPON_CLEARED:
        print(
            f"[rollover] step={int(step)} borrow_policy=coupon_cleared: "
            "if payment due this step, must clear it before project borrow; else project cap only"
        )
    elif pol == ROLLOVER_BORROW_BLOCK_ALL:
        print(
            f"[rollover] step={int(step)} borrow_policy=block_all: "
            "all new interbank borrowing blocked for rollover borrowers"
        )


def filter_borrowers_for_rollover_block(
    borrowers: list[int],
    book: Any,
    step: int,
    *,
    precomputed_blocked: set[int] | None = None,
    borrow_policy: str = ROLLOVER_BORROW_PROJECT_ONLY,
) -> tuple[list[int], set[int]]:
    _ = (book, step, precomputed_blocked, borrow_policy)
    return list(borrowers), set()


def _add_pair_flow(
    L: np.ndarray,
    lender: int,
    borrower: int,
    amount: float,
) -> None:
    if amount <= 1e-12:
        return
    L[borrower, lender] += float(amount)


def build_due_liability_matrix(
    flows: list[tuple[int, int, float]],
    n: int,
) -> np.ndarray:
    liabilities = np.zeros((n, n), dtype=float)
    for lender, borrower, amount in flows:
        _add_pair_flow(liabilities, int(lender), int(borrower), float(amount))
    return liabilities


def single_payment_due(c: ContractLike, step: int) -> bool:
    if getattr(c, "status", "active") == "suspended":
        return False
    if normalize_schedule_type(getattr(c, "schedule_type", "")) != SCHEDULE_SINGLE_PAYMENT:
        return False
    if outstanding_principal(c) <= 1e-8:
        return False
    return int(step) >= int(c.maturity_step)


# Backward-compatible name used by danger-zone helpers.
bullet_maturity_due = single_payment_due


def installment_payment_due(c: ContractLike, step: int) -> bool:
    """True when an installment contract owes a total payment this step."""
    if getattr(c, "status", "active") == "suspended":
        return False
    if normalize_schedule_type(getattr(c, "schedule_type", "")) != SCHEDULE_INSTALLMENT:
        return False
    step = int(step)
    if step <= int(c.created_step):
        return False
    if effective_notional(c) <= 1e-8:
        return False
    paid = int(getattr(c, "periods_paid", 0))
    tenor = max(1, int(getattr(c, "tenor_total", 1)))
    # After scheduled tenor, keep chasing residual principal / arrears each period.
    if paid >= tenor:
        return True
    next_due = int(c.created_step) + paid + 1
    return step >= next_due


# Legacy alias
installment_coupon_due = installment_payment_due


def scheduled_interest_and_principal(c: ContractLike) -> tuple[float, float]:
    """
    Planned (I_t, P_t) from outstanding principal only.
    Arrears are added separately in contract_total_due — never charge
    normal installment interest on arrears_due here.
    """
    rp = outstanding_principal(c)
    cr = float(getattr(c, "coupon_rate", c.rate) or c.rate)
    paid = int(getattr(c, "periods_paid", 0))
    tenor = max(1, int(getattr(c, "tenor_total", 1)))
    interest = rp * cr
    if paid >= tenor:
        principal_part = rp
    else:
        remaining_periods = tenor - paid
        if remaining_periods <= 1:
            principal_part = rp
        else:
            principal_part = rp / float(remaining_periods)
    return float(interest), float(principal_part)


# Legacy name
coupon_interest_and_principal = scheduled_interest_and_principal


def contract_total_due(c: ContractLike, step: int) -> float:
    """Total amount due this step for one contract (0 if not due)."""
    st = normalize_schedule_type(getattr(c, "schedule_type", SCHEDULE_SINGLE_PAYMENT))
    if st == SCHEDULE_SINGLE_PAYMENT:
        if not single_payment_due(c, step):
            return 0.0
        P = outstanding_principal(c)
        r = float(getattr(c, "settlement_rate", None) or c.rate)
        return float(P * (1.0 + r))
    if not installment_payment_due(c, step):
        return 0.0
    arrears = max(0.0, float(getattr(c, "arrears_due", 0.0) or 0.0))
    interest, principal_part = scheduled_interest_and_principal(c)
    return float(arrears + interest + principal_part)


def due_amounts_by_borrower(
    book: Any,
    step: int,
    *,
    schedule_type: str | None = None,
) -> dict[int, float]:
    """Preview amounts due this step, grouped by borrower. No transfers."""
    want = None
    if schedule_type is not None:
        want = normalize_schedule_type(schedule_type)
    out: dict[int, float] = {}
    for c in list(getattr(book, "contracts", []) or []):
        if getattr(c, "status", "active") == "suspended":
            continue
        st = normalize_schedule_type(getattr(c, "schedule_type", ""))
        if want is not None and st != want:
            continue
        due = float(contract_total_due(c, step))
        if due <= 1e-9:
            continue
        bj = int(getattr(c, "borrower_idx", -1))
        if bj <= 0:
            continue
        out[bj] = float(out.get(bj, 0.0)) + due
    return out


def schedule_config_from_mapping(m: dict[str, Any]) -> ScheduleConfig:
    sel = str(m.get("schedule_selection", "auto"))
    if sel.lower() == "bullet":
        sel = SCHEDULE_SINGLE_PAYMENT
    return ScheduleConfig(
        schedule_selection=sel,
        single_payment_maturity_periods=int(
            m.get("single_payment_maturity_periods", m.get("bullet_maturity_periods", 1))
        ),
        bullet_maturity_periods=int(m.get("bullet_maturity_periods", 1)),
        bullet_max_principal=float(m.get("bullet_max_principal", 150.0)),
        installment_min_principal=float(m.get("installment_min_principal", 200.0)),
        lcr_installment_cutoff=float(m.get("lcr_installment_cutoff", 1.0)),
        min_tenor=int(m.get("rollover_min_tenor", m.get("min_tenor", 5))),
        max_tenor=int(m.get("rollover_max_tenor", m.get("max_tenor", 20))),
        ref_small=float(m.get("rollover_ref_small", m.get("ref_small", 50.0))),
        ref_large=float(m.get("rollover_ref_large", m.get("ref_large", 2000.0))),
        spread_short=float(m.get("rollover_spread_short", m.get("spread_short", DAILY_ROLLOVER_SPREAD_SHORT))),
        spread_long=float(m.get("rollover_spread_long", m.get("spread_long", DAILY_ROLLOVER_SPREAD_LONG))),
        rollover_mode=str(m.get("rollover_mode", "rollover")),
        rollover_extension_periods=max(
            1, int(m.get("rollover_extension_periods", m.get("bullet_maturity_periods", 1)))
        ),
        soft_principal_deferral=bool(m.get("soft_principal_deferral", True)),
        allow_early_full_repay=bool(m.get("allow_early_full_repay", False)),
        early_repay_lcr_buffer=float(m.get("early_repay_lcr_buffer", 0.85)),
        # Legacy maps may still carry installment_miss_limit; it is intentionally ignored.
    )


def make_installment_contract(
    template: ContractLike,
    *,
    new_id: Callable[[], str],
    step: int,
    principal: float,
    settlement_rate: float,
    cfg: ScheduleConfig,
    tenor: int | None = None,
) -> ContractLike:
    """Create an installment contract at trade time (ON path helper)."""
    t = int(tenor) if tenor is not None else tenor_from_principal(
        principal,
        min_tenor=cfg.min_tenor,
        max_tenor=cfg.max_tenor,
        ref_small=cfg.ref_small,
        ref_large=cfg.ref_large,
    )
    cr = rollover_coupon_rate(float(settlement_rate), t, cfg)
    step = int(step)
    kwargs = dict(
        contract_id=new_id(),
        principal=float(principal),
        remaining_principal=float(principal),
        rate=float(settlement_rate),
        coupon_rate=float(cr),
        schedule_type=SCHEDULE_INSTALLMENT,
        tenor_total=int(t),
        periods_paid=0,
        settlement_rate=float(settlement_rate),
        created_step=step,
        maturity_step=step + int(t),
        consecutive_misses=0,
        arrears_due=0.0,
    )
    try:
        return replace(template, **kwargs)
    except TypeError:
        kwargs.pop("arrears_due", None)
        return replace(template, **kwargs)


def settle_interbank_period(
    book: Any,
    banks: list,
    n: int,
    step: int,
    *,
    cfg: ScheduleConfig,
    liquidity_default_candidates: Callable[..., np.ndarray],
    run_en_clearing_and_recovery: Callable[..., tuple],
    issue_liquidity_support: Callable[[int, float, int], None] | None,
    corridor_lending_rate: float,
    use_core: bool = False,
    clear_max_iter: int = 100,
    clear_tol: float = 1e-6,
    verbose_rollover: bool = True,
    clawback_settlement_support: Callable[[np.ndarray, np.ndarray, int], None] | None = None,
) -> SettleInterbankResult:
    """
    全部到期债务 → 一个 EN 负债矩阵 → 一次清算 → 再按合约处理未付。

    严禁：对 OFF/single_payment 先直接扣款/转账，再拿剩余债务跑 EN
    （那会制造人为的清算先后顺序，让先收款银行用新现金支付其他债务）。
    """
    _ = corridor_lending_rate
    step = int(step)
    failed_all: set[int] = set()
    to_remove: list[Any] = []
    due_count: Counter[int] = Counter()
    cleared_count: Counter[int] = Counter()
    unpaid_total = 0.0
    # ON: consecutive misses are statistics only. Never add to failed_all from miss count.
    _ = getattr(cfg, "installment_miss_limit", None)  # legacy attr ignored if present
    # Align hard-default residual with EN solver tolerance (avoid false OFF defaults).
    abs_tol = max(float(clear_tol), 1e-6)
    rel_tol = 1e-9

    def _settle_eps(claim: float) -> float:
        return max(abs_tol, rel_tol * max(0.0, float(claim)))

    # ---- 3–4) 全部到期债务（OFF+ON）合并为 due_flows，稍后一次建矩阵 ----
    due_flows: list[tuple[Any, int, int, float]] = []
    scheduled_parts: dict[str, tuple[float, float, float]] = {}  # cid -> (arrears, I, P)

    for c in list(getattr(book, "contracts", [])):
        # Suspended / deferred notes stay out of this period's EN matrix.
        if getattr(c, "status", "active") == "suspended":
            continue
        due = float(contract_total_due(c, step))
        if due <= _settle_eps(due):
            continue
        bj = int(c.borrower_idx)
        li = int(c.lender_idx)
        if bj <= 0 or li < 0:
            continue
        due_count[bj] += 1
        cid = str(getattr(c, "contract_id", ""))
        st = normalize_schedule_type(getattr(c, "schedule_type", ""))
        if st == SCHEDULE_INSTALLMENT:
            arrears = max(0.0, float(getattr(c, "arrears_due", 0.0) or 0.0))
            interest, prin = scheduled_interest_and_principal(c)
            scheduled_parts[cid] = (arrears, float(interest), float(prin))
        else:
            # OFF: full principal + interest already inside ``due``; no pre-pay.
            scheduled_parts[cid] = (0.0, 0.0, 0.0)
        due_flows.append((c, li, bj, due))

    def _run_one_en(
        flow_by_contract: list[tuple[Any, int, int, float]],
    ) -> tuple[list[int], dict[str, float]]:
        nonlocal unpaid_total
        if not flow_by_contract:
            return [], {}

        flows = [
            (li, bj, amt)
            for _, li, bj, amt in flow_by_contract
            if float(amt) > 1e-12
        ]
        if not flows:
            return [], {
                str(getattr(c, "contract_id", "")): 0.0
                for c, *_ in flow_by_contract
            }

        liabilities = build_due_liability_matrix(flows, n)
        shortfall = liquidity_default_candidates(
            liabilities, banks, n, use_core=use_core,
        )
        if issue_liquidity_support is not None:
            for i in range(n):
                if i == 0 or not shortfall[i]:
                    continue
                p_bar_i = float(liabilities[i].sum())
                need = max(0.0, p_bar_i - float(banks[i].get("liquid_assets", 0.0)))
                if need > 1e-6:
                    issue_liquidity_support(i, min(need, 8000.0), step)

        p_bar = liabilities.sum(axis=1)
        try:
            p, failed = run_en_clearing_and_recovery(
                liabilities, banks, n,
                use_core=use_core, max_iter=clear_max_iter, tol=clear_tol,
            )
        except TypeError:
            p, failed = run_en_clearing_and_recovery(
                liabilities, banks, n, use_core=use_core,
            )
        p = np.asarray(p, dtype=float)
        unpaid_total += float(np.maximum(np.asarray(p_bar, dtype=float) - p, 0.0).sum())

        if clawback_settlement_support is not None:
            clawback_settlement_support(np.asarray(liabilities, dtype=float), p, int(step))

        # Pro-rata allocate each debtor's EN payment across its due contracts.
        by_bj: dict[int, list[tuple[str, float]]] = {}
        for c, _li, bj, amt in flow_by_contract:
            cid = str(getattr(c, "contract_id", ""))
            by_bj.setdefault(int(bj), []).append((cid, float(amt)))
        paid: dict[str, float] = {}
        for bj, items in by_bj.items():
            due_sum = sum(a for _, a in items)
            pay = float(p[bj]) if 0 <= int(bj) < len(p) else 0.0
            if due_sum <= _settle_eps(due_sum):
                for cid, _a in items:
                    paid[cid] = 0.0
                continue
            scale = min(1.0, max(0.0, pay) / due_sum)
            for cid, a in items:
                paid[cid] = float(a * scale)
        return list(failed), paid

    # ---- 5) 全市场一次 EN：建负债矩阵 → 清算 → 实际转账 ----
    _failed_en, paid_by_cid = _run_one_en(due_flows)

    # ---- 6) 仅根据 EN 实付结果更新合约（OFF/ON）；此处不再发生新的现金转移 ----
    for c, _li, bj, due in due_flows:
        cid = str(getattr(c, "contract_id", ""))
        paid = float(paid_by_cid.get(cid, 0.0))
        unpaid = max(0.0, float(due) - paid)
        st = normalize_schedule_type(getattr(c, "schedule_type", ""))
        settle_eps = _settle_eps(due)

        if st == SCHEDULE_SINGLE_PAYMENT:
            # OFF: matured fully into the EN matrix; any shortfall → hard default.
            if unpaid <= settle_eps:
                cleared_count[bj] += 1
                to_remove.append(c)
            else:
                failed_all.add(int(bj))
                # Keep residual unpaid claim for LGD / estate recovery (do not delete).
                try:
                    c.principal = float(unpaid)
                    c.remaining_principal = float(unpaid)
                    c.arrears_due = 0.0
                    c.rate = 0.0
                    c.settlement_rate = 0.0
                    c.coupon_rate = 0.0
                    c.maturity_step = int(step)
                except Exception:
                    pass
            continue

        # ---- Installment: advance period; roll unpaid as arrears ----
        _arrears0, _I_t, P_t = scheduled_parts.get(cid, (0.0, 0.0, 0.0))
        rp = outstanding_principal(c)
        # Amortization clock always advances for this period's planned principal.
        new_rp = max(0.0, rp - max(0.0, float(P_t)))
        try:
            c.remaining_principal = float(new_rp)
            c.principal = float(new_rp)
            c.periods_paid = int(getattr(c, "periods_paid", 0)) + 1
        except Exception:
            pass

        if unpaid <= settle_eps:
            try:
                c.consecutive_misses = 0
                c.arrears_due = 0.0
            except Exception:
                pass
            cleared_count[bj] += 1
        else:
            # Paid portion already settled in EN; roll all unpaid into arrears.
            # consecutive_misses is overdue statistics only — NEVER hard-default on miss count
            # (including the retired "3 consecutive underpayments" rule).
            misses = int(getattr(c, "consecutive_misses", 0) or 0) + 1
            try:
                c.consecutive_misses = misses
                c.arrears_due = float(unpaid)
                # Keep chasing after maturity window.
                c.maturity_step = max(int(getattr(c, "maturity_step", step)), int(step) + 1)
            except Exception:
                pass
            # Do not: if misses >= N: failed_all.add(bj)

        if (
            float(getattr(c, "remaining_principal", 0.0) or 0.0) <= settle_eps
            and float(getattr(c, "arrears_due", 0.0) or 0.0) <= settle_eps
        ):
            to_remove.append(c)
    for c in to_remove:
        book.remove_contract(c)

    coupon_due_borrowers = {bj for bj, n_due in due_count.items() if n_due > 0}
    coupon_cleared_borrowers = {
        bj for bj, n_due in due_count.items() if cleared_count.get(bj, 0) >= n_due
    }

    log_rollover_status(step, book, phase="post_settle", verbose=verbose_rollover)
    if verbose_rollover and coupon_due_borrowers:
        print(
            f"[rollover] step={int(step)} due_borrowers={sorted(coupon_due_borrowers)} "
            f"cleared={sorted(coupon_cleared_borrowers)} "
            f"one_en=1 on_miss_default=off"
        )
    return SettleInterbankResult(
        failed=sorted(failed_all),
        coupon_due_borrowers=coupon_due_borrowers,
        coupon_cleared_borrowers=coupon_cleared_borrowers,
        unpaid_amount=float(unpaid_total),
    )
