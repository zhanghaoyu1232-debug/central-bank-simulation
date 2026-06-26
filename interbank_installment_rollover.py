"""
同业分期 rollover：20–120 个工作日对应不同每期利率；成交自动选 bullet（隔夜结清）或 installment（分期）。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol

import numpy as np

DAILY_ROLLOVER_SPREAD_SHORT = 0.00002
DAILY_ROLLOVER_SPREAD_LONG = 0.00008
DAILY_OPPORTUNITY_MARGIN = 0.00012
DAILY_PROJECT_RETURN_DEFAULT = 0.00008
DAILY_PROJECT_RISK_DEFAULT = 0.00016
DAILY_BORROW_SPREAD = 0.00005


@dataclass
class SettleInterbankResult:
    """同业分期/bullet 结算结果（供 Step1 借款政策使用）。"""
    failed: list[int]
    coupon_due_borrowers: set[int] = field(default_factory=set)
    coupon_cleared_borrowers: set[int] = field(default_factory=set)


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
    schedule_selection: str = "auto"
    bullet_maturity_periods: int = 1  # overnight: T+1 business-day principal+interest payment
    bullet_max_principal: float = 250_000.0
    installment_min_principal: float = 400_000.0
    lcr_installment_cutoff: float = 1.0
    min_tenor: int = 20
    max_tenor: int = 120
    ref_small: float = 50_000.0
    ref_large: float = 2_000_000.0
    spread_short: float = DAILY_ROLLOVER_SPREAD_SHORT
    spread_long: float = DAILY_ROLLOVER_SPREAD_LONG
    rollover_mode: str = "installment"


def tenor_from_principal(
    principal: float,
    *,
    min_tenor: int = 20,
    max_tenor: int = 120,
    ref_small: float = 50_000.0,
    ref_large: float = 2_000_000.0,
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
    min_tenor: int = 20,
    max_tenor: int = 120,
    spread_short: float = DAILY_ROLLOVER_SPREAD_SHORT,
    spread_long: float = DAILY_ROLLOVER_SPREAD_LONG,
) -> float:
    """
    分期每期利率：20 个工作日用较短久期利差，120 个工作日用较长久期利差，中间线性插值。
    （同期 bullet 结清仍用 settlement_rate，通常低于分期 coupon。）
    """
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
    自动选择成交结构。
    - bullet：短久期，到期按 settlement_rate 先息后本一次结清。
    - installment：20–120 个工作日，每期按 coupon_rate_by_tenor 先息后本。
    """
    P = max(0.0, float(principal))
    r_settle = float(settlement_rate)
    sel = str(cfg.schedule_selection).lower()
    lcr = _borrower_lcr(borrower)

    def _installment_decision(reason: str) -> dict[str, Any]:
        tenor = tenor_from_principal(
            P,
            min_tenor=cfg.min_tenor,
            max_tenor=cfg.max_tenor,
            ref_small=cfg.ref_small,
            ref_large=cfg.ref_large,
        )
        cr = rollover_coupon_rate(r_settle, tenor, cfg)
        return {
            "schedule_type": "installment",
            "reason": reason,
            "tenor": int(tenor),
            "coupon_rate": float(cr),
            "settlement_rate": r_settle,
            "maturity_in_periods": None,
        }

    def _bullet_decision(reason: str) -> dict[str, Any]:
        return {
            "schedule_type": "bullet",
            "reason": reason,
            "tenor": None,
            "coupon_rate": None,
            "settlement_rate": r_settle,
            "maturity_in_periods": int(cfg.bullet_maturity_periods),
        }

    if sel == "bullet":
        return _bullet_decision("forced_bullet")
    if sel == "installment":
        return _installment_decision("forced_installment")

    # --- auto ---
    if P >= float(cfg.installment_min_principal):
        return _installment_decision("auto_large_notional")
    if P <= float(cfg.bullet_max_principal) and lcr >= float(cfg.lcr_installment_cutoff):
        return _bullet_decision("auto_small_notional_high_lcr")
    if lcr < float(cfg.lcr_installment_cutoff):
        return _installment_decision("auto_low_lcr")
    return _bullet_decision("auto_default_bullet")


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
    """
    与 collect_intentions 中 need_inv 同口径：仅项目机会驱动的新增借款上限。
    用于 rollover 存续期银行——禁止为补 LCR/还同业而借，但允许项目融资额度。
    """
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
) -> float:
    """
    计算借款人本步可新增的同业需求额度。
    coupon_cleared（默认）：本期有分期到期的，须先还清当期息/本且项目融资能力>0 才允许借；
    本期无到期的 rollover 银行仅允许项目额度；未还清当期则禁止新借。
    """
    i = int(bank_idx)
    blocked = rollover_blocked or set()
    if i not in blocked:
        return float(max(0.0, need_liq) + max(0.0, need_inv))

    policy = str(borrow_policy).lower()
    if policy == ROLLOVER_BORROW_BLOCK_ALL:
        return 0.0

    proj_cap = compute_project_investment_borrow_cap(
        borrower,
        base_rate,
        opportunity_borrow=opportunity_borrow,
        opp_margin=opp_margin,
        opp_risk_lambda=opp_risk_lambda,
        opp_borrow_scale=opp_borrow_scale,
        last_avg_rate=last_avg_rate,
    )

    if policy == ROLLOVER_BORROW_PROJECT_ONLY:
        return proj_cap

    cleared = coupon_cleared or set()
    due = coupon_due or set()
    if i in due:
        if i in cleared and proj_cap > 1e-6:
            return proj_cap
        return 0.0
    return proj_cap


def active_installment_rollover_borrowers(book: Any, step: int) -> set[int]:
    """
    存续同业分期（installment）的借款人集合：处于 rollover 还款期。
    这些银行不得新增同业借款用于偿还既有分期（禁止拆东墙补西墙）。
    """
    step = int(step)
    out: set[int] = set()
    for c in getattr(book, "contracts", []):
        if getattr(c, "schedule_type", "bullet") != "installment":
            continue
        if effective_notional(c) <= 1e-8:
            continue
        paid = int(getattr(c, "periods_paid", 0))
        tenor = int(getattr(c, "tenor_total", 1))
        if paid >= tenor:
            continue
        bj = int(c.borrower_idx)
        if bj > 0:
            out.add(bj)
    return out


def log_rollover_status(
    step: int, book: Any, *, phase: str = "post_settle", verbose: bool = True,
) -> set[int]:
    """打印当前处于 rollover 的银行数量与编号，并返回借款人集合。"""
    blocked = active_installment_rollover_borrowers(book, step)
    idx = sorted(blocked)
    if not verbose:
        return blocked
    print(
        f"[rollover] step={int(step)} phase={phase} "
        f"banks_in_rollover={len(idx)} bank_indices={idx}"
    )
    if idx:
        print(
            f"[rollover] step={int(step)} 禁止拆东墙补西墙: "
            f"上述 {len(idx)} 家存续分期借款人不得新增同业借款"
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
            "if coupon due this step, must clear it before project borrow; else project cap only"
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
    """从撮合借款人池中剔除 rollover 银行（仅 block_all）；project_only 不剔除。"""
    blocked = (
        precomputed_blocked
        if precomputed_blocked is not None
        else active_installment_rollover_borrowers(book, step)
    )
    pol = str(borrow_policy).lower()
    if not blocked or pol in (ROLLOVER_BORROW_PROJECT_ONLY, ROLLOVER_BORROW_COUPON_CLEARED):
        return list(borrowers), blocked
    removed = sorted(set(borrowers) & blocked)
    if removed:
        print(
            f"[rollover] step={int(step)} phase=match "
            f"blocked_new_borrow={removed}"
        )
    return [j for j in borrowers if j not in blocked], blocked


def effective_notional(c: ContractLike) -> float:
    if getattr(c, "schedule_type", "bullet") == "installment":
        rp = float(getattr(c, "remaining_principal", 0.0) or 0.0)
        return rp if rp > 1e-12 else float(c.principal)
    return float(c.principal)


def _add_pair_flow(L: np.ndarray, lender: int, borrower: int, amount: float) -> None:
    if amount <= 1e-12:
        return
    L[lender, borrower] += amount
    L[borrower, lender] -= amount


def build_L_from_pair_flows(flows: list[tuple[int, int, float]], n: int) -> np.ndarray:
    L = np.zeros((n, n), dtype=float)
    for li, bj, amt in flows:
        _add_pair_flow(L, int(li), int(bj), float(amt))
    return L


def installment_coupon_due(c: ContractLike, step: int) -> bool:
    if getattr(c, "schedule_type", "bullet") != "installment":
        return False
    step = int(step)
    if step <= int(c.created_step):
        return False
    if step > int(c.maturity_step):
        return False
    paid = int(getattr(c, "periods_paid", 0))
    tenor = int(getattr(c, "tenor_total", 1))
    if paid >= tenor:
        return False
    next_due = int(c.created_step) + paid + 1
    return step >= next_due


def bullet_maturity_due(c: ContractLike, step: int) -> bool:
    if getattr(c, "schedule_type", "bullet") == "installment":
        return False
    return int(step) >= int(c.maturity_step)


def coupon_interest_and_principal(c: ContractLike) -> tuple[float, float]:
    rp = effective_notional(c)
    cr = float(getattr(c, "coupon_rate", c.rate) or c.rate)
    paid = int(getattr(c, "periods_paid", 0))
    tenor = max(1, int(getattr(c, "tenor_total", 1)))
    interest = rp * cr
    remaining_periods = tenor - paid
    if remaining_periods <= 1:
        principal_part = rp
    else:
        principal_part = rp / float(remaining_periods)
    return float(interest), float(principal_part)


def apply_installment_advance(c: ContractLike, interest_paid: float, principal_paid: float) -> None:
    c.periods_paid = int(getattr(c, "periods_paid", 0)) + 1
    rp = effective_notional(c)
    c.remaining_principal = max(0.0, rp - float(principal_paid))
    c.principal = float(c.remaining_principal)
    if int(c.periods_paid) >= int(getattr(c, "tenor_total", 1)) or c.remaining_principal <= 1e-8:
        c.remaining_principal = 0.0
        c.principal = 0.0


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
    t = int(tenor) if tenor is not None else tenor_from_principal(
        principal,
        min_tenor=cfg.min_tenor,
        max_tenor=cfg.max_tenor,
        ref_small=cfg.ref_small,
        ref_large=cfg.ref_large,
    )
    cr = rollover_coupon_rate(float(settlement_rate), t, cfg)
    step = int(step)
    return replace(
        template,
        contract_id=new_id(),
        principal=float(principal),
        remaining_principal=float(principal),
        rate=float(settlement_rate),
        coupon_rate=float(cr),
        schedule_type="installment",
        tenor_total=int(t),
        periods_paid=0,
        settlement_rate=float(settlement_rate),
        created_step=step,
        maturity_step=step + int(t),
    )


def schedule_config_from_mapping(m: dict[str, Any]) -> ScheduleConfig:
    return ScheduleConfig(
        schedule_selection=str(m.get("schedule_selection", "auto")),
        bullet_maturity_periods=int(m.get("bullet_maturity_periods", 1)),
        bullet_max_principal=float(m.get("bullet_max_principal", 250_000.0)),
        installment_min_principal=float(m.get("installment_min_principal", 400_000.0)),
        lcr_installment_cutoff=float(m.get("lcr_installment_cutoff", 1.0)),
        min_tenor=int(m.get("rollover_min_tenor", m.get("min_tenor", 20))),
        max_tenor=int(m.get("rollover_max_tenor", m.get("max_tenor", 120))),
        ref_small=float(m.get("rollover_ref_small", m.get("ref_small", 50_000.0))),
        ref_large=float(m.get("rollover_ref_large", m.get("ref_large", 2_000_000.0))),
        spread_short=float(m.get("rollover_spread_short", m.get("spread_short", DAILY_ROLLOVER_SPREAD_SHORT))),
        spread_long=float(m.get("rollover_spread_long", m.get("spread_long", DAILY_ROLLOVER_SPREAD_LONG))),
        rollover_mode=str(m.get("rollover_mode", "installment")),
    )


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
) -> SettleInterbankResult:
    step = int(step)
    failed_all: set[int] = set()
    to_remove: list[Any] = []
    to_add: list[Any] = []
    due_count: Counter[int] = Counter()
    cleared_count: Counter[int] = Counter()

    installment_list = [c for c in book.contracts if installment_coupon_due(c, step)]
    bullets_due = [c for c in book.contracts if bullet_maturity_due(c, step)]

    def _run_phase(flows: list[tuple[int, int, float]]) -> list[int]:
        if not flows:
            return []
        L = build_L_from_pair_flows(flows, n)
        shortfall = liquidity_default_candidates(L, banks, n, use_core=use_core)
        if issue_liquidity_support is not None:
            for i in range(n):
                if i == 0 or not shortfall[i]:
                    continue
                p_bar_i = float(np.maximum(-L[i], 0.0).sum())
                need = max(0.0, p_bar_i - float(banks[i].get("liquid_assets", 0.0)))
                if need > 1e-6:
                    issue_liquidity_support(i, min(need, 2000.0), step)
        try:
            _, failed = run_en_clearing_and_recovery(
                L, banks, n, use_core=use_core, max_iter=clear_max_iter, tol=clear_tol,
            )
        except TypeError:
            _, failed = run_en_clearing_and_recovery(L, banks, n, use_core=use_core)
        return list(failed)

    for c in installment_list:
        bj = int(c.borrower_idx)
        due_count[bj] += 1
        interest, principal_part = coupon_interest_and_principal(c)
        failed_i = _run_phase([(c.lender_idx, c.borrower_idx, interest)])
        failed_p = _run_phase([(c.lender_idx, c.borrower_idx, principal_part)])
        failed_all.update(failed_i)
        failed_all.update(failed_p)
        if bj not in failed_i and bj not in failed_p:
            cleared_count[bj] += 1
        apply_installment_advance(c, interest, principal_part)
        if effective_notional(c) <= 1e-8 or int(c.periods_paid) >= int(c.tenor_total):
            to_remove.append(c)

    coupon_due_borrowers = {bj for bj, n in due_count.items() if n > 0}
    coupon_cleared_borrowers = {
        bj for bj, n in due_count.items() if cleared_count.get(bj, 0) >= n
    }

    rollover_on = str(cfg.rollover_mode).lower() in ("installment", "on", "yes", "auto")

    for c in bullets_due:
        P = float(c.principal)
        r_settle = float(getattr(c, "settlement_rate", None) or c.rate)
        interest = P * r_settle
        principal_part = P
        failed_i = _run_phase([(c.lender_idx, c.borrower_idx, interest)])
        failed_p = _run_phase([(c.lender_idx, c.borrower_idx, principal_part)])
        failed_all.update(failed_i)
        failed_all.update(failed_p)
        to_remove.append(c)
        if rollover_on:
            if (
                c.borrower_idx > 0
                and c.lender_idx > 0
                and c.borrower_idx < n
                and c.lender_idx < n
                and banks[c.borrower_idx].get("is_active", True)
                and banks[c.lender_idx].get("is_active", True)
            ):
                new_c = make_installment_contract(
                    c,
                    new_id=book._new_id,
                    step=step,
                    principal=P,
                    settlement_rate=r_settle,
                    cfg=cfg,
                )
                to_add.append(new_c)

    for c in to_remove:
        book.remove_contract(c)
    for c in to_add:
        book.add_contract(c)

    log_rollover_status(step, book, phase="post_settle", verbose=verbose_rollover)
    if verbose_rollover and coupon_due_borrowers:
        print(
            f"[rollover] step={int(step)} coupon_due_borrowers={sorted(coupon_due_borrowers)} "
            f"coupon_cleared={sorted(coupon_cleared_borrowers)}"
        )
    return SettleInterbankResult(
        failed=sorted(failed_all),
        coupon_due_borrowers=coupon_due_borrowers,
        coupon_cleared_borrowers=coupon_cleared_borrowers,
    )
