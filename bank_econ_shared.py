# -*- coding: utf-8 -*-
"""Shared economic-logic helpers for DEN/CEN (phase-1 balance sheet & project risk)."""
from __future__ import annotations

from math import ceil
from typing import Any

DEFAULT_DEBT_BURDEN_KAPPA = 3.0
ORIGINATION_PD_ALPHA = 1.0
ORIGINATION_SHOCK_BETA = 0.5
LIQ_FROM_ASSETS_FRAC = 0.10
# Float slack so CAR==threshold does not flicker across 0.079999999.
CAR_COMPARE_TOL = 1e-9
# A narrow 5bp recovery band filters floating-point threshold chatter without
# suppressing genuine CAR recovery. CBS is calculated only over living banks.
CBS_EXIT_HYSTERESIS = 0.0005
# Newly borrowed funds: this share of project slots settle P+return the same period.
# Both regimes use the same asset-maturity mix; the rollover switch changes the
# liability schedule, not the riskiness of the project purchased with the loan.
BORROWED_PERIOD_SETTLED_SHARE = 0.5
BORROWED_PERIOD_SETTLED_SHARE_ON = 0.5
BORROWED_PERIOD_SETTLED_SHARE_OFF = 0.5

# Measure-only SR sweeps (same recorded batch; no re-simulation).
MEASURE_W1_MIN = 0.10
MEASURE_W1_MAX = 0.90
MEASURE_W1_POINTS = 9  # 0.10, 0.20, ..., 0.90
MEASURE_THETA_MIN = 0.05
MEASURE_THETA_MAX = 0.15
MEASURE_THETA_POINTS = 11  # 0.05, 0.06, ..., 0.15
MEASURE_THETA_BASELINE = 0.08


def _closed_step_grid(lo: float, hi: float, n: int) -> list[float]:
    n = max(1, int(n))
    if n == 1:
        return [round(float(lo), 2)]
    step = (float(hi) - float(lo)) / float(n - 1)
    return [round(float(lo) + i * step, 2) for i in range(n)]


def measure_w1_grid(
    w1_min: float = MEASURE_W1_MIN,
    w1_max: float = MEASURE_W1_MAX,
    n_w1: int = MEASURE_W1_POINTS,
) -> list[float]:
    """FR-weight grid: w1 from 0.1 to 0.9 inclusive."""
    return _closed_step_grid(w1_min, w1_max, n_w1)


def measure_theta_grid(
    theta_min: float = MEASURE_THETA_MIN,
    theta_max: float = MEASURE_THETA_MAX,
    n_theta: int = MEASURE_THETA_POINTS,
    baseline_theta: float | None = MEASURE_THETA_BASELINE,
) -> list[float]:
    """CBS-threshold grid: θ from 0.05 to 0.15 inclusive.

    The baseline 0.08 is re-inserted only if a custom grid would skip it.
    """
    grid = _closed_step_grid(theta_min, theta_max, n_theta)
    if baseline_theta is None:
        return grid
    b = round(float(baseline_theta), 2)
    if all(abs(x - b) > 1e-12 for x in grid):
        grid.append(b)
        grid.sort()
    return grid


def car_below_threshold(car: float, threshold: float, *, tol: float = CAR_COMPARE_TOL) -> bool:
    """True iff CAR is strictly below the threshold after a tiny compare slack."""
    return float(car) < float(threshold) - float(tol)


def bank_in_cbs_active(
    bank: dict,
    car_threshold: float = 0.08,
    *,
    car: float,
) -> bool:
    """CBS membership for a *living* bank, with one-period recovery hysteresis.

    Enter when current CAR < θ.  Stay until CAR ≥ θ + ``CBS_EXIT_HYSTERESIS``.
    Previous-step CAR is ``lag_car`` (frozen after SR), so recorded snapshots
    replay the same latch. Failed banks are excluded by the caller and recorded
    only in FR.
    """
    th = float(car_threshold)
    current = float(car)
    if car_below_threshold(current, th):
        return True
    lag = bank.get("lag_car")
    if lag is None:
        return False
    was_low = car_below_threshold(float(lag), th)
    if was_low and current < th + float(CBS_EXIT_HYSTERESIS) - CAR_COMPARE_TOL:
        return True
    return False


def assets_ex_equity(bank: dict) -> float:
    """Cash + interbank assets + projects (excludes core capital / equity)."""
    projects = float(
        (bank.get("investment") or {}).get("projects", {}).get("amount", 0.0) or 0.0
    )
    return (
        float(bank.get("liquid_assets", 0.0) or 0.0)
        + float(bank.get("interbank_assets", 0.0) or 0.0)
        + projects
    )


def liabilities_total(bank: dict) -> float:
    """Prefer current_liabilities when it already embeds IBL; avoid double count.

    Always adds ``termed_out_liabilities`` (long-term rollover stock).
    """
    cur = float(bank.get("current_liabilities", 0.0) or 0.0)
    ibl = float(bank.get("interbank_liabilities", 0.0) or 0.0)
    termed = float(bank.get("termed_out_liabilities", 0.0) or 0.0)
    bd = bank.get("liabilities_breakdown") or {}
    bd_ib = float(bd.get("interbank", 0.0) or 0.0)
    # If breakdown.interbank tracks IBL and current_liabilities includes it, use cur only.
    if abs(cur - (float(bd.get("deposits", 0.0) or 0.0)
                  + float(bd.get("wholesale", 0.0) or 0.0)
                  + bd_ib)) < 1e-6:
        base = cur
    else:
        # Legacy: current_liabilities = external only.
        base = cur + ibl
    return float(base + max(0.0, termed))


# t=0 operating CAR sits above the 8% CBS/regulatory floor (conservation-style
# buffer). High appetite thins the buffer; it does not birth banks on θ.
# 12–16%: first-week cash→project origination drops CAR ~2–3pp; 10% still
# punched through 8% and support pinned those banks at 7.5% (stuck in CBS).
INIT_TARGET_CAR_FLOOR = 0.12
INIT_TARGET_CAR_CEILING = 0.16
INIT_TARGET_CAR_APPETITE_SLOPE = 0.040
# Do not warehouse the opening book as cash (RW=0). Incremental origination
# would then explode RWA relative to a tiny floor RWA=0.08L.
INIT_MAX_LIQUID_FRAC = 0.40


def target_car_from_risk_appetite(risk_appetite: float) -> float:
    """Higher appetite → thinner buffer, clipped to [12%, 16%]."""
    ra = float(risk_appetite)
    raw = INIT_TARGET_CAR_CEILING - INIT_TARGET_CAR_APPETITE_SLOPE * ra
    return float(min(INIT_TARGET_CAR_CEILING, max(INIT_TARGET_CAR_FLOOR, raw)))


def solve_initial_equity_for_target_car(
    *,
    liabilities: float,
    liquid_target: float,
    risk_appetite: float,
    interbank_assets: float = 0.0,
    invest: float = 0.0,
    n_iter: int = 10,
    regulatory_rwa_fn=None,
) -> tuple[float, float, float, float]:
    """
    Reverse-engineer equity so CAR hits target_car(risk_appetite).

    Cash has zero risk weight, so the last step plugs any residual as liquid
    assets. That keeps A = L + E without a post-invest RWA jump that would
    drop realized CAR below the target (and into CBS at t=0).

    Returns (equity, liquid, projects, total_assets).
    """
    from bank_regulatory import regulatory_rwa as _default_rwa

    rwa_fn = regulatory_rwa_fn or _default_rwa
    lia = max(0.0, float(liabilities))
    iba = max(0.0, float(interbank_assets))
    target_car = target_car_from_risk_appetite(risk_appetite)
    equity = float(INIT_TARGET_CAR_CEILING) * lia
    liquid = 0.0
    projects = 0.0
    total_assets = lia + equity
    for _ in range(max(1, int(n_iter))):
        liquid, projects, total_assets = close_initial_asset_split(
            intended_core=equity,
            liabilities=lia,
            liquid_target=float(liquid_target),
            interbank_assets=iba,
        )
        if float(invest) > 1e-12:
            liquid, projects = transfer_liquid_to_projects(liquid, projects, float(invest))
            total_assets = liquid + iba + projects
        rwa = float(
            rwa_fn(
                interbank_assets=iba,
                projects_amt=projects,
                liabilities=lia,
            )
        )
        equity = float(target_car) * max(rwa, 1e-9)
    liquid, projects, total_assets = close_initial_asset_split(
        intended_core=equity,
        liabilities=lia,
        liquid_target=float(liquid_target),
        interbank_assets=iba,
    )
    if float(invest) > 1e-12:
        liquid, projects = transfer_liquid_to_projects(liquid, projects, float(invest))
        total_assets = liquid + iba + projects
    rwa = float(
        rwa_fn(
            interbank_assets=iba,
            projects_amt=projects,
            liabilities=lia,
        )
    )
    equity = float(target_car) * max(rwa, 1e-9)
    accounting = float(liquid) + float(iba) + float(projects) - lia
    liquid = float(liquid) + (equity - accounting)
    if liquid < 0.0:
        projects = max(0.0, float(projects) + liquid)
        liquid = 0.0
        rwa = float(
            rwa_fn(
                interbank_assets=iba,
                projects_amt=projects,
                liabilities=lia,
            )
        )
        equity = float(target_car) * max(rwa, 1e-9)
        accounting = float(liquid) + float(iba) + float(projects) - lia
        liquid = max(0.0, float(liquid) + (equity - accounting))
    total_assets = float(liquid) + float(iba) + float(projects)
    equity = float(total_assets) - lia
    return float(equity), float(liquid), float(projects), float(total_assets)


def close_initial_asset_split(
    *,
    intended_core: float,
    liabilities: float,
    liquid_target: float,
    liq_frac: float = LIQ_FROM_ASSETS_FRAC,
    interbank_assets: float = 0.0,
    max_liquid_frac: float | None = None,
) -> tuple[float, float, float]:
    """
    Close A = L + Core, then split into liquid vs projects (IB assets reserved).

    Liquid is clipped to ``[liq_frac, max_liquid_frac]`` of non-IB assets so the
    opening book already has an operating loan portfolio in RWA.

    Returns (liquid_assets, projects_amt, total_assets).
    """
    core = max(0.0, float(intended_core))
    lia = max(0.0, float(liabilities))
    iba = max(0.0, float(interbank_assets))
    total_assets = lia + core
    remaining = max(0.0, total_assets - iba)
    floor_liq = float(liq_frac) * remaining
    cap_frac = INIT_MAX_LIQUID_FRAC if max_liquid_frac is None else float(max_liquid_frac)
    cap_liq = max(0.0, float(cap_frac)) * remaining
    lo = min(floor_liq, remaining)
    hi = min(max(cap_liq, lo), remaining)
    target = max(0.0, float(liquid_target))
    liquid = min(max(target, lo), hi)
    projects = max(0.0, remaining - liquid)
    return liquid, projects, total_assets


def transfer_liquid_to_projects(liquid: float, projects: float, invest: float) -> tuple[float, float]:
    """Move cash into projects without changing total assets."""
    amt = max(0.0, min(float(invest), float(liquid)))
    return float(liquid) - amt, float(projects) + amt


def car_risk_grade(car: float, car_threshold: float = 0.08) -> float:
    thr = max(float(car_threshold), 1e-9)
    c = float(car)
    if c >= 1.5 * thr:
        return 0.0
    if c >= thr:
        return 0.33
    if c >= 0.5 * thr:
        return 0.66
    return 1.0


def lcr_risk_grade(lcr: float) -> float:
    x = float(lcr)
    if x >= 1.2:
        return 0.0
    if x >= 1.0:
        return 0.33
    if x >= 0.8:
        return 0.66
    return 1.0


def debt_burden(bank: dict, *, kappa: float = DEFAULT_DEBT_BURDEN_KAPPA) -> float:
    ibl = max(0.0, float(bank.get("interbank_liabilities", 0.0) or 0.0))
    eq = max(assets_ex_equity(bank) - liabilities_total(bank), 1e-9)
    k = max(1e-9, float(kappa))
    return float(min(max(ibl / (k * eq), 0.0), 1.0))


def arrears_intensity(bank: dict, system: Any = None, bank_idx: int | None = None) -> float:
    """[0,1] overdue intensity from contracts if available, else bank counters."""
    book = getattr(system, "contract_book", None) if system is not None else None
    contracts = list(getattr(book, "contracts", []) or [])
    bj = int(bank_idx) if bank_idx is not None else int(bank.get("id", -1))
    n_notes = 0
    miss_sum = 0.0
    n_late = 0
    for c in contracts:
        if int(getattr(c, "borrower_idx", -1)) != bj:
            continue
        n_notes += 1
        misses = int(getattr(c, "consecutive_misses", 0) or 0)
        arrears = float(getattr(c, "arrears_due", 0.0) or 0.0)
        miss_sum += float(misses)
        if misses > 0 or arrears > 1e-9:
            n_late += 1
    if n_notes > 0:
        underpay = min(miss_sum / max(3.0 * n_notes, 1.0), 1.0)
        late = n_late / max(n_notes, 1)
        return float(min(max(0.5 * underpay + 0.5 * late, 0.0), 1.0))
    under = float(bank.get("public_underpay_events", 0.0) or 0.0)
    ontime = float(bank.get("public_ontime_events", 0.0) or 0.0)
    tot = under + ontime
    if tot <= 1e-12:
        return 0.0
    return float(min(under / max(tot, 1.0), 1.0))


def borrower_origination_risk_q(
    bank: dict,
    *,
    system: Any = None,
    bank_idx: int | None = None,
    car_threshold: float = 0.08,
    kappa: float = DEFAULT_DEBT_BURDEN_KAPPA,
    car: float | None = None,
) -> float:
    """
    q_j = 0.5 RiskCAR + 0.2 RiskLCR + 0.2 Debt + 0.1 Arrears.
    Frozen onto ProjectLoan at origination.
    """
    if car is None:
        car = float(bank.get("capital_adequacy_ratio", 0.0) or 0.0)
    risk_car = car_risk_grade(car, car_threshold)
    risk_lcr = lcr_risk_grade(float(bank.get("liquidity_coverage_ratio", 1.0) or 0.0))
    debt = debt_burden(bank, kappa=kappa)
    arr = arrears_intensity(bank, system=system, bank_idx=bank_idx)
    q = 0.5 * risk_car + 0.2 * risk_lcr + 0.2 * debt + 0.1 * arr
    return float(min(max(q, 0.0), 1.0))


def effective_project_pd(
    base_pd: float,
    origination_risk: float,
    macro_multiplier: float = 1.0,
    *,
    alpha: float = ORIGINATION_PD_ALPHA,
) -> float:
    q = max(0.0, float(origination_risk))
    return float(
        min(1.0, max(0.0, float(base_pd)) * (1.0 + float(alpha) * q) * max(0.0, float(macro_multiplier)))
    )


def effective_shock_std(
    sigma: float,
    origination_risk: float,
    *,
    beta: float = ORIGINATION_SHOCK_BETA,
) -> float:
    q = max(0.0, float(origination_risk))
    return float(max(0.0, float(sigma)) * (1.0 + float(beta) * q))


SCREENING_LOSS_KEYS = (
    "funded_amount",
    "funded_q_amount",
    "project_default_principal",
    "project_default_lgd_loss",
    "project_default_q_weighted_lgd_loss",
    "project_negative_return_loss",
    "interbank_writeoff_loss",
    "funding_gap",
)


def empty_screening_loss() -> dict[str, float]:
    return {k: 0.0 for k in SCREENING_LOSS_KEYS}


def reset_screening_loss(system, *, totals: bool = True, step: bool = True) -> None:
    if system is None:
        return
    if totals:
        system.screening_loss_totals = empty_screening_loss()
        system.first_failure_events = []
        system._first_failure_seen = set()
    if step:
        system.screening_loss_step = empty_screening_loss()


def note_screening_loss(system, key: str, amount: float) -> None:
    if system is None:
        return
    amt = float(amount)
    if amt == 0.0:
        return
    for attr in ("screening_loss_step", "screening_loss_totals"):
        rec = getattr(system, attr, None)
        if not isinstance(rec, dict):
            rec = empty_screening_loss()
            setattr(system, attr, rec)
        rec[str(key)] = float(rec.get(str(key), 0.0)) + amt


def accrue_funded_trade(system, amount: float, q: float) -> None:
    a = max(0.0, float(amount))
    if a <= 0.0:
        return
    note_screening_loss(system, "funded_amount", a)
    note_screening_loss(system, "funded_q_amount", a * float(q))


def accrue_project_default_loss(system, *, principal: float, lgd: float, q: float) -> None:
    p = max(0.0, float(principal))
    l = min(max(0.0, float(lgd)), 1.0)
    qq = min(max(0.0, float(q)), 1.0)
    note_screening_loss(system, "project_default_principal", p)
    note_screening_loss(system, "project_default_lgd_loss", p * l)
    note_screening_loss(system, "project_default_q_weighted_lgd_loss", qq * p * l)


def borrowed_project_maturity(
    drawn_maturity: int,
    *,
    rollover_enabled: bool,
    slot: int,
    n_slots: int,
    short_share: float | None = None,
) -> int:
    """Common project-tenor mix for ON and OFF feature scenarios.

    Half of borrowed-project slots settle after one period and the remainder
    retain the drawn 20–60 day tenor. The project asset must not become safer
    merely because rollover is OFF.
    """
    drawn = max(1, int(drawn_maturity))
    n = max(1, int(n_slots))
    _ = bool(rollover_enabled)  # retained for API compatibility
    if short_share is None:
        short_share = BORROWED_PERIOD_SETTLED_SHARE
    share = min(1.0, max(0.0, float(short_share)))
    n_short = min(n, max(1, int(ceil(n * share))))
    if int(slot) < n_short:
        return 1
    return drawn


def accrue_project_negative_return_loss(system, loss: float) -> None:
    note_screening_loss(system, "project_negative_return_loss", max(0.0, float(loss)))


def _sync_project_book_amount(bank: dict, project_book: list) -> float:
    amt = float(
        sum(max(0.0, float(getattr(loan, "principal", 0.0) or 0.0)) for loan in project_book)
    )
    inv = bank.setdefault("investment", {})
    if not isinstance(inv, dict):
        bank["investment"] = {"projects": {"amount": amt}}
        return amt
    proj = inv.get("projects")
    if not isinstance(proj, dict):
        inv["projects"] = {"amount": amt}
    else:
        proj["amount"] = amt
    return amt


def unwind_project_principal_to_cash(bank: dict, project_book: list, amount: float) -> float:
    """Asset swap: project principal → liquid cash. Accounting equity unchanged."""
    need = max(0.0, float(amount))
    if need <= 1e-12 or not project_book:
        return 0.0
    raised = 0.0
    kept: list = []
    for loan in reversed(list(project_book)):
        if raised >= need - 1e-12:
            kept.append(loan)
            continue
        prin = max(0.0, float(getattr(loan, "principal", 0.0) or 0.0))
        take = min(prin, need - raised)
        if take <= 1e-12:
            kept.append(loan)
            continue
        bank["liquid_assets"] = float(bank.get("liquid_assets", 0.0) or 0.0) + take
        raised += take
        left = prin - take
        if left > 1e-12:
            loan.principal = float(left)
            kept.append(loan)
    project_book[:] = list(reversed(kept))
    _sync_project_book_amount(bank, project_book)
    return float(raised)


def cover_cash_shortfall_with_core(
    bank: dict,
    project_book: list,
    *,
    due: float,
    equity: float,
    equity_floor_amount: float = 0.0,
    period_loss: float = 0.0,
) -> float:
    """If cash < due, unwind projects only up to this period's project loss and CET1.

    Locked 20–60 day principal is not sold to repay overnight. Core pads a *loss*,
    not an ALM mismatch.
    """
    gap = max(0.0, float(due) - float(bank.get("liquid_assets", 0.0) or 0.0))
    if gap <= 1e-12:
        return 0.0
    pad = max(0.0, float(equity) - float(equity_floor_amount))
    pad = min(pad, max(0.0, float(period_loss)))
    if pad <= 1e-12:
        return 0.0
    return unwind_project_principal_to_cash(bank, project_book, min(gap, pad))


def cover_overnight_after_project_settlement(
    system,
    bank_idx: int,
    bank: dict,
    project_book: list,
    *,
    period_loss: float = 0.0,
) -> float:
    """OFF: after period P&L, core may replace project *losses* for T+1 overnight."""
    if bool(getattr(system, "rollover_enabled", True)):
        return 0.0
    from interbank_installment_rollover import (
        SCHEDULE_SINGLE_PAYMENT,
        bank_equity_proxy,
        due_amounts_by_borrower,
    )

    step = int(getattr(system, "current_step", 0))
    due = float(
        (
            due_amounts_by_borrower(
                getattr(system, "contract_book", None),
                step + 1,
                schedule_type=SCHEDULE_SINGLE_PAYMENT,
            )
            or {}
        ).get(int(bank_idx), 0.0)
        or 0.0
    )
    equity = bank_equity_proxy(bank)
    assets = max(assets_ex_equity(bank), 1e-9)
    floor_amt = float(getattr(system, "solvency_support_equity_floor", 0.0)) * assets
    return cover_cash_shortfall_with_core(
        bank,
        project_book,
        due=due,
        equity=equity,
        equity_floor_amount=floor_amt,
        period_loss=float(period_loss),
    )


def note_first_failure(system, bank_idx: int, step: int, reason: str) -> None:
    if system is None:
        return
    i = int(bank_idx)
    if i <= 0:
        return
    seen = getattr(system, "_first_failure_seen", None)
    if not isinstance(seen, set):
        seen = set()
        system._first_failure_seen = seen
    if i in seen:
        return
    seen.add(i)
    events = getattr(system, "first_failure_events", None)
    if not isinstance(events, list):
        events = []
        system.first_failure_events = events
    events.append({
        "bank_idx": i,
        "step": int(step),
        "reason": str(reason or ""),
    })


def funded_q_weighted(totals: dict | None) -> float | None:
    if not totals:
        return None
    funded = float(totals.get("funded_amount", 0.0) or 0.0)
    if funded <= 1e-12:
        return None
    return float(totals.get("funded_q_amount", 0.0) or 0.0) / funded


def screening_snapshot_fields(system) -> dict[str, float]:
    rec = getattr(system, "screening_loss_step", None) if system is not None else None
    if not isinstance(rec, dict):
        rec = empty_screening_loss()
    return {k: float(rec.get(k, 0.0) or 0.0) for k in SCREENING_LOSS_KEYS}


def json_safe_metric(value):
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return value
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return x


SCREENING_SNAPSHOT_KEYS = tuple(
    k for k in SCREENING_LOSS_KEYS if k != "interbank_writeoff_loss"
)


def screening_from_snapshots(snaps) -> dict:
    """Run-level screening totals from per-step snapshots (DEN/CEN 同口径)."""
    out: dict[str, float | None] = {}
    for key in SCREENING_SNAPSHOT_KEYS:
        out[key] = float(
            sum(float(getattr(s, key, 0.0) or 0.0) for s in (snaps or []))
        )
    out["funded_q_weighted"] = funded_q_weighted(
        {
            "funded_amount": float(out.get("funded_amount") or 0.0),
            "funded_q_amount": float(out.get("funded_q_amount") or 0.0),
        }
    )
    return out


def pack_network_summary_run(seed: int, net_summary: dict, first_failures=None) -> dict:
    packed: dict = {"seed": int(seed)}
    for key, value in (net_summary or {}).items():
        if isinstance(value, (list, dict)):
            packed[key] = value
        else:
            packed[key] = json_safe_metric(value)
    packed["first_failures"] = [
        {
            "bank_idx": int(ev.get("bank_idx", -1)),
            "step": int(ev.get("step", -1)),
            "reason": str(ev.get("reason") or ""),
        }
        for ev in (first_failures or [])
        if isinstance(ev, dict)
    ]
    return packed


def first_failure_reason_counts(events) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        reason = str(ev.get("reason") or "unknown")
        counts[reason] = int(counts.get(reason, 0)) + 1
    return counts


def diagnose_screening_channel(
    *,
    funded_q_den: float | None,
    funded_q_cen: float | None,
    project_lgd_den: float | None,
    project_lgd_cen: float | None,
    collapse_same: bool | None,
    loss_rel_tol: float = 0.05,
) -> list[str]:
    """Judgment lines only; callers must not change K / 25% cap / PD here."""
    lines: list[str] = []
    if funded_q_den is None or funded_q_cen is None:
        lines.append(
            "[diag] screening rule: funded_q 缺失（无成交）→ 跳过 GNN 风险筛选判断。"
        )
    elif float(funded_q_den) + 1e-12 >= float(funded_q_cen):
        lines.append(
            "[diag] screening rule: funded_q_DEN >= funded_q_CEN → "
            "GNN 没有产生风险筛选，应修教师标签/跨借款人竞争。"
        )
    else:
        lines.append(
            "[diag] screening rule: funded_q_DEN < funded_q_CEN → "
            "GNN 成交偏向更低 q。"
        )
        ld = float(project_lgd_den or 0.0)
        lc = float(project_lgd_cen or 0.0)
        denom = max(abs(ld), abs(lc), 1e-9)
        if abs(ld - lc) / denom <= float(loss_rel_tol):
            lines.append(
                "[diag] screening rule: 项目损失近似相同 → 风险损失通道太弱。"
                "风险损失通道太弱。长期项目期限已改为 20–60；若仍不足再单独加 PD，不要改回扣现金。"
            )
    ld = float(project_lgd_den or 0.0)
    lc = float(project_lgd_cen or 0.0)
    denom = max(abs(ld), abs(lc), 1e-9)
    den_loss_lower = (ld + 1e-12) < lc and abs(ld - lc) / denom > float(loss_rel_tol)
    if den_loss_lower and collapse_same:
        lines.append(
            "[diag] screening rule: DEN 项目损失更低，但崩溃步仍相同 → "
            "崩溃由流动性或共同冲击主导，应查首次失败原因。"
        )
    return lines


def format_screening_run_line(system, *, seed: int | None = None) -> str:
    tot = getattr(system, "screening_loss_totals", None) or empty_screening_loss()
    fq = funded_q_weighted(tot)
    fq_s = "nan" if fq is None else f"{fq:.4f}"
    fails = getattr(system, "first_failure_events", None) or []
    reasons: dict[str, int] = {}
    for ev in fails:
        r = str(ev.get("reason") or "unknown")
        reasons[r] = int(reasons.get(r, 0)) + 1
    reason_s = " ".join(f"{k}={v}" for k, v in sorted(reasons.items())) or "none"
    seed_s = f"seed={int(seed)} " if seed is not None else ""
    return (
        f"[diag] screening {seed_s}"
        f"funded_q={fq_s} "
        f"default_prin={float(tot.get('project_default_principal', 0.0)):.2f} "
        f"lgd={float(tot.get('project_default_lgd_loss', 0.0)):.2f} "
        f"q_lgd={float(tot.get('project_default_q_weighted_lgd_loss', 0.0)):.2f} "
        f"neg_ret={float(tot.get('project_negative_return_loss', 0.0)):.2f} "
        f"ib_wo={float(tot.get('interbank_writeoff_loss', 0.0)):.2f} "
        f"gap={float(tot.get('funding_gap', 0.0)):.2f} "
        f"first_fail_n={len(fails)} {reason_s}"
    )


def _nudge_deposit_breakdown(bank: dict, delta: float) -> None:
    breakdown = bank.get("liabilities_breakdown")
    if not isinstance(breakdown, dict):
        return
    deposits = float(breakdown.get("deposits", 0.0) or 0.0)
    breakdown["deposits"] = max(0.0, deposits + float(delta))


def apply_deposit_flow(bank: dict, flow_rate: float) -> float:
    """Balance-sheet-neutral deposit flow: ΔA = ΔL, so ΔE = 0.

    ``flow_rate >= 0`` is an inflow of ``|rate| * current_liabilities``.
    ``flow_rate < 0`` is a withdrawal capped by cash and deposit liabilities.
    """
    if not isinstance(bank, dict):
        return 0.0
    cash = max(0.0, float(bank.get("liquid_assets", 0.0) or 0.0))
    liabilities = max(0.0, float(bank.get("current_liabilities", 0.0) or 0.0))
    rate = float(flow_rate)
    desired = abs(rate) * liabilities
    if desired <= 1e-15:
        return 0.0

    if rate >= 0.0:
        amount = desired
        bank["liquid_assets"] = cash + amount
        bank["current_liabilities"] = liabilities + amount
        _nudge_deposit_breakdown(bank, amount)
        return amount

    amount = min(desired, cash, liabilities)
    bank["liquid_assets"] = cash - amount
    bank["current_liabilities"] = liabilities - amount
    _nudge_deposit_breakdown(bank, -amount)
    return -amount


def apply_common_liquidity_run(banks, *, multiplier: float = 1.25) -> float:
    """Equity-neutral deposit withdrawal on a common-liquidity-shock day.

    Cash and current liabilities fall by the same amount, so accounting equity
    is unchanged. Extra size is ``(multiplier-1) * outflow_rate * deposits``.
    Central banks are skipped.
    """
    extra = max(0.0, float(multiplier) - 1.0)
    if extra <= 1e-12:
        return 0.0
    withdrawn = 0.0
    for bank in banks or []:
        if not isinstance(bank, dict):
            continue
        if not bank.get("is_active", True):
            continue
        if str(bank.get("type", "")).lower() == "central":
            continue
        out_rate = max(0.0, float(bank.get("outflow_rate", 0.4) or 0.0))
        moved = apply_deposit_flow(bank, -extra * out_rate)
        withdrawn += max(0.0, -moved)
    return withdrawn


BOOL_SERIES_KEYS = ("is_auction_day", "common_liquidity_shock")

EXTRA_SERIES_KEYS = (
    "total_volume",
    "num_trades",
    "unmet_demand_rate",
    "network_density",
    "exposure_hhi",
    "max_counterparty_exposure",
    "en_unpaid_amount",
    "interbank_writeoff_amount",
    "estate_transfer_discount_amount",
    "is_auction_day",
    "funding_gap",
    "common_liquidity_shock",
    "mean_lcr",
)


def snapshot_mean_lcr(banks) -> float:
    """Active non-central mean LCR; 0 if none."""
    total = 0.0
    n = 0
    for bank in banks or []:
        if not isinstance(bank, dict):
            continue
        if not bank.get("is_active", True):
            continue
        if str(bank.get("type", "")).lower() == "central":
            continue
        total += float(bank.get("liquidity_coverage_ratio", 0.0) or 0.0)
        n += 1
    return float(total / n) if n else 0.0


def extra_series_from_snapshots(snaps, keys: tuple[str, ...] = EXTRA_SERIES_KEYS) -> dict[str, list]:
    """Per-step extra-metric paths from recorded snapshots (no edges)."""
    out: dict[str, list] = {k: [] for k in keys}
    for snap in snaps or []:
        for key in keys:
            val = getattr(snap, key, 0.0)
            if key in BOOL_SERIES_KEYS:
                out[key].append(1.0 if bool(val) else 0.0)
            else:
                out[key].append(float(val or 0.0))
    return out


def mean_extra_series_from_runs(runs, keys: tuple[str, ...] = EXTRA_SERIES_KEYS) -> dict[str, list]:
    """Cross-run mean of extra-metric paths; shock flag is the share in shock."""
    if not runs:
        return {k: [] for k in keys}
    horizon = max(len(getattr(run, "snapshots", None) or []) for run in runs)
    out: dict[str, list] = {k: [] for k in keys}
    for t in range(horizon):
        for key in keys:
            vals: list[float] = []
            for run in runs:
                snaps = getattr(run, "snapshots", None) or []
                if t >= len(snaps):
                    continue
                val = getattr(snaps[t], key, 0.0)
                if key in BOOL_SERIES_KEYS:
                    vals.append(1.0 if bool(val) else 0.0)
                else:
                    vals.append(float(val or 0.0))
            out[key].append(float(sum(vals) / len(vals)) if vals else 0.0)
    return out


def event_study_around_flags(
    flags,
    series,
    *,
    pre: int = 2,
    post: int = 5,
) -> dict:
    """Pool a series on days where ``flags >= 0.5``. Index 0 is the shock day."""
    pre_n = max(0, int(pre))
    post_n = max(0, int(post))
    taus = list(range(-pre_n, post_n + 1))
    obs: dict[int, list[float]] = {tau: [] for tau in taus}
    flag_arr = list(flags or [])
    y_arr = list(series or [])
    n = min(len(flag_arr), len(y_arr))
    n_events = 0
    for t in range(n):
        try:
            hit = float(flag_arr[t]) >= 0.5
        except (TypeError, ValueError):
            hit = bool(flag_arr[t])
        if not hit:
            continue
        n_events += 1
        for tau in taus:
            j = t + tau
            if 0 <= j < n:
                try:
                    obs[tau].append(float(y_arr[j]))
                except (TypeError, ValueError):
                    continue
    means: list[float | None] = []
    for tau in taus:
        vals = obs[tau]
        means.append(float(sum(vals) / len(vals)) if vals else None)
    return {
        "tau": taus,
        "mean": means,
        "n_obs": [len(obs[tau]) for tau in taus],
        "n_events": int(n_events),
        "obs": obs,
    }

