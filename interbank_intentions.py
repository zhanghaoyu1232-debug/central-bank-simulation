"""
共用同业意图收集：centralized 与 decentralized(RFQ) 必须使用同一套 collect_intentions，
仅替换撮合函数，以便对比只反映 matching 机制差异。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interbank_installment_rollover import (
    ROLLOVER_BORROW_COUPON_CLEARED,
    DEFAULT_DEBT_BURDEN_KAPPA,
    DEFAULT_IBL_CAP_ASSET_LAMBDA,
    opportunity_borrow_phi,
    rollover_borrow_quantity,
)
from bank_econ_shared import car_below_threshold

DAILY_INTERBANK_INTENTION_LCR_TARGET = 0.85
DAILY_OPPORTUNITY_MARGIN = 0.00012
DAILY_SWITCH_HYSTERESIS = 0.00004
DAILY_OPPORTUNITY_BORROW_SCALE = 0.25
DAILY_RFQ_QUOTE_SPREAD = 0.00005
DAILY_BORROW_SPREAD = 0.00005
DAILY_PROJECT_RETURN_DEFAULT = 0.00008
DAILY_PROJECT_RISK_DEFAULT = 0.00016
# Borrower willingness-to-pay is driven by funding urgency/debt stress.  Risk
# remains a lender-side screening variable in DEN; CEN deliberately ignores it.
DAILY_LIQUIDITY_URGENCY_MARKUP = 0.00018
DAILY_DEBT_STRESS_MARKUP = 0.00010
DAILY_BORROWER_WTP_MAX_MARKUP = 0.00035


@dataclass
class Intention:
    """银行当步意图：角色 + 保留价 + 数量。"""
    bank_idx: int
    role: str  # "lender" | "borrower"
    reserve_bid: float
    reserve_ask: float
    quantity: float


def _intentions_to_arrays(intentions: list[Intention]):
    """intentions -> lenders, borrowers, supply, demand, r_min, r_max。"""
    lenders, borrowers = [], []
    supply, demand = [], []
    r_min, r_max = {}, {}
    for it in intentions:
        i = int(it.bank_idx)
        if str(it.role).lower() == "lender":
            lenders.append(i)
            supply.append(float(it.quantity))
            r_min[i] = float(it.reserve_bid)
        else:
            borrowers.append(i)
            demand.append(float(it.quantity))
            r_max[i] = float(it.reserve_ask)
    return lenders, borrowers, supply, demand, r_min, r_max


def _expected_project_return(bank, step, *, default_mu=DAILY_PROJECT_RETURN_DEFAULT):
    return float(bank.get("proj_mu", default_mu))


def _project_risk_proxy(bank, *, default_sigma=DAILY_PROJECT_RISK_DEFAULT):
    return float(bank.get("proj_sigma", default_sigma))


def _risk_penalty(bank, lam=0.5):
    return lam * _project_risk_proxy(bank)


def _expected_borrow_rate(base_rate, bank, *, add_spread=None, last_avg_rate=None):
    if last_avg_rate is not None:
        return float(last_avg_rate)
    spread = add_spread if add_spread is not None else bank.get("borrow_spread", DAILY_BORROW_SPREAD)
    return base_rate + float(spread)


def _expected_lend_rate(base_rate, bank, *, add_spread=None):
    spread = add_spread if add_spread is not None else bank.get("lend_spread", 0.00)
    return base_rate + float(spread)


def _borrower_liquidity_wtp(
    base_rate: float,
    *,
    liquid_assets: float,
    target_liquidity: float,
    debt_phi: float,
) -> float:
    """Maximum liquidity-funding rate; monotone in urgency and debt stress."""
    target = max(float(target_liquidity), 1e-9)
    urgency = float(np.clip((target - float(liquid_assets)) / target, 0.0, 1.0))
    debt_stress = float(np.clip(1.0 - float(debt_phi), 0.0, 1.0))
    markup = (
        DAILY_RFQ_QUOTE_SPREAD
        + DAILY_LIQUIDITY_URGENCY_MARKUP * urgency
        + DAILY_DEBT_STRESS_MARKUP * debt_stress
    )
    return float(base_rate) + min(float(markup), DAILY_BORROWER_WTP_MAX_MARKUP)


def collect_intentions(
    banks: list,
    n: int,
    roles: np.ndarray,
    reserve_buffer: np.ndarray,
    base_rate: float,
    lcr_target: float = DAILY_INTERBANK_INTENTION_LCR_TARGET,
    opportunity_borrow: bool = True,
    opportunity_lend: bool = True,
    opp_margin: float = DAILY_OPPORTUNITY_MARGIN,
    opp_risk_lambda: float = 0.5,
    opp_borrow_scale: float = DAILY_OPPORTUNITY_BORROW_SCALE,
    opp_lend_scale: float = 0.5,
    hard_liquidity_floor: float = 0.0,
    step: int = 0,
    last_avg_rate: float | None = None,
    switch_hysteresis: float = DAILY_SWITCH_HYSTERESIS,
    rollover_blocked: set[int] | None = None,
    rollover_borrow_policy: str = ROLLOVER_BORROW_COUPON_CLEARED,
    coupon_cleared_borrowers: set[int] | None = None,
    coupon_due_borrowers: set[int] | None = None,
    car_cutoff: float = 0.08,
    debt_burden_kappa: float = DEFAULT_DEBT_BURDEN_KAPPA,
    ibl_cap_asset_lambda: float = DEFAULT_IBL_CAP_ASSET_LAMBDA,
) -> list[Intention]:
    """根据 roles 与流动性计算每家银行的 intention（与 RFQ / centralized 共用）。"""
    intentions = []
    for i in range(n):
        if i == 0:
            continue
        b = banks[i]
        if not b.get("is_active", True):
            continue
        liq = float(b.get("liquid_assets", 0.0))
        current_lia = float(b.get("current_liabilities", 0.0))
        interbank_lia = float(b.get("interbank_liabilities", 0.0))
        total_lia = current_lia + interbank_lia
        reserve_need = float(reserve_buffer[i] * current_lia)
        stress_outflow = total_lia * float(b.get("outflow_rate", 0.4))
        target_liq = max(reserve_need, float(lcr_target) * stress_outflow)
        loan_rt = float(b.get("loan_interest_rate", base_rate))
        car = float(b.get("capital_adequacy_ratio", 0.0))
        car_low = car_below_threshold(car, car_cutoff)

        roi = _expected_project_return(b, step)
        pen = _risk_penalty(b, lam=opp_risk_lambda)
        r_hat = _expected_borrow_rate(base_rate, b, last_avg_rate=last_avg_rate)
        r_lend = _expected_lend_rate(base_rate, b)
        effective_role = int(roles[i])
        if effective_role == +1 and opportunity_borrow and (roi - r_hat) > (opp_margin + pen + switch_hysteresis):
            effective_role = -1
        if (
            effective_role == -1
            and opportunity_lend
            and not car_low
            and (r_lend - (roi + pen)) > (opp_margin + switch_hysteresis)
        ):
            effective_role = +1
        if effective_role == +1:
            if car_low:
                continue
            avail = max(0.0, liq - target_liq)
            phi = float(b.get("risk_appetite", 0.5))
            avail *= (0.6 + 0.4 * phi)
            extra_liq = avail
            extra_ret = 0.0
            if opportunity_lend and (r_lend - (roi + pen)) > opp_margin:
                investable = max(0.0, liq - max(hard_liquidity_floor, 0.0))
                extra_ret = opp_lend_scale * investable
            quantity = extra_liq + extra_ret
            safe_lend_cap = max(0.0, liq - target_liq)
            quantity = min(quantity, safe_lend_cap)
            if quantity > 1e-6:
                reserve_bid = loan_rt
                if opportunity_lend and extra_ret > 0:
                    reserve_bid = min(reserve_bid, base_rate)
                intentions.append(Intention(
                    bank_idx=i, role="lender",
                    reserve_bid=reserve_bid, reserve_ask=loan_rt + DAILY_RFQ_QUOTE_SPREAD,
                    quantity=quantity,
                ))
        elif effective_role == -1:
            gap = max(0.0, target_liq - liq)
            phi = float(b.get("risk_appetite", 0.5))
            extra = 0.08 * total_lia * phi
            need_liq = min(gap + extra, 0.5 * total_lia)
            need_inv = 0.0
            if opportunity_borrow and (roi - r_hat) > (opp_margin + pen):
                need_inv = min(opp_borrow_scale * total_lia, 0.5 * total_lia)
            quantity = rollover_borrow_quantity(
                i,
                b,
                base_rate,
                need_liq,
                need_inv,
                rollover_blocked=rollover_blocked,
                coupon_cleared=coupon_cleared_borrowers,
                coupon_due=coupon_due_borrowers,
                borrow_policy=rollover_borrow_policy,
                opportunity_borrow=opportunity_borrow,
                opp_margin=opp_margin,
                opp_risk_lambda=opp_risk_lambda,
                opp_borrow_scale=opp_borrow_scale,
                last_avg_rate=last_avg_rate,
                debt_burden_kappa=float(
                    b.get("debt_burden_kappa", debt_burden_kappa)
                ),
                ibl_cap_asset_lambda=float(
                    b.get("ibl_cap_asset_lambda", ibl_cap_asset_lambda)
                ),
            )
            if quantity > 1e-6:
                reserve_bid = loan_rt - DAILY_BORROW_SPREAD
                reserve_ask = loan_rt + DAILY_RFQ_QUOTE_SPREAD
                debt_phi = opportunity_borrow_phi(
                    b,
                    kappa=float(b.get("debt_burden_kappa", debt_burden_kappa)),
                )
                reserve_ask = max(
                    reserve_ask,
                    _borrower_liquidity_wtp(
                        base_rate,
                        liquid_assets=liq,
                        target_liquidity=target_liq,
                        debt_phi=debt_phi,
                    ),
                )
                if opportunity_borrow and need_inv > 0:
                    # Project risk controls whether/how much the bank borrows;
                    # it must not mechanically make a riskier borrower quote a
                    # lower maximum rate (which erased CEN adverse selection).
                    r_max = max(base_rate, roi - opp_margin)
                    reserve_ask = max(reserve_ask, r_max)
                intentions.append(Intention(
                    bank_idx=i, role="borrower",
                    reserve_bid=reserve_bid, reserve_ask=reserve_ask,
                    quantity=quantity,
                ))
    return intentions


def exposure_network_metrics(
    gross_exposure_matrix: np.ndarray,
    *,
    exclude_central_bank: bool = True,
) -> dict:
    """
    Network statistics based on gross outstanding creditor-to-debtor
    exposures.

    G[i, j] > 0 means that lender i has an outstanding claim
    on borrower j. The central bank is excluded from ordinary
    interbank-network statistics by default.
    """
    G = np.maximum(
        np.asarray(gross_exposure_matrix, dtype=float),
        0.0,
    ).copy()

    np.fill_diagonal(G, 0.0)

    if exclude_central_bank and G.shape[0] > 1:
        G = G[1:, 1:]

    n = G.shape[0]
    eps = 1e-12

    directed_adj = G > eps

    active_links = int(np.sum(directed_adj))

    max_possible = n * (n - 1)
    density = (
        active_links / max_possible
        if max_possible > 0
        else 0.0
    )

    out_degree = np.sum(directed_adj, axis=1)
    in_degree = np.sum(directed_adj, axis=0)
    total_degree = out_degree + in_degree

    mean_degree = (
        float(np.mean(total_degree))
        if n > 0 else 0.0
    )
    degree_std = (
        float(np.std(total_degree, ddof=0))
        if n > 0 else 0.0
    )

    weighted_degree = (
        np.sum(G, axis=1)
        + np.sum(G, axis=0)
    )

    mean_weighted_degree = (
        float(np.mean(weighted_degree))
        if n > 0 else 0.0
    )
    weighted_degree_std = (
        float(np.std(weighted_degree, ddof=0))
        if n > 0 else 0.0
    )

    exposures = G[directed_adj]

    if exposures.size > 0:
        total_exposure = float(np.sum(exposures))
        shares = exposures / total_exposure

        exposure_hhi = float(
            np.sum(shares ** 2)
        )

        largest_bilateral_exposure = float(
            np.max(exposures)
        )
    else:
        exposure_hhi = 0.0
        largest_bilateral_exposure = 0.0

    # Weakly connected components
    weak_adj = directed_adj | directed_adj.T
    visited = np.zeros(n, dtype=bool)
    largest_component_size = 0

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        size = 0

        while stack:
            node = stack.pop()
            size += 1

            neighbours = np.where(
                weak_adj[node]
            )[0]

            for neighbour in neighbours:
                neighbour = int(neighbour)

                if not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)

        largest_component_size = max(
            largest_component_size,
            size,
        )

    largest_component_share = (
        largest_component_size / n
        if n > 0
        else 0.0
    )

    return {
        "active_links": float(active_links),
        "network_density": float(density),
        "mean_degree": mean_degree,
        "degree_std": degree_std,
        "mean_weighted_degree": mean_weighted_degree,
        "weighted_degree_std": weighted_degree_std,
        "exposure_hhi": exposure_hhi,
        "max_counterparty_exposure":
            largest_bilateral_exposure,
        "largest_component_size":
            float(largest_component_size),
        "largest_component_share":
            float(largest_component_share),
    }


def undirected_edge_set(exposure_matrix: np.ndarray, eps: float = 1e-12) -> set[tuple[int, int]]:
    L = np.asarray(exposure_matrix, dtype=float)
    n = L.shape[0]
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if abs(float(L[i, j])) > eps:
                edges.add((i, j))
    return edges


def edge_jaccard(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union > 0 else 0.0
