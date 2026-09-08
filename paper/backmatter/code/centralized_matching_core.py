import numpy as np


def is_auction_day(step: int, cycle_length: int = 1) -> bool:
    """Formal centralized matching uses cycle_length = 1, i.e. every business day."""
    cycle_length = max(1, int(cycle_length))
    return ((int(step) + 1) % cycle_length) == 0


def queue_latest_orders(order_book: dict[int, dict], intentions, step: int) -> None:
    """Each bank keeps only its most recent intention; quantities are overwritten."""
    for intention in intentions:
        i = int(intention.bank_idx)
        order_book[i] = {
            "bank_idx": i,
            "role": str(intention.role).lower(),
            "quantity": float(intention.quantity),
            "reserve_bid": float(getattr(intention, "reserve_bid", 0.0)),
            "reserve_ask": float(getattr(intention, "reserve_ask", 0.0)),
            "step_submitted": int(step),
        }


def centralized_rate_matching(
    lenders: list[int],
    borrowers: list[int],
    supply: np.ndarray,
    demand: np.ndarray,
    degree: np.ndarray,
    max_degree: int,
    trade_cap: float,
    r_min: dict[int, float],
    r_max: dict[int, float],
) -> list[tuple[int, int, float]]:
    """
    Rate-prioritized allocation with a borrower-level all-or-nothing rule.
    Provisional edges are committed only if the borrower's complete demand
    can be assembled; otherwise that borrower remains unfunded.
    """
    eps = 1e-8
    supply_left = np.asarray(supply, dtype=float).copy()
    demand_left = np.asarray(demand, dtype=float).copy()
    degree_left = np.asarray(degree, dtype=int).copy()
    plan: list[tuple[int, int, float]] = []

    # Screen reservation-rate-incompatible pairs and rank eligible
    # borrower/lender combinations by their feasible midpoint rate.
    eligible = {b: [] for b in borrowers}
    best_rate = {b: -np.inf for b in borrowers}
    for lender_pos, lender in enumerate(lenders):
        for borrower in borrowers:
            if lender == borrower or r_min[lender] > r_max[borrower] + eps:
                continue
            rate = 0.5 * (r_min[lender] + r_max[borrower])
            eligible[borrower].append((rate, lender_pos, lender))
            best_rate[borrower] = max(best_rate[borrower], rate)

    borrower_order = sorted(
        range(len(borrowers)),
        key=lambda pos: best_rate[borrowers[pos]],
        reverse=True,
    )
    for borrower_pos in borrower_order:
        borrower = borrowers[borrower_pos]
        remaining = float(demand_left[borrower_pos])
        if remaining <= eps:
            continue
        provisional = []
        candidates = sorted(eligible[borrower], reverse=True)
        for _rate, lender_pos, lender in candidates:
            if degree_left[lender] >= max_degree:
                continue
            if degree_left[borrower] + len(provisional) >= max_degree:
                break
            amount = min(float(trade_cap), float(supply_left[lender_pos]), remaining)
            if amount <= eps:
                continue
            provisional.append((lender, borrower, amount, lender_pos))
            remaining -= amount
            if remaining <= eps:
                break
        if remaining > eps:
            continue
        for lender, borrower, amount, lender_pos in provisional:
            plan.append((lender, borrower, float(amount)))
            supply_left[lender_pos] -= amount
            degree_left[lender] += 1
        degree_left[borrower] += len(provisional)
        demand_left[borrower_pos] = 0.0
    return plan


def centralized_daily_step(system, intentions, step: int) -> list:
    """Current formal CEN: refresh orders and match every business day."""
    queue_latest_orders(system.central_order_book, intentions, step)
    cycle = int(getattr(system, "centralized_cycle_length", 1))
    if not is_auction_day(step, cycle):
        return []
    confirmed = system.reconfirm_central_order_quantities(intentions)
    lenders, borrowers, supply, demand, r_min, r_max = (
        system.aggregate_central_order_book(confirmed)
    )
    plan = centralized_rate_matching(
        lenders,
        borrowers,
        np.asarray(supply, dtype=float),
        np.asarray(demand, dtype=float),
        system.current_network_degree(),
        system.max_degree,
        system.trade_cap,
        r_min,
        r_max,
    )
    system.central_order_book.clear()
    return system.plan_to_trades(plan, step)
