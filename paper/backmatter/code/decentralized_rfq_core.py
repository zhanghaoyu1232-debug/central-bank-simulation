DAILY_RFQ_MAX_ROUNDS = 4
RFQ_CANDIDATE_K = 4
LOCAL_UNIQUE_LENDER_BUDGET = 8
RFQ_RANDOM_EXPLORE_N = 1
MAX_LOCAL_DEGREE = 12


class RFQMarket:
    """
    Current formal RFQ: local discovery, four rounds, K = 4, partial funding.

    Outstanding rollover lenders are kept in the pool and do not consume the
    new-discovery budget of eight lenders. At most one random explorer is
    added. Acceptable bilateral trades are retained even when they do not
    cover the borrower's complete demand.
    """

    def __init__(self, max_rounds: int = DAILY_RFQ_MAX_ROUNDS, min_trade_size: float = 1e-9):
        self.max_rounds = int(max_rounds)
        self.min_trade_size = float(min_trade_size)

    def run(self, intentions, banks, system, step: int, B_max: float = 1200.0, K: int = RFQ_CANDIDATE_K):
        lenders = [x for x in intentions if x.role == "lender"]
        borrowers = [x for x in intentions if x.role == "borrower"]
        supply_left = {x.bank_idx: float(x.quantity) for x in lenders}
        demand_left = {x.bank_idx: float(x.quantity) for x in borrowers}
        lender_ids = [x.bank_idx for x in lenders]
        trades = []

        local_pools = {}
        rollover_by_borrower = {}
        exhausted = {}
        queried = {}
        for borrower in borrowers:
            j = int(borrower.bank_idx)
            pool, _hist, _roll = build_local_lender_pool(
                lender_ids,
                j,
                system,
                budget=int(getattr(system, "local_unique_lender_budget", LOCAL_UNIQUE_LENDER_BUDGET)),
                explore_n=int(getattr(system, "rfq_random_explore_n", RFQ_RANDOM_EXPLORE_N)),
                banks=banks,
                supply_left=supply_left,
            )
            local_pools[j] = pool
            rollover_by_borrower[j] = set(_roll)
            exhausted[j] = set()
            queried[j] = set()

        for _round in range(self.max_rounds):
            for borrower in borrowers:
                j = int(borrower.bank_idx)
                need = float(demand_left.get(j, 0.0))
                if need < self.min_trade_size:
                    continue
                pool = [i for i in local_pools.get(j, []) if i not in exhausted[j]]
                # Later rounds contact fresh local names before revisiting a
                # non-exhausted partial quote. Rollover lenders lead round 1.
                pool.sort(key=lambda i: (
                    0 if i in rollover_by_borrower[j] and i not in queried[j]
                    else 1 if i not in queried[j]
                    else 2
                ))
                candidates = pool[: int(K)]
                queried[j].update(candidates)
                for lender in candidates:
                    if lender == j:
                        continue
                    supp = float(supply_left.get(lender, 0.0))
                    if supp < self.min_trade_size:
                        exhausted[j].add(lender)
                        continue
                    amount = min(float(B_max), supp, need)
                    if amount < self.min_trade_size:
                        continue
                    if current_degree(system, lender, j) >= MAX_LOCAL_DEGREE:
                        continue
                    if not gnn_accept(system, lender, j, amount):
                        exhausted[j].add(lender)
                        continue
                    trades.append((lender, j, amount))
                    supply_left[lender] -= amount
                    demand_left[j] -= amount
                    need = float(demand_left[j])
                    if need < self.min_trade_size:
                        break
        return trades


def build_local_lender_pool(
    lender_ids,
    borrower_idx: int,
    system,
    budget: int,
    explore_n: int,
    banks,
    supply_left,
):
    """Build the fixed daily local pool; never append all untried banks."""
    j = int(borrower_idx)
    eligible = {int(i) for i in lender_ids if int(i) != j}
    rollover = list(system.outstanding_lenders_for_borrower(j))
    hist = list(getattr(system, "counterparty_history", {}).get(j, []) or [])
    acquaintances = set(getattr(system, "initial_acquaintances", {}).get(j, []) or [])
    hop1 = acquaintances | set(system.exposure_neighbors(j))
    hop2 = set()
    for neighbour in hop1:
        hop2.update(getattr(system, "initial_acquaintances", {}).get(neighbour, []) or [])
        hop2.update(system.exposure_neighbors(neighbour))
    hop2.discard(j)
    pool = []
    seen = set()

    def add(i):
        i = int(i)
        if i in seen or i not in eligible:
            return
        if not banks[i].get("is_active", True) or supply_left.get(i, 0.0) <= 1e-9:
            return
        seen.add(i)
        pool.append(i)

    # Outstanding lenders do not consume the discovery budget.
    for i in rollover:
        add(i)
    rollover_kept = set(pool)

    # Newly discovered names are bounded and preserve source priority.
    for group in (hist, list(hop1), list(hop2)):
        system.rng_matching.shuffle(group)
        for i in group:
            if sum(x not in rollover_kept for x in pool) >= int(budget):
                break
            add(i)

    # At most one whole-market random explorer is allowed.
    remaining = [i for i in eligible if i not in seen]
    system.rng_matching.shuffle(remaining)
    for i in remaining[: max(0, min(int(explore_n), 1))]:
        if sum(x not in rollover_kept for x in pool) < int(budget):
            add(i)
    return pool, set(hist), list(rollover_kept)


def current_degree(system, lender: int, borrower: int) -> int:
    return int(getattr(system, "degree", {}).get((int(lender), int(borrower)), 0))


def gnn_accept(system, lender: int, borrower: int, amount: float) -> bool:
    matcher = getattr(system, "gnn_matcher", None)
    if matcher is None:
        return True
    return bool(matcher.accept(int(lender), int(borrower), float(amount)))
