from dataclasses import dataclass

import numpy as np

SCHEDULE_SINGLE_PAYMENT = "single_payment"
SCHEDULE_INSTALLMENT = "installment"


@dataclass
class Contract:
    contract_id: str
    lender_idx: int
    borrower_idx: int
    principal: float
    rate: float
    created_step: int
    maturity_step: int
    schedule_type: str = SCHEDULE_SINGLE_PAYMENT
    remaining_principal: float = 0.0
    coupon_rate: float = 0.0
    tenor_total: int = 1
    periods_paid: int = 0
    settlement_rate: float = 0.0
    arrears_due: float = 0.0


def tenor_from_principal(
    principal: float,
    min_tenor: int = 5,
    max_tenor: int = 20,
    ref_small: float = 50.0,
    ref_large: float = 2000.0,
) -> int:
    """Map originated principal to the current 5--20 business-day tenor."""
    p = max(0.0, float(principal))
    if p <= ref_small:
        return int(min_tenor)
    if p >= ref_large:
        return int(max_tenor)
    share = (p - ref_small) / (ref_large - ref_small)
    return int(round(min_tenor + share * (max_tenor - min_tenor)))


def decide_origination_schedule(
    principal: float,
    settlement_rate: float,
    rollover_enabled: bool,
    min_tenor: int = 5,
    max_tenor: int = 20,
) -> dict:
    """
    Current contract rule used by the formal experiments.

    Rollover ON  -> installment at origination (5--20 periods).
    Rollover OFF -> next-day single payment of principal and interest.
    There is no 10-day bullet stage and no maturity-date conversion.
    """
    P = max(0.0, float(principal))
    r = float(settlement_rate)
    if rollover_enabled:
        tenor = tenor_from_principal(P, min_tenor=min_tenor, max_tenor=max_tenor)
        return {
            "schedule_type": SCHEDULE_INSTALLMENT,
            "tenor": int(tenor),
            "settlement_rate": r,
            "maturity_in_periods": int(tenor),
        }
    return {
        "schedule_type": SCHEDULE_SINGLE_PAYMENT,
        "tenor": 1,
        "settlement_rate": r,
        "maturity_in_periods": 1,
    }


def due_amount(contract: Contract) -> float:
    if contract.schedule_type == SCHEDULE_SINGLE_PAYMENT:
        return float(contract.principal) * (1.0 + float(contract.settlement_rate))
    remaining = max(0.0, float(contract.remaining_principal))
    left = max(1, int(contract.tenor_total) - int(contract.periods_paid))
    interest = float(contract.coupon_rate) * remaining
    principal = remaining if left == 1 else remaining / float(left)
    return float(contract.arrears_due) + interest + principal


def aggregate_gross_due_matrix(contracts, n_banks: int, step: int) -> np.ndarray:
    """Build a non-negative debtor-to-creditor due-payment matrix."""
    L = np.zeros((n_banks, n_banks), dtype=float)
    for contract in contracts:
        if int(contract.maturity_step) <= int(step) and contract.schedule_type == SCHEDULE_SINGLE_PAYMENT:
            due = due_amount(contract)
        elif contract.schedule_type == SCHEDULE_INSTALLMENT:
            due = due_amount(contract)
        else:
            continue
        i = int(contract.borrower_idx)
        j = int(contract.lender_idx)
        if i != j and due > 0.0:
            L[i, j] += due
    np.fill_diagonal(L, 0.0)
    return L


def assign_roles_by_risk(banks, car_cutoff: float = 0.08, lcr_cutoff: float = 0.85):
    """Current daily role assignment: lender / borrower / non-participant."""
    roles = []
    for bank in banks:
        if (not bank.get("is_active", True)) or bank.get("type") == "central":
            roles.append("non_participant")
            continue
        car = float(bank.get("capital_adequacy_ratio", 0.0))
        lcr = float(bank.get("liquidity_coverage_ratio", 0.0))
        if car >= car_cutoff and lcr >= lcr_cutoff:
            roles.append("lender")
        elif lcr < lcr_cutoff:
            roles.append("borrower")
        else:
            roles.append("non_participant")
    return roles
