from dataclasses import dataclass

import numpy as np


@dataclass
class ClearingResult:
    payment: np.ndarray
    receipts: np.ndarray
    creditor_shortfall: np.ndarray
    defaults: list[int]


def run_en_clearing(
    liabilities: np.ndarray,
    banks: list[dict],
    tolerance: float = 1e-6,
    max_iterations: int = 100,
):
    """
    Eisenberg--Noe clearing using gross debtor-to-creditor
    due-payment liabilities.

    liabilities[i, j] >= 0 means that debtor i owes
    creditor j. Reciprocal obligations remain separate.
    """
    n = len(banks)

    liabilities = np.maximum(
        np.asarray(liabilities, dtype=float),
        0.0,
    ).copy()

    np.fill_diagonal(liabilities, 0.0)

    nominal_payment = liabilities.sum(axis=1)

    if nominal_payment.sum() <= 1e-9:
        zeros = np.zeros(n, dtype=float)
        return ClearingResult(
            zeros,
            zeros,
            zeros,
            [],
        )

    relative_liabilities = np.divide(
        liabilities,
        nominal_payment[:, None],
        out=np.zeros_like(liabilities),
        where=nominal_payment[:, None] > 0.0,
    )

    external_assets = np.array(
        [
            float(bank.get("liquid_assets", 0.0))
            for bank in banks
        ],
        dtype=float,
    )

    payment = nominal_payment.copy()

    for _ in range(max_iterations):
        updated = np.minimum(
            nominal_payment,
            relative_liabilities.T @ payment
            + external_assets,
        )

        if np.max(
            np.abs(updated - payment)
        ) < tolerance:
            payment = updated
            break

        payment = updated

    receipts = relative_liabilities.T @ payment

    defaults = [
        i
        for i in range(n)
        if payment[i]
        < nominal_payment[i] - tolerance
    ]

    for i, bank in enumerate(banks):
        bank["liquid_assets"] = max(
            0.0,
            external_assets[i] - payment[i] + receipts[i],
        )

    # Creditor j's loss is the unpaid part of claims owed *to j*.
    # The debtor recovery rate scales every bilateral liability row.
    recovery_rate = np.divide(
        payment,
        nominal_payment,
        out=np.ones_like(payment),
        where=nominal_payment > 0.0,
    )
    realized_bilateral = liabilities * recovery_rate[:, None]
    creditor_shortfall = (
        liabilities - realized_bilateral
    ).sum(axis=0)

    return ClearingResult(
        payment,
        receipts,
        creditor_shortfall,
        defaults,
    )


def accounting_equity(bank: dict) -> float:
    project_assets = float(
        bank.get("investment", {}).get("projects", {}).get("amount", 0.0)
    )
    total_assets = (
        float(bank.get("liquid_assets", 0.0))
        + float(bank.get("interbank_assets", 0.0))
        + project_assets
    )
    total_liabilities = (
        float(bank.get("current_liabilities", 0.0))
        + float(bank.get("interbank_liabilities", 0.0))
        + float(bank.get("termed_out_liabilities", 0.0))
    )
    return total_assets - total_liabilities


def classify_negative_equity_defaults(
    banks: list[dict],
    bank_types: list[str],
) -> list[int]:
    defaults = []

    for i, bank in enumerate(banks):
        if bank_types[i] == "central" or not bank.get("is_active", True):
            continue

        # Use untruncated accounting equity for insolvency detection.
        equity = accounting_equity(bank)

        if equity < 0.0:
            # The resolution batch handles recoveries and absorbing exit.
            defaults.append(i)

    return defaults


def refresh_capital_after_clearing(
    banks: list[dict],
    creditor_shortfall: np.ndarray,
) -> None:
    """
    This function is called after processed contracts have been removed and
    interbank positions have been rebuilt from the remaining contract book.

    ON arrears remain in the updated contract book. Default resolution
    removes written-off claims and replaces receivables by recovered cash.
    Recomputed equity reflects these book changes; the unpaid-due vector
    is not itself treated as an additional realized loss.
    """
    for i, bank in enumerate(banks):
        # Unpaid due claims are diagnostic; ON arrears remain in the book.
        # They are not an additional equity deduction here.
        equity = accounting_equity(bank)
        bank["core_capital"] = max(0.0, float(equity))


def regulatory_rwa(bank: dict) -> float:
    interbank_assets = float(bank.get("interbank_assets", 0.0))
    project_assets = float(
        bank.get("investment", {}).get("projects", {}).get("amount", 0.0)
    )
    total_liabilities = max(0.0,
        float(bank.get("current_liabilities", 0.0))
        + float(bank.get("interbank_liabilities", 0.0))
        + float(bank.get("termed_out_liabilities", 0.0)))
    rwa = 0.5 * interbank_assets + project_assets
    return max(rwa, 0.08 * total_liabilities)


def measured_car(bank: dict) -> float:
    capital = max(0.0, float(bank.get("core_capital", 0.0)))
    rwa = regulatory_rwa(bank)
    if rwa <= 0.0:
        return 0.0
    return float(np.clip(capital / (rwa + 1e-9), 0.0, 1.5))


def systemic_risk(
    banks: list[dict],
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    car_threshold: float = 0.08,
) -> tuple[float, float, float, float]:
    noncentral = [
        bank
        for bank in banks
        if bank.get("type") != "central"
        and bank.get("name") != "CentralBank"
    ]
    count = max(1, len(noncentral))

    failure_rate = sum(
        not bank.get("is_active", True) for bank in noncentral
    ) / count

    active = [bank for bank in noncentral if bank.get("is_active", True)]
    delta = 0.0005
    capital_breach_share = sum(
        measured_car(bank) < car_threshold - 1e-9
        or (
            bank.get("lag_car") is not None
            and float(bank["lag_car"]) < car_threshold - 1e-9
            and measured_car(bank) < car_threshold + delta - 1e-9
        )
        for bank in active
    ) / max(1, len(active))

    capital_gap = 0.0
    required_total = 0.0
    for bank in noncentral:
        if bank.get("is_active", True):
            required_capital = car_threshold * regulatory_rwa(bank)
            actual_capital = max(0.0, float(bank.get("core_capital", 0.0)))
        else:
            required_capital = float(bank.get("required_capital_at_failure", 0.0))
            actual_capital = 0.0
        capital_gap += max(0.0, required_capital - actual_capital)
        required_total += required_capital

    capital_gap_ratio = capital_gap / (required_total + 1e-9)

    w1, w2, w3 = weights
    risk = np.clip(
        w1 * failure_rate
        + w2 * capital_breach_share
        + w3 * capital_gap_ratio,
        0.0,
        1.0,
    )

    return (
        float(risk),
        float(failure_rate),
        float(capital_breach_share),
        float(capital_gap_ratio),
    )
