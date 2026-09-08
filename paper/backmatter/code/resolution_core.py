from interbank_resolution import _is_payable_creditor, _pay_creditor


def creditor_recovery_phase(banks, failed, borrower_claims, lgd):
    """Condensed creditor-payment phase of the resolution batch.

    The failure cohort and contract claims are already frozen.
    Failed-lender claim transfers have already added their proceeds
    to estate cash. borrower_claims[i] contains (lender, claim, contract).
    """
    failed_set = set(failed)
    results = {i: {"recovery": 0.0, "creditor_writeoff": 0.0}
               for i in failed}
    recorded = {}

    def record_writeoff(borrower, contract, amount):
        cid = getattr(contract, "contract_id", None)
        key = ("contract_id", str(cid)) if cid is not None else (
            "object_id", id(contract)
        )
        loss = max(0.0, float(amount))
        previous = float(recorded.get(key, 0.0))
        increment = max(0.0, loss - previous)
        if increment > 1e-12:
            recorded[key] = loss
            results[borrower]["creditor_writeoff"] += increment

    for borrower in failed:
        payable, skipped = [], []
        for lender, claim, contract in borrower_claims[borrower]:
            if lender in failed_set or not _is_payable_creditor(
                banks[lender], lender
            ):
                skipped.append((lender, claim, contract))
            else:
                payable.append((lender, claim, contract))

        total_claim = sum(claim for _, claim, _ in payable)
        estate_cash = max(0.0, float(banks[borrower]["liquid_assets"]))
        recovery_cap = total_claim * (1.0 - lgd)
        pool = min(estate_cash, recovery_cap) if total_claim > 1e-12 else 0.0
        banks[borrower]["liquid_assets"] = estate_cash - pool

        for lender, claim, contract in payable:
            share = (pool * claim / total_claim
                     if total_claim > 1e-12 and pool > 1e-12 else 0.0)
            recovered, unpaid = _pay_creditor(
                banks, lender, share, claim=claim
            )
            results[borrower]["recovery"] += recovered
            record_writeoff(borrower, contract, unpaid)
        for lender, claim, contract in skipped:
            record_writeoff(borrower, contract, claim)

    # The enclosing resolution batch removes the original contracts,
    # adds central-bank replacement claims, and rebuilds IB positions.
    # Equity then reflects recovered cash and the removal of claims.
    # creditor_writeoff is a diagnostic; it is not deducted again.
    return results
