# -*- coding: utf-8 -*-
"""Absorbing-default resolution: estate transfer + pro-rata creditor recovery.

Two-phase batch clearing (recommended when several banks fail in one step):
  Phase A — freeze the failure set on a shared claim snapshot
  Phase B — monetize failed-lender claims into estates, then pro-rata pay creditors

Primary loss metric:
  * ``creditor_writeoff`` — amount creditors cannot recover because the *debtor* defaulted
    (deduped by contract within one resolution batch).

Auxiliary metric:
  * ``estate_transfer_discount`` — gap between failed-lender claim book value and the price
    CB pays into the estate when transferring a live borrower's claim. Not debtor-default loss.

Rules:
  * Central bank (idx 0) is a payable creditor; its claims must not vanish.
  * Failed-lender claims transfer to CB for a recovery-priced consideration that
    enters the failed bank's liquidation cash pool.
  * ``resolution_estate_claims`` rises on transfer and falls when those claims
    are repaid or written off in a subsequent default.
  * Never credit liquid assets to already-failed / frozen commercial banks.
  * When lender and borrower both fail in the same batch, creditor_writeoff is
    recorded once under the borrower (B2), never again under the lender (B1).
"""
from __future__ import annotations

from typing import Any, Callable, Iterable


def _is_central(bank: dict, bank_idx: int) -> bool:
    return int(bank_idx) == 0 or bank.get("type") == "central" or bank.get("name") == "CentralBank"


def _is_payable_creditor(bank: dict, bank_idx: int = -1) -> bool:
    """CB always receives recovery; frozen/failed commercial banks do not."""
    if _is_central(bank, bank_idx):
        return True
    if not bank.get("is_active", True):
        return False
    if bank.get("absorbing_default") or bank.get("balance_sheet_frozen"):
        return False
    return True


def _adjust_estate_claims(cb: dict, delta: float) -> None:
    cur = float(cb.get("resolution_estate_claims", 0.0))
    cb["resolution_estate_claims"] = max(0.0, cur + float(delta))


def _pay_creditor(
    banks: list,
    lender_idx: int,
    amount: float,
    *,
    claim: float,
) -> tuple[float, float]:
    """Credit recovery to lender; reduce CB estate stock when lender is CB.

    Returns (recovery_paid, unpaid_claim).
    """
    pay = max(0.0, float(amount))
    claim = max(0.0, float(claim))
    unpaid = max(0.0, claim - pay)
    if pay > 1e-15:
        banks[lender_idx]["liquid_assets"] = (
            float(banks[lender_idx].get("liquid_assets", 0.0)) + pay
        )
    if _is_central(banks[lender_idx], lender_idx):
        # Full claim leaves the estate stock (recovered + written off).
        _adjust_estate_claims(banks[lender_idx], -claim)
    return pay, unpaid


def settle_absorbing_defaults_batch(
    *,
    banks: list,
    book: Any,
    exposure_matrix,
    failed_indices: Iterable[int],
    step: int,
    reason: str,
    lgd: float,
    make_contract: Callable[..., Any],
    effective_notional: Callable[[Any], float],
    remove_contract: Callable[[Any], None] | None = None,
) -> list[dict]:
    """
    Two-phase batch resolution on one shared contract snapshot.

    1) Freeze all failed banks (tombstone flags) so they are not payable counterparties.
    2) From the pre-removal contract list, monetize failed-as-lender claims into
       each failed bank's cash estate (CB pays recovery price).
    3) Pro-rata distribute each failed bank's cash to payable creditors (incl. CB).
    4) Close original contracts, open CB replacement claims, zero network rows.
    """
    lgd = float(max(0.0, min(1.0, lgd)))
    failed = sorted(
        {
            int(i)
            for i in failed_indices
            if 0 < int(i) < len(banks)
            and not banks[int(i)].get("absorbing_default")
            and not banks[int(i)].get("balance_sheet_frozen")
        }
    )
    if not failed:
        return []

    failed_set = set(failed)
    # Phase A: freeze failure set before any cash moves across the cohort.
    for i in failed:
        banks[i]["is_active"] = False
        banks[i]["absorbing_default"] = True  # provisional; finalized below
        banks[i]["balance_sheet_frozen"] = False  # still allow estate cash edits

    contracts = list(getattr(book, "contracts", [])) if book is not None else []
    # Snapshot edges involving any failed bank.
    borrower_claims: dict[int, list[tuple[int, float, Any]]] = {i: [] for i in failed}
    lender_claims: dict[int, list[Any]] = {i: [] for i in failed}
    to_remove: list = []
    remove_ids: set[int] = set()

    for c in contracts:
        li = int(getattr(c, "lender_idx", -1))
        bj = int(getattr(c, "borrower_idx", -1))
        touches = (li in failed_set) or (bj in failed_set)
        if not touches:
            continue
        to_remove.append(c)
        remove_ids.add(id(c))
        claim = float(effective_notional(c))
        if claim <= 1e-12:
            continue
        # Failed as borrower: include CB (li==0) and commercial lenders.
        if bj in failed_set and 0 <= li < len(banks) and li != bj:
            borrower_claims[bj].append((li, claim, c))
        # Failed as lender: will transfer to CB (skip if borrower also failed → B2).
        if li in failed_set and 0 <= bj < len(banks) and bj != li:
            lender_claims[li].append(c)

    cb = banks[0] if banks else None
    summaries: dict[int, dict] = {
        i: {
            "recovery": 0.0,
            # Primary: debtor default → creditor unrecovered amount.
            "creditor_writeoff": 0.0,
            # Auxiliary: failed-lender book value vs CB transfer price.
            "estate_transfer_discount": 0.0,
            # Backward-compatible alias of creditor_writeoff (set before return).
            "writeoff": 0.0,
            "estate_claims_transferred": 0.0,
            "transfer_consideration": 0.0,
            "contracts_removed": 0,
            "affected_counterparties": set(),
            "lgd": float(lgd),
            "reason": str(reason),
            "step": int(step),
            "bank_idx": int(i),
        }
        for i in failed
    }

    recorded_creditor_losses: dict[tuple, float] = {}

    def _claim_key(c) -> tuple:
        cid = getattr(c, "contract_id", None)
        if cid is not None:
            return ("contract_id", str(cid))
        return ("object_id", int(id(c)))

    def _record_creditor_writeoff(borrower_idx: int, contract, amount: float) -> float:
        """同一合同在一个 resolution batch 内最多记录一次。"""
        loss = max(0.0, float(amount))
        if loss <= 1e-12:
            return 0.0
        key = _claim_key(contract)
        previous = float(recorded_creditor_losses.get(key, 0.0))
        increment = max(0.0, loss - previous)
        if increment <= 1e-12:
            return 0.0
        recorded_creditor_losses[key] = loss
        summaries[int(borrower_idx)]["creditor_writeoff"] += increment
        return increment

    # Phase B1: monetize failed-lender claims → consideration into estate cash.
    new_contracts: list = []
    for i in failed:
        bank = banks[i]
        for c in lender_claims[i]:
            claim = float(effective_notional(c))
            bj = int(c.borrower_idx)
            summaries[i]["affected_counterparties"].add(bj)
            if claim <= 1e-12 or cb is None:
                continue
            if bj in failed_set:
                # Recorded once under the failed borrower in B2.
                continue
            # Transfer price = recovery value of the claim; paid by CB into the estate.
            price_cap = claim * (1.0 - lgd)
            pay = min(max(0.0, float(cb.get("liquid_assets", 0.0))), price_cap)
            if pay > 1e-15:
                cb["liquid_assets"] = float(cb.get("liquid_assets", 0.0)) - pay
                bank["liquid_assets"] = float(bank.get("liquid_assets", 0.0)) + pay
            summaries[i]["transfer_consideration"] += pay
            # Book value − disposal proceeds (not debtor-default loss).
            summaries[i]["estate_transfer_discount"] += max(0.0, claim - pay)

            new_c = make_contract(
                lender_idx=0,
                borrower_idx=bj,
                # Fold full claim (principal + arrears) into replacement notional.
                principal=float(claim),
                rate=0.0,
                created_step=int(step),
                maturity_step=int(getattr(c, "maturity_step", step + 1)),
                schedule_type=str(getattr(c, "schedule_type", "single_payment")),
                remaining_principal=float(claim),
                coupon_rate=0.0,
                tenor_total=int(getattr(c, "tenor_total", 1) or 1),
                periods_paid=int(getattr(c, "periods_paid", 0) or 0),
                settlement_rate=0.0,
                is_rollover_residual=bool(getattr(c, "is_rollover_residual", False)),
                consecutive_misses=int(getattr(c, "consecutive_misses", 0) or 0),
                arrears_due=0.0,
            )
            new_contracts.append(new_c)
            summaries[i]["estate_claims_transferred"] += claim
            _adjust_estate_claims(cb, claim)

    # Phase B2: pro-rata pay creditors of each failed borrower from estate cash.
    for i in failed:
        bank = banks[i]
        claims = borrower_claims[i]
        if not claims:
            continue
        payable: list[tuple[int, float, Any]] = []
        skipped: list[tuple[int, float, Any]] = []
        for li, claim, c in claims:
            summaries[i]["affected_counterparties"].add(li)
            if li in failed_set:
                skipped.append((li, claim, c))
                continue
            if _is_payable_creditor(banks[li], li):
                payable.append((li, claim, c))
            else:
                skipped.append((li, claim, c))

        total_payable = float(sum(claim for _, claim, _ in payable))
        estate_cash = max(0.0, float(bank.get("liquid_assets", 0.0)))
        recovery_cap = total_payable * (1.0 - lgd)
        pool = min(estate_cash, recovery_cap) if total_payable > 1e-12 else 0.0
        bank["liquid_assets"] = estate_cash - pool

        if total_payable > 1e-12 and pool > 1e-12:
            for li, claim, c in payable:
                share = pool * claim / total_payable
                recovery, _ = _pay_creditor(banks, li, share, claim=claim)
                summaries[i]["recovery"] += recovery
                _record_creditor_writeoff(i, c, claim - recovery)
        else:
            for li, claim, c in payable:
                _pay_creditor(banks, li, 0.0, claim=claim)
                _record_creditor_writeoff(i, c, claim)

        for li, claim, c in skipped:
            _record_creditor_writeoff(i, c, claim)

    # Close originals, add CB replacement contracts.
    remover = remove_contract
    if remover is None and book is not None:
        remover = getattr(book, "remove_contract", None)
    if remover is not None:
        for c in to_remove:
            try:
                remover(c)
            except Exception:
                pass
    elif book is not None:
        book.contracts = [c for c in list(getattr(book, "contracts", [])) if id(c) not in remove_ids]

    if book is not None:
        for new_c in new_contracts:
            if hasattr(book, "add_contract"):
                book.add_contract(new_c)
            else:
                book.contracts.append(new_c)

    out: list[dict] = []
    for i in failed:
        bank = banks[i]
        if exposure_matrix is not None:
            exposure_matrix[i, :] = 0.0
            exposure_matrix[:, i] = 0.0
        bank["interbank_assets"] = 0.0
        bank["interbank_liabilities"] = 0.0
        summ = summaries[i]
        summ["writeoff"] = float(summ["creditor_writeoff"])
        summ["contracts_removed"] = int(
            sum(
                1
                for c in to_remove
                if int(getattr(c, "lender_idx", -1)) == i
                or int(getattr(c, "borrower_idx", -1)) == i
            )
        )
        summ["affected_counterparties"] = sorted(
            int(x) for x in summ["affected_counterparties"] if int(x) != i
        )
        out.append(summ)
    return out


def settle_absorbing_default(
    *,
    banks: list,
    book: Any,
    exposure_matrix,
    bank_idx: int,
    step: int,
    reason: str,
    lgd: float,
    make_contract: Callable[..., Any],
    effective_notional: Callable[[Any], float],
    remove_contract: Callable[[Any], None] | None = None,
) -> dict:
    """Single-bank wrapper around the batch clearer."""
    rows = settle_absorbing_defaults_batch(
        banks=banks,
        book=book,
        exposure_matrix=exposure_matrix,
        failed_indices=[int(bank_idx)],
        step=step,
        reason=reason,
        lgd=lgd,
        make_contract=make_contract,
        effective_notional=effective_notional,
        remove_contract=remove_contract,
    )
    if not rows:
        return {
            "recovery": 0.0,
            "creditor_writeoff": 0.0,
            "estate_transfer_discount": 0.0,
            "writeoff": 0.0,
            "estate_claims_transferred": 0.0,
            "transfer_consideration": 0.0,
            "contracts_removed": 0,
            "affected_counterparties": [],
            "lgd": float(lgd),
            "reason": str(reason),
            "step": int(step),
            "bank_idx": int(bank_idx),
        }
    return rows[0]
