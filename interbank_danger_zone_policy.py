"""
Interbank danger-zone / forbearance policy.

When a bank approaches a tipping point, interbank payments and new origination
are suspended while contracts remain on the ContractBook.  On exit, suspended
borrower notes are restructured rather than wiped or fully repaid at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from typing import Protocol

    class _Contract(Protocol):
        contract_id: str
        lender_idx: int
        borrower_idx: int
        principal: float
        rate: float
        created_step: int
        maturity_step: int
        status: str
        suspended_at_step: int | None
        deferred_count: int

        def cashflow_at_maturity(self) -> tuple[float, float]: ...


@dataclass
class DangerZoneConfig:
    car_tipping: float = 0.04
    car_resume: float = 0.06
    lcr_tipping: float = 0.85
    lcr_resume: float = 1.00
    equity_ratio_tipping: float = 0.02
    equity_ratio_resume: float = 0.04
    suspend_min_steps: int = 3
    resume_window: int = 5
    accrue_during_suspension: bool = False
    exit_partial_pay_ratio: float = 0.10
    restructure_maturity: int = 6
    restructure_penalty_spread: float = 0.01
    hard_default_car: float = 0.0
    hard_default_after_steps: int = 30
    defer_maturity_bump: int = 1


@dataclass
class BankZoneRecord:
    zone: str = "normal"
    suspended_at_step: int | None = None
    suspend_steps: int = 0
    stable_count: int = 0
    restructure_count: int = 0
    last_event: str = "init"


class DangerZoneManager:
    """Per-bank danger zone state and contract forbearance helpers."""

    def __init__(self, config: DangerZoneConfig | None = None):
        self.config = config or DangerZoneConfig()
        self.records: dict[int, BankZoneRecord] = {}
        self.event_log: list[dict[str, Any]] = []

    def reset(self, num_banks: int) -> None:
        self.records = {i: BankZoneRecord() for i in range(num_banks)}
        self.event_log = []

    def _record(self, bank_idx: int) -> BankZoneRecord:
        if bank_idx not in self.records:
            self.records[bank_idx] = BankZoneRecord()
        return self.records[bank_idx]

    def is_suspended(self, bank_idx: int) -> bool:
        return self._record(bank_idx).zone == "suspended"

    def is_flow_blocked(self, bank_idx: int) -> bool:
        rec = self._record(bank_idx)
        return rec.zone in {"warning", "suspended"}

    def _equity_ratio(self, bank: dict) -> float:
        core = float(bank.get("core_capital", 0.0))
        lia = float(bank.get("current_liabilities", 0.0))
        return core / (lia + 1e-9)

    def _metrics(self, bank: dict) -> tuple[float, float, float]:
        car = float(bank.get("capital_adequacy_ratio", 0.0))
        lcr = float(bank.get("liquidity_coverage_ratio", 0.0))
        eq = self._equity_ratio(bank)
        return car, lcr, eq

    def _tipping_hit(self, bank: dict) -> bool:
        car, lcr, eq = self._metrics(bank)
        cfg = self.config
        return (
            car <= cfg.car_tipping
            or lcr <= cfg.lcr_tipping
            or eq <= cfg.equity_ratio_tipping
        )

    def _resume_ready(self, bank: dict, rec: BankZoneRecord) -> bool:
        if rec.suspend_steps < self.config.suspend_min_steps:
            return False
        car, lcr, eq = self._metrics(bank)
        cfg = self.config
        ok = (
            car >= cfg.car_resume
            and lcr >= cfg.lcr_resume
            and eq >= cfg.equity_ratio_resume
        )
        if ok:
            rec.stable_count += 1
        else:
            rec.stable_count = 0
        return rec.stable_count >= cfg.resume_window

    def _log(self, step: int, bank_idx: int, event: str, **extra: Any) -> None:
        row = {"step": int(step), "bank_idx": int(bank_idx), "event": event, **extra}
        self.event_log.append(row)
        rec = self._record(bank_idx)
        rec.last_event = event

    def update_bank_states(self, sim, step: int) -> None:
        """Evaluate enter / exit / hard-default transitions for all banks."""
        banks = sim.banks
        cfg = self.config
        for i, bank in enumerate(banks):
            if i == 0 or not bank.get("is_active", True):
                continue
            rec = self._record(i)
            car, lcr, eq = self._metrics(bank)

            if rec.zone == "suspended":
                rec.suspend_steps += 1
                if (
                    rec.suspend_steps >= cfg.hard_default_after_steps
                    and car <= cfg.hard_default_car
                ):
                    bank["is_active"] = False
                    bank["liquid_assets"] = float(bank.get("liquid_assets", 0.0)) * 0.8
                    rec.zone = "defaulted"
                    self._log(step, i, "hard_default", car=car, lcr=lcr, eq=eq)
                    continue

                if self._resume_ready(bank, rec):
                    self.exit_and_restructure(sim, i, step)
                continue

            if self._tipping_hit(bank):
                if rec.zone != "suspended":
                    rec.zone = "suspended"
                    rec.suspended_at_step = int(step)
                    rec.suspend_steps = 0
                    rec.stable_count = 0
                    self.suspend_contracts_for_bank(sim.contract_book, i, step)
                    bank["danger_zone"] = "suspended"
                    self._log(step, i, "enter_suspended", car=car, lcr=lcr, eq=eq)
            elif rec.zone == "normal" and (
                car <= cfg.car_resume or lcr <= cfg.lcr_resume
            ):
                rec.zone = "warning"
                bank["danger_zone"] = "warning"
                self._log(step, i, "enter_warning", car=car, lcr=lcr, eq=eq)
            elif rec.zone == "warning":
                if not self._tipping_hit(bank) and car >= cfg.car_resume and lcr >= cfg.lcr_resume:
                    rec.zone = "normal"
                    bank["danger_zone"] = "normal"
                    self._log(step, i, "exit_warning", car=car, lcr=lcr, eq=eq)
                elif self._tipping_hit(bank):
                    rec.zone = "suspended"
                    rec.suspended_at_step = int(step)
                    rec.suspend_steps = 0
                    rec.stable_count = 0
                    self.suspend_contracts_for_bank(sim.contract_book, i, step)
                    bank["danger_zone"] = "suspended"
                    self._log(step, i, "warning_to_suspended", car=car, lcr=lcr, eq=eq)
            else:
                bank["danger_zone"] = rec.zone

    def suspend_contracts_for_book(self, book, bank_idx: int, step: int) -> None:
        self.suspend_contracts_for_bank(book, bank_idx, step)

    def suspend_contracts_for_bank(self, book, bank_idx: int, step: int) -> None:
        for c in book.contracts:
            if c.lender_idx == bank_idx or c.borrower_idx == bank_idx:
                c.status = "suspended"
                c.suspended_at_step = int(step)

    def split_due_contracts(self, due: list, sim, step: int) -> tuple[list, list]:
        """Return (settle_now, defer) depending on suspension state."""
        settle_now: list = []
        defer: list = []
        for c in due:
            if self.is_flow_blocked(c.lender_idx) or self.is_flow_blocked(c.borrower_idx):
                defer.append(c)
            elif getattr(c, "status", "active") == "suspended":
                defer.append(c)
            else:
                settle_now.append(c)
        return settle_now, defer

    def defer_contracts(self, contracts: list, step: int) -> None:
        bump = int(self.config.defer_maturity_bump)
        for c in contracts:
            c.maturity_step = int(step) + bump
            c.deferred_count = int(getattr(c, "deferred_count", 0)) + 1
            c.status = "suspended"

    def accrue_suspended_interest(self, book, step: int) -> None:
        if not self.config.accrue_during_suspension:
            return
        for c in book.contracts:
            if getattr(c, "status", "active") != "suspended":
                continue
            interest, _ = c.cashflow_at_maturity()
            c.principal = float(c.principal) + float(interest)
            c.maturity_step = int(step) + int(self.config.defer_maturity_bump)

    def exit_and_restructure(self, sim, bank_idx: int, step: int) -> None:
        """Restructure all suspended borrower notes for bank j on exit."""
        book = sim.contract_book
        banks = sim.banks
        bank = banks[bank_idx]
        rec = self._record(bank_idx)

        borrower_notes = [
            c for c in book.contracts
            if c.borrower_idx == bank_idx and getattr(c, "status", "active") == "suspended"
        ]
        if not borrower_notes:
            rec.zone = "normal"
            rec.suspended_at_step = None
            rec.suspend_steps = 0
            rec.stable_count = 0
            bank["danger_zone"] = "normal"
            self._log(step, bank_idx, "exit_suspended_no_notes")
            return

        total_debt = 0.0
        lender_weights: dict[int, float] = {}
        for c in borrower_notes:
            interest, principal = c.cashflow_at_maturity()
            claim = float(principal) if not self.config.accrue_during_suspension else float(principal + interest)
            total_debt += claim
            lender_weights[c.lender_idx] = lender_weights.get(c.lender_idx, 0.0) + claim

        partial = 0.0
        ratio = float(self.config.exit_partial_pay_ratio)
        if ratio > 0.0 and total_debt > 0.0:
            lia = float(bank.get("current_liabilities", 0.0))
            reserve_need = float(sim.reserve_buffer[bank_idx]) * lia
            avail = max(0.0, float(bank.get("liquid_assets", 0.0)) - reserve_need)
            partial = min(avail, ratio * total_debt)
            if partial > 1e-8:
                bank["liquid_assets"] = float(bank.get("liquid_assets", 0.0)) - partial
                remaining = partial
                for lender_idx, weight in lender_weights.items():
                    if remaining <= 1e-8:
                        break
                    share = partial * (weight / total_debt)
                    share = min(share, remaining)
                    banks[lender_idx]["liquid_assets"] = float(
                        banks[lender_idx].get("liquid_assets", 0.0)
                    ) + share
                    remaining -= share

        remaining_debt = max(0.0, total_debt - partial)
        for c in list(borrower_notes):
            book.remove_contract(c)

        if remaining_debt > 1e-8:
            primary_lender = max(lender_weights, key=lender_weights.get)
            new_rate = float(sim.base_rate) + float(self.config.restructure_penalty_spread)
            new_contract = type(borrower_notes[0])(
                contract_id=book._new_id(),
                lender_idx=int(primary_lender),
                borrower_idx=int(bank_idx),
                principal=float(remaining_debt),
                rate=new_rate,
                created_step=int(step),
                maturity_step=int(step) + int(self.config.restructure_maturity),
                status="active",
                suspended_at_step=None,
                deferred_count=0,
            )
            book.add_contract(new_contract)

        for c in book.contracts:
            if c.lender_idx == bank_idx and getattr(c, "status", "active") == "suspended":
                c.status = "active"
                c.suspended_at_step = None

        rec.zone = "normal"
        rec.suspended_at_step = None
        rec.suspend_steps = 0
        rec.stable_count = 0
        rec.restructure_count += 1
        bank["danger_zone"] = "normal"
        self._log(
            step,
            bank_idx,
            "exit_restructured",
            total_debt=total_debt,
            partial_paid=partial,
            remaining_debt=remaining_debt,
            note_count=len(borrower_notes),
        )

    def export_summary_rows(self) -> list[dict[str, Any]]:
        rows = []
        for bank_idx, rec in sorted(self.records.items()):
            rows.append({
                "bank_idx": bank_idx,
                "zone": rec.zone,
                "suspended_at_step": rec.suspended_at_step,
                "suspend_steps": rec.suspend_steps,
                "stable_count": rec.stable_count,
                "restructure_count": rec.restructure_count,
                "last_event": rec.last_event,
            })
        return rows


def contract_counts_in_exposure(c, current_step: int) -> bool:
    """Include active future contracts and any suspended/forborne note."""
    status = getattr(c, "status", "active")
    if status == "suspended":
        return True
    return int(c.maturity_step) > int(current_step)
