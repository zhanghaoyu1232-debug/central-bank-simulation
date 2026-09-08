def apply_en_contract_updates(
    book, due_flows, paid_by_cid, scheduled_parts, step,
    clear_tol=1e-6,
):
    """Contract-update portion of settle_interbank_period.

    EN has already transferred each debtor's payment. due_flows holds
    (contract, lender, borrower, amount_due); paid_by_cid allocates each
    debtor's payment pro rata across its due contracts.
    """
    hard_defaults = set()
    for c, lender, borrower, due in due_flows:
        cid = str(c.contract_id)
        paid = float(paid_by_cid.get(cid, 0.0))
        unpaid = max(0.0, float(due) - paid)
        tol = max(float(clear_tol), 1e-6, 1e-9 * float(due))

        if c.schedule_type == "single_payment":
            if unpaid <= tol:
                book.remove_contract(c)
            else:
                hard_defaults.add(int(borrower))
                c.principal = c.remaining_principal = unpaid
                c.arrears_due = 0.0
                c.rate = c.settlement_rate = c.coupon_rate = 0.0
                c.maturity_step = int(step)
            continue

        arrears, interest, planned_principal = scheduled_parts[cid]
        remaining = max(0.0, float(c.remaining_principal))
        c.remaining_principal = max(0.0, remaining - planned_principal)
        c.principal = c.remaining_principal
        c.periods_paid += 1
        if unpaid <= tol:
            c.consecutive_misses = 0
            c.arrears_due = 0.0
        else:
            c.consecutive_misses += 1
            c.arrears_due = unpaid
            c.maturity_step = max(int(c.maturity_step), int(step) + 1)
        if c.remaining_principal <= tol and c.arrears_due <= tol:
            book.remove_contract(c)
    return sorted(hard_defaults)


def _classify_equity_defaults(self):
    """Classify from accounting equity, not truncated core capital."""
    return [
        i for i in range(self.num_banks)
        if self.bank_types[i] != "central"
        and self.banks[i].get("is_active", True)
        and self._bank_equity(self.banks[i]) < 0.0
    ]


def _cascade_negative_equity_defaults(self, step, *, reason="negative_equity"):
    """Shared DEN/CEN refresh-classify-resolve loop."""
    resolved = []
    for _ in range(max(1, int(self.num_banks))):
        # Always entered, including when EN produced no hard default.
        # This rebuilds IB stocks from the remaining ContractBook.
        self._refresh_regulatory_metrics(int(step))
        newly = self._classify_equity_defaults()
        if not newly:
            break
        self._resolve_bank_defaults_batch(newly, int(step), reason=reason)
        resolved.extend(int(i) for i in newly)
    return sorted(set(resolved))


def simulate_step(self, step):
    """Condensed DEN/CEN daily sequence; detailed market blocks elided."""
    self.current_step = int(step)
    self.last_interbank_writeoff = 0.0
    self.last_estate_transfer_discount = 0.0
    self._reset_policy_step_budget()
    self._disburse_pending_policy_support(step)
    self._run_central_bank_policy_cycle(step)
    self._settle_central_bank_loans(step)

    # One EN settlement, followed by contract updates shown above.
    self._refresh_regulatory_metrics(step)
    failed = self._settle_interbank_installment_period(step)
    self._resolve_bank_defaults_batch(failed, step, reason="en_settlement")
    contagion_en = self._cascade_negative_equity_defaults(
        step, reason="contagion_after_en"
    )
    self._refresh_metrics_after_settle(step)
    self._queue_policy_support_from_stress(step)

    ...  # Market rates, deposit flows, and common shocks.
    self._refresh_regulatory_metrics(step)
    ...  # Role assignment and daily CEN or local GNN-RFQ matching.
    ...  # Origination registers installment ON / next-day OFF contracts.

    for i in range(self.num_banks):
        self.allocate_borrowed_to_projects(i)
        self.update_project_book(i)
    self._post_project_phase_hook(step)

    self._refresh_regulatory_metrics(step)
    support = self._apply_solvency_support(step)
    if float(support.get("solvency_support_total", 0.0)) > 0.0:
        self._refresh_regulatory_metrics(step)
    self._post_regulatory_refresh_hook(step)
    equity_defaults = self._cascade_negative_equity_defaults(
        step, reason="negative_equity"
    )
    en_defaults = list(getattr(self, "last_defaulted_banks", []) or [])
    self.last_defaulted_banks = sorted(set(
        en_defaults + contagion_en + equity_defaults
    ))
    risk = self.calculate_systemic_risk()
    self._record_systemic_risk(risk)
    ...  # Record paths; stop at T or at most one active non-central bank.
    return risk
