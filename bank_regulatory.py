# -*- coding: utf-8 -*-
"""Unified regulatory RWA / CAR for simulation, policy, roles, and SR.

Single definition used everywhere:

    RWA = max(0.5 · interbank_assets + 1.0 · projects, κ · total_liabilities)

with κ = 0.08 and total_liabilities = current + interbank + termed_out.
CAR = clip(core_capital / RWA, 0, MAX_CAR_RATIO).
"""
from __future__ import annotations

from typing import Any, Mapping

MIN_CAR_RWA_LIA_FRAC = 0.08
MAX_CAR_RATIO = 1.50


def project_amount(bank: Mapping[str, Any] | None) -> float:
    if not bank:
        return 0.0
    inv = bank.get("investment") or {}
    proj = inv.get("projects") if isinstance(inv, Mapping) else None
    if isinstance(proj, Mapping):
        return max(0.0, float(proj.get("amount", 0.0) or 0.0))
    return 0.0


def total_liabilities(bank: Mapping[str, Any] | None) -> float:
    """Accounting liabilities for equity / RWA.

    Includes termed-out (long) liabilities. LCR short-term outflow uses a
    separate denominator (current + interbank only) and must not call this.
    """
    if not bank:
        return 0.0
    return max(
        0.0,
        float(bank.get("current_liabilities", 0.0) or 0.0)
        + float(bank.get("interbank_liabilities", 0.0) or 0.0)
        + float(bank.get("termed_out_liabilities", 0.0) or 0.0),
    )


def regulatory_rwa(
    bank: Mapping[str, Any] | None = None,
    *,
    interbank_assets: float | None = None,
    projects_amt: float | None = None,
    liabilities: float | None = None,
    min_lia_frac: float = MIN_CAR_RWA_LIA_FRAC,
) -> float:
    """Paper-aligned regulatory RWA (with liability floor).

    Pass a bank dict, or explicit component floats (for init / cache refresh).
    """
    if bank is not None:
        if interbank_assets is None:
            interbank_assets = float(bank.get("interbank_assets", 0.0) or 0.0)
        if projects_amt is None:
            projects_amt = project_amount(bank)
        if liabilities is None:
            liabilities = total_liabilities(bank)
    ib = max(0.0, float(interbank_assets or 0.0))
    pa = max(0.0, float(projects_amt or 0.0))
    lia = max(0.0, float(liabilities or 0.0))
    rwa = 0.5 * ib + 1.0 * pa
    return float(max(rwa, float(min_lia_frac) * lia))


def regulatory_car(
    bank: Mapping[str, Any] | None = None,
    *,
    core_capital: float | None = None,
    interbank_assets: float | None = None,
    projects_amt: float | None = None,
    liabilities: float | None = None,
    max_ratio: float = MAX_CAR_RATIO,
    min_lia_frac: float = MIN_CAR_RWA_LIA_FRAC,
) -> float:
    """CAR = core / regulatory_rwa(...), clipped to [0, max_ratio]."""
    if bank is not None and core_capital is None:
        core_capital = float(bank.get("core_capital", 0.0) or 0.0)
    core = max(0.0, float(core_capital or 0.0))
    rwa = regulatory_rwa(
        bank,
        interbank_assets=interbank_assets,
        projects_amt=projects_amt,
        liabilities=liabilities,
        min_lia_frac=min_lia_frac,
    )
    if rwa <= 0.0:
        return 0.0
    return float(max(0.0, min(float(max_ratio), core / (rwa + 1e-9))))
