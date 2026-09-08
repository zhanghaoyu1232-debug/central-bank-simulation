# -*- coding: utf-8 -*-
"""Key-based project RNG shared by DEN and CEN.

Draws are determined by (seed, step, bank_id, project_id, draw_kind), not by
call order. Extra DEN projects therefore cannot shift CEN project shocks.
"""
from __future__ import annotations

import numpy as np

PROJECT_RNG_SCHEMA_VERSION = 1

PROJECT_ORIGIN_INITIAL = 0
PROJECT_ORIGIN_FREE_CASH = 1
PROJECT_ORIGIN_BORROWED = 2

DRAW_CREATION = 11
DRAW_PERIOD = 21


def make_project_id(
    creation_step: int,
    origin: int,
    slot: int,
) -> int:
    """
    Stable project ID: creation_step + origin + slot within that creation.

    Avoids a global counter so an extra DEN project does not renumber later IDs.
    """
    step_part = (int(creation_step) + 1) & 0xFFFFFFFF
    origin_part = int(origin) & 0xFF
    slot_part = int(slot) & 0xFFFFFF
    return (step_part << 32) | (origin_part << 24) | slot_part


def _project_rng(
    *,
    seed: int,
    step: int,
    bank_id: int,
    project_id: int,
    draw_kind: int,
):
    pid = int(project_id)
    entropy = [
        int(seed) & 0xFFFFFFFF,
        int(step) & 0xFFFFFFFF,
        int(bank_id) & 0xFFFFFFFF,
        pid & 0xFFFFFFFF,
        (pid >> 32) & 0xFFFFFFFF,
        int(draw_kind) & 0xFFFFFFFF,
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def project_creation_draws(
    *,
    seed: int,
    creation_step: int,
    bank_id: int,
    project_id: int,
    maturity_range: tuple[int, int],
    pd_range: tuple[float, float],
    lgd_range: tuple[float, float],
) -> tuple[int, float, float]:
    rng = _project_rng(
        seed=seed,
        step=creation_step,
        bank_id=bank_id,
        project_id=project_id,
        draw_kind=DRAW_CREATION,
    )
    maturity = int(rng.integers(*maturity_range))
    pd = float(rng.uniform(*pd_range))
    lgd = float(rng.uniform(*lgd_range))
    return maturity, pd, lgd


def project_period_draws(
    *,
    seed: int,
    step: int,
    bank_id: int,
    project_id: int,
    shock_mean: float,
    shock_std: float,
) -> tuple[float, float]:
    rng = _project_rng(
        seed=seed,
        step=step,
        bank_id=bank_id,
        project_id=project_id,
        draw_kind=DRAW_PERIOD,
    )
    # Always draw both streams (default and shock), independent of default outcome.
    pd_roll = float(rng.random())
    shock = float(rng.normal(shock_mean, shock_std))
    return pd_roll, shock
