"""Candidate risk policy helpers.

These functions make the vault's sizing and cost-gate contracts executable
before the full TCB/KOTH implementation exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite


COST_TO_EDGE_HAIRCUT_THRESHOLD = 0.15
COST_TO_EDGE_HARD_REJECT = 0.30
HIGH_COST_EDGE_MULT = 0.75

LAMBDA_SOURCE_VALIDATED = "validated_or_injected"
THESIS_RIGHT_TAIL_CONVEX = "right_tail_convex"

ELEVATED_BASE_RISK_BUDGET = 0.0125
M1_CATASTROPHE_SIGMA_MULTIPLE = 3.0
M1_CATASTROPHE_FLOOR = 0.15


@dataclass(frozen=True)
class CostGateResult:
    """Decision payload from the cost-to-edge gate."""

    ratio: float
    decision: str
    edge_multiplier: float


def gross_optimizer_edge(
    raw_expected_edge: float,
    *,
    pattern_weight: float,
    shrinkage: float,
    validation_weight_multiplier: float,
) -> float:
    """Shrunk, validation-adjusted gross edge before cost subtraction."""
    return (
        raw_expected_edge
        * pattern_weight
        * shrinkage
        * validation_weight_multiplier
    )


def cost_to_edge_ratio(
    expected_round_trip_cost: float,
    raw_expected_edge: float,
    *,
    pattern_weight: float,
    shrinkage: float,
    validation_weight_multiplier: float,
    epsilon: float = 1e-12,
) -> float:
    """Return expected cost as a share of gross optimizer edge."""

    denominator = gross_optimizer_edge(
        raw_expected_edge,
        pattern_weight=pattern_weight,
        shrinkage=shrinkage,
        validation_weight_multiplier=validation_weight_multiplier,
    )
    if denominator <= 0 or not isfinite(denominator):
        return inf
    return expected_round_trip_cost / max(denominator, epsilon)


def cost_gate_result(cost_ratio: float) -> CostGateResult:
    """Classify a cost ratio as pass, haircut, or reject."""

    if cost_ratio > COST_TO_EDGE_HARD_REJECT:
        return CostGateResult(cost_ratio, "reject", 0.0)
    if cost_ratio > COST_TO_EDGE_HAIRCUT_THRESHOLD:
        return CostGateResult(cost_ratio, "haircut", HIGH_COST_EDGE_MULT)
    return CostGateResult(cost_ratio, "pass", 1.0)


def base_risk_budget_pct(nav: float) -> float:
    """Return the default per-position risk budget for an NAV tier."""

    if nav <= 10_000:
        return 0.01
    if nav <= 50_000:
        return 0.0075
    return 0.005


def candidate_base_risk_budget_pct(
    *,
    nav: float,
    thesis_category: str,
    candidate_rank: int,
    cost_ratio: float,
    lambda_source: str,
) -> float:
    """Return the candidate's base risk budget before multiplicative controls."""

    base = base_risk_budget_pct(nav)
    if (
        nav <= 10_000
        and thesis_category == THESIS_RIGHT_TAIL_CONVEX
        and candidate_rank == 1
        and cost_ratio <= COST_TO_EDGE_HAIRCUT_THRESHOLD
        and lambda_source == LAMBDA_SOURCE_VALIDATED
    ):
        return ELEVATED_BASE_RISK_BUDGET
    return base


def adjusted_risk_budget_pct(
    *,
    base_risk_budget: float,
    hazard_multiplier: float,
    liquidity_multiplier: float,
    fidelity_multiplier: float,
    concentration_multiplier: float,
    crisis_multiplier: float,
) -> float:
    """Apply hazard, liquidity, fidelity, concentration, and crisis multipliers."""

    return (
        base_risk_budget
        * hazard_multiplier
        * liquidity_multiplier
        * fidelity_multiplier
        * concentration_multiplier
        * crisis_multiplier
    )


def risk_sized_cap(
    *,
    adjusted_risk_budget: float,
    effective_hard_stop_pct: float,
) -> float:
    """Convert risk budget into a position cap using the effective stop distance."""

    if effective_hard_stop_pct <= 0:
        return 0.0
    return adjusted_risk_budget / effective_hard_stop_pct


def final_position_cap(
    *,
    aum_tier_cap: float,
    tcb_max_position_pct: float,
    concentration_multiplier: float,
    crisis_multiplier: float,
    risk_cap: float,
    adv_liquidity_cap: float,
    deployable_cash_cap: float,
) -> float:
    """Return the binding position cap across policy, risk, liquidity, and cash."""

    adjusted_tcb_cap = (
        tcb_max_position_pct * concentration_multiplier * crisis_multiplier
    )
    return min(
        aum_tier_cap,
        adjusted_tcb_cap,
        risk_cap,
        adv_liquidity_cap,
        deployable_cash_cap,
    )


def unstopped_heat_pct(position_weight: float, sigma_20d: float) -> float:
    """Estimate stress heat contributed by a position without a hard stop."""

    stress_loss = min(0.20, max(0.05, 2.0 * sigma_20d))
    return position_weight * stress_loss


def no_stop_heat_cap(nav: float) -> float:
    """Return maximum aggregate unstopped heat allowed for an NAV tier."""

    if nav <= 10_000:
        return 0.02
    if nav <= 50_000:
        return 0.015
    return 0.01


def can_add_unstopped_position(
    *,
    current_unstopped_heat: float,
    position_weight: float,
    sigma_20d: float,
    nav: float,
) -> bool:
    """Return whether another unstopped position fits the heat cap."""

    return (
        current_unstopped_heat + unstopped_heat_pct(position_weight, sigma_20d)
        <= no_stop_heat_cap(nav)
    )


def m1_catastrophe_loss_threshold(sigma_20d: float) -> float:
    """Positive abnormal-loss threshold for M1's far thesis-invalidation backstop."""
    return max(M1_CATASTROPHE_FLOOR, M1_CATASTROPHE_SIGMA_MULTIPLE * sigma_20d)


def m1_idiosyncratic_catastrophe_triggered(
    *,
    stock_return_since_entry: float,
    sector_return_since_entry: float,
    sigma_20d: float,
) -> bool:
    """Return True when stock-specific loss is too large to treat as PEAD noise.

    The trigger is sector-relative so broad market/sector drawdowns do not trip
    the backstop by beta alone.
    """
    idiosyncratic_return = stock_return_since_entry - sector_return_since_entry
    return idiosyncratic_return <= -m1_catastrophe_loss_threshold(sigma_20d)
