import pytest

from alpha.risk import (
    COST_TO_EDGE_HAIRCUT_THRESHOLD,
    LAMBDA_SOURCE_VALIDATED,
    THESIS_RIGHT_TAIL_CONVEX,
    adjusted_risk_budget_pct,
    can_add_unstopped_position,
    candidate_base_risk_budget_pct,
    cost_gate_result,
    cost_to_edge_ratio,
    final_position_cap,
    m1_catastrophe_loss_threshold,
    m1_idiosyncratic_catastrophe_triggered,
    risk_sized_cap,
    unstopped_heat_pct,
)


def test_risk_multipliers_compound_before_risk_cap():
    adjusted = adjusted_risk_budget_pct(
        base_risk_budget=0.01,
        hazard_multiplier=1.0,
        liquidity_multiplier=1.0,
        fidelity_multiplier=0.70,
        concentration_multiplier=1.0,
        crisis_multiplier=0.70,
    )

    assert adjusted == pytest.approx(0.0049)
    assert adjusted != pytest.approx(0.0070)
    assert risk_sized_cap(
        adjusted_risk_budget=adjusted,
        effective_hard_stop_pct=0.05,
    ) == pytest.approx(0.098)


def test_final_position_cap_applies_true_ceilings_after_compounded_haircuts():
    adjusted = adjusted_risk_budget_pct(
        base_risk_budget=0.01,
        hazard_multiplier=0.70,
        liquidity_multiplier=1.0,
        fidelity_multiplier=0.70,
        concentration_multiplier=1.0,
        crisis_multiplier=1.0,
    )
    risk_cap = risk_sized_cap(
        adjusted_risk_budget=adjusted,
        effective_hard_stop_pct=0.005,
    )

    assert risk_cap == pytest.approx(0.98)
    assert final_position_cap(
        aum_tier_cap=0.40,
        tcb_max_position_pct=0.40 * 0.70 * 1.0 * 0.70,
        concentration_multiplier=1.0,
        crisis_multiplier=1.0,
        risk_cap=risk_cap,
        adv_liquidity_cap=0.20,
        deployable_cash_cap=0.35,
    ) == pytest.approx(0.196)


def test_cost_gate_uses_shrunk_validation_adjusted_edge_not_raw_lambda():
    raw_ratio = 0.011 / 0.04
    shrunk_ratio = cost_to_edge_ratio(
        expected_round_trip_cost=0.011,
        raw_expected_edge=0.04,
        pattern_weight=1.0,
        shrinkage=0.70,
        validation_weight_multiplier=0.75,
    )

    assert cost_gate_result(raw_ratio).decision == "haircut"
    assert shrunk_ratio > 0.30
    assert cost_gate_result(shrunk_ratio).decision == "reject"


@pytest.mark.parametrize(
    "override,expected",
    [
        ({}, 0.0125),
        ({"thesis_category": "continuation"}, 0.0100),
        ({"candidate_rank": 2}, 0.0100),
        ({"cost_ratio": COST_TO_EDGE_HAIRCUT_THRESHOLD + 0.001}, 0.0100),
        ({"lambda_source": "shadow_prior"}, 0.0100),
    ],
)
def test_elevated_risk_budget_requires_all_four_conditions(override, expected):
    args = {
        "nav": 1_000,
        "thesis_category": THESIS_RIGHT_TAIL_CONVEX,
        "candidate_rank": 1,
        "cost_ratio": 0.10,
        "lambda_source": LAMBDA_SOURCE_VALIDATED,
    }
    args.update(override)

    assert candidate_base_risk_budget_pct(**args) == pytest.approx(expected)


def test_unstopped_heat_is_conservative_vs_stopped_position_heat():
    position_weight = 0.25
    sigma_20d = 0.04
    stopped_heat_same_size = position_weight * 0.05

    assert unstopped_heat_pct(position_weight, sigma_20d) == pytest.approx(0.02)
    assert unstopped_heat_pct(position_weight, sigma_20d) >= stopped_heat_same_size


def test_m1_unstopped_heat_consumes_capacity_and_can_block_new_entry():
    assert can_add_unstopped_position(
        current_unstopped_heat=0.0,
        position_weight=0.25,
        sigma_20d=0.04,
        nav=1_000,
    )
    assert not can_add_unstopped_position(
        current_unstopped_heat=0.019,
        position_weight=0.25,
        sigma_20d=0.04,
        nav=1_000,
    )


def test_m1_catastrophe_backstop_is_sector_relative():
    assert m1_catastrophe_loss_threshold(0.04) == pytest.approx(0.15)
    assert m1_idiosyncratic_catastrophe_triggered(
        stock_return_since_entry=-0.18,
        sector_return_since_entry=-0.02,
        sigma_20d=0.04,
    )
    assert not m1_idiosyncratic_catastrophe_triggered(
        stock_return_since_entry=-0.18,
        sector_return_since_entry=-0.12,
        sigma_20d=0.04,
    )
    assert not m1_idiosyncratic_catastrophe_triggered(
        stock_return_since_entry=-0.14,
        sector_return_since_entry=0.0,
        sigma_20d=0.04,
    )


def test_m1_catastrophe_backstop_scales_with_high_volatility():
    assert m1_catastrophe_loss_threshold(0.08) == pytest.approx(0.24)
    assert not m1_idiosyncratic_catastrophe_triggered(
        stock_return_since_entry=-0.18,
        sector_return_since_entry=0.0,
        sigma_20d=0.08,
    )
    assert m1_idiosyncratic_catastrophe_triggered(
        stock_return_since_entry=-0.25,
        sector_return_since_entry=0.0,
        sigma_20d=0.08,
    )
