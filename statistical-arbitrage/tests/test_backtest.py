import pandas as pd
import pytest

from src.backtest import generate_positions, backtest_spread, performance_summary


def test_position_enters_long_below_negative_threshold():
    z = pd.Series([0.0, -2.5, -2.5, -0.1, 0.0])
    positions = generate_positions(z, entry=2.0, exit=0.5)
    assert list(positions) == [0, 1, 1, 0, 0]


def test_position_enters_short_above_positive_threshold():
    z = pd.Series([0.0, 2.5, 2.5, 0.1, 0.0])
    positions = generate_positions(z, entry=2.0, exit=0.5)
    assert list(positions) == [0, -1, -1, 0, 0]


def test_backtest_matches_hand_computed_pnl_with_zero_cost():
    spread_series = pd.Series([100.0, 101.0, 103.0, 102.0])
    positions = pd.Series([0, 1, 1, 0])
    pnl, _equity = backtest_spread(spread_series, positions, cost_bps=0.0)
    assert list(pnl.fillna(0)) == pytest.approx([0.0, 0.0, 2.0, -1.0])


def test_performance_summary_reports_correct_total_and_drawdown():
    pnl = pd.Series([1.0, 1.0, -3.0, 2.0])
    summary = performance_summary(pnl)
    assert summary["total_pnl"] == pytest.approx(1.0)
    assert summary["max_drawdown"] == pytest.approx(-3.0)
