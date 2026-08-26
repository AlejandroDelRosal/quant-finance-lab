from __future__ import annotations

import numpy as np
import pytest

from volsurface.black_scholes import price, intrinsic_value
from volsurface.implied_vol import implied_volatility, NoSolution


@pytest.mark.parametrize("true_volatility", [0.05, 0.15, 0.35, 0.90, 2.0])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_round_trip_recovers_the_volatility_used_to_price(true_volatility, option_type):
    """Price with a known volatility, then invert: the solver must return it."""
    theoretical = price(option_type, 100.0, 105.0, 0.5, 0.04, true_volatility, 0.01)
    recovered = implied_volatility(theoretical, option_type, 100.0, 105.0, 0.5, 0.04, 0.01)
    assert recovered == pytest.approx(true_volatility, rel=1e-6)


@pytest.mark.parametrize("strike", [70.0, 80.0, 100.0, 130.0, 160.0])
def test_round_trip_holds_across_the_strikes_that_carry_information(strike):
    """Newton alone diverges at the wings; the bracketed fallback must still
    converge wherever the price actually depends on volatility."""
    theoretical = price("call", 100.0, strike, 0.25, 0.04, 0.3)
    recovered = implied_volatility(theoretical, "call", 100.0, strike, 0.25, 0.04)
    assert recovered == pytest.approx(0.3, rel=1e-5)


@pytest.mark.parametrize("strike", [40.0, 250.0])
def test_volatility_is_unrecoverable_once_time_value_vanishes(strike):
    """Not a solver defect but an information limit, and the reason the market
    pipeline screens these quotes out.

    At these strikes an option priced with 30 percent volatility is worth its
    intrinsic value to within 1e-9, so no inversion can recover the volatility
    that produced it: many different volatilities give the same price to
    within any realistic quote precision.
    """
    theoretical = price("call", 100.0, strike, 0.25, 0.04, 0.3)
    time_value = theoretical - intrinsic_value("call", 100.0, strike, 0.25, 0.04)
    assert time_value < 1e-8

    # Volatilities anywhere from 5 to 25 percent price identically to within
    # 1e-12, so the inverse problem has no unique answer over that whole range
    # regardless of the algorithm used.
    spread_of_prices = abs(
        price("call", 100.0, strike, 0.25, 0.04, 0.05) - price("call", 100.0, strike, 0.25, 0.04, 0.25)
    )
    assert spread_of_prices < 1e-12


def test_price_below_intrinsic_value_is_rejected():
    with pytest.raises(NoSolution):
        implied_volatility(0.5, "call", 100.0, 50.0, 1.0, 0.05)


def test_price_above_any_attainable_value_is_rejected():
    with pytest.raises(NoSolution):
        implied_volatility(99.9, "call", 100.0, 100.0, 1.0, 0.05)


def test_solver_agrees_with_an_independent_brentq_search():
    theoretical = price("call", 100.0, 110.0, 0.75, 0.03, 0.42)
    from scipy.optimize import brentq

    reference = brentq(lambda v: price("call", 100.0, 110.0, 0.75, 0.03, v) - theoretical, 1e-4, 5.0)
    assert implied_volatility(theoretical, "call", 100.0, 110.0, 0.75, 0.03) == pytest.approx(reference, rel=1e-6)
