from __future__ import annotations

import numpy as np
import pytest

from volsurface.black_scholes import call_price, put_price, vega, intrinsic_value

BASE = dict(spot=100.0, strike=100.0, maturity=1.0, rate=0.05)


def test_put_call_parity_holds_identically():
    """C - P = S*exp(-qT) - K*exp(-rT), independent of volatility."""
    for volatility in [0.1, 0.3, 0.8]:
        call = call_price(volatility=volatility, dividend_yield=0.02, **BASE)
        put = put_price(volatility=volatility, dividend_yield=0.02, **BASE)
        expected = BASE["spot"] * np.exp(-0.02 * BASE["maturity"]) - BASE["strike"] * np.exp(
            -BASE["rate"] * BASE["maturity"]
        )
        assert call - put == pytest.approx(expected, abs=1e-10)


def test_price_is_strictly_increasing_in_volatility():
    """This monotonicity is what guarantees implied volatility is unique."""
    prices = [call_price(volatility=v, **BASE) for v in np.linspace(0.05, 2.0, 40)]
    assert np.all(np.diff(prices) > 0)


def test_price_respects_no_arbitrage_bounds():
    for volatility in [0.1, 0.5, 1.5]:
        call = call_price(volatility=volatility, **BASE)
        lower = intrinsic_value("call", BASE["spot"], BASE["strike"], BASE["maturity"], BASE["rate"])
        assert lower <= call <= BASE["spot"]


def test_vega_matches_a_numerical_derivative_of_price():
    step = 1e-6
    analytic = vega(volatility=0.3, **BASE)
    numerical = (call_price(volatility=0.3 + step, **BASE) - call_price(volatility=0.3 - step, **BASE)) / (2 * step)
    assert analytic == pytest.approx(numerical, rel=1e-6)


def test_deep_in_the_money_call_approaches_its_forward_value():
    deep = call_price(spot=100.0, strike=1.0, maturity=1.0, rate=0.05, volatility=0.2)
    assert deep == pytest.approx(100.0 - 1.0 * np.exp(-0.05), abs=1e-6)


def test_vega_collapses_far_from_the_money():
    """The numerical reason implied volatility is unreliable for deep in or out
    of the money quotes: price stops responding to volatility."""
    at_the_money = vega(volatility=0.2, **BASE)
    far_out = vega(spot=100.0, strike=400.0, maturity=0.02, rate=0.05, volatility=0.2)
    assert far_out < at_the_money * 1e-6
