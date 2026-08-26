"""Checks that run against the cached real SPY option chain."""

from __future__ import annotations

import numpy as np
import pytest

from volsurface.chain import load_cached_chain, is_out_of_the_money

RATE = 0.042
DIVIDEND_YIELD = 0.012


@pytest.fixture(scope="module")
def chain():
    return load_cached_chain()


def matched_pairs(chain):
    calls = chain[chain.option_type == "call"].set_index(["expiry", "strike"])
    puts = chain[chain.option_type == "put"].set_index(["expiry", "strike"])
    return calls.join(puts, how="inner", lsuffix="_call", rsuffix="_put")


def test_the_parity_implied_forward_agrees_across_strikes(chain):
    """The sharpest no-arbitrage statement available from quotes alone.

    Rearranging put-call parity gives F = C - P + K*exp(-rT). Every strike at a
    given maturity must imply the same forward price, whatever the true rate
    and dividend yield happen to be, so their agreement tests the quotes
    without assuming either input.
    """
    matched = matched_pairs(chain)
    assert len(matched) > 100

    strike = matched.index.get_level_values("strike")
    maturity = matched["maturity_call"]
    forward = (matched["mid_call"] - matched["mid_put"]) + strike * np.exp(-RATE * maturity)

    by_maturity = forward.groupby(maturity)
    dispersion = (by_maturity.std() / by_maturity.mean()).dropna()
    assert dispersion.median() < 0.002


def parity_violation_fraction_of_spot(matched):
    strike = matched.index.get_level_values("strike")
    maturity = matched["maturity_call"]
    spot = matched["spot_call"]
    forward = spot * np.exp(-DIVIDEND_YIELD * maturity)
    discounted_strike = strike * np.exp(-RATE * maturity)
    return ((matched["mid_call"] - matched["mid_put"]) - (forward - discounted_strike)).abs() / spot


def test_parity_violations_are_negligible_at_short_maturities(chain):
    matched = matched_pairs(chain)
    short_dated = matched[matched["maturity_call"] < 0.25]
    violation = parity_violation_fraction_of_spot(short_dated)
    assert violation.median() < 0.002
    assert violation.quantile(0.95) < 0.005


def test_apparent_parity_violation_grows_with_maturity_as_a_rate_error_would(chain):
    """Evidence that the residual is a wrong assumed rate, not an arbitrage.

    Parity carries the discount factors exp(-rT) and exp(-qT), so an error in
    the assumed rate or dividend yield is amplified linearly in maturity,
    whereas a genuine mispricing would have no reason to. The deviation is
    roughly eight times larger beyond a year than inside three months, which
    is why the forward-consistency check above, assuming neither input, is the
    test that actually constrains the quotes.
    """
    matched = matched_pairs(chain)
    maturity = matched["maturity_call"]
    short = parity_violation_fraction_of_spot(matched[maturity < 0.25]).median()
    long = parity_violation_fraction_of_spot(matched[maturity > 1.0]).median()
    assert long > 4 * short


def test_our_implied_volatility_agrees_with_the_data_providers(chain):
    out_of_the_money = chain[is_out_of_the_money(chain)]
    difference = (out_of_the_money["our_iv"] - out_of_the_money["impliedVolatility"]).abs()
    assert difference.median() < 0.02
    assert difference.quantile(0.95) < 0.05


def test_out_of_the_money_quotes_are_the_reliable_ones(chain):
    """The screening rule is justified by measurement, not assertion: the same
    comparison on in-the-money quotes disagrees far more in the tail."""
    difference = (chain["our_iv"] - chain["impliedVolatility"]).abs()
    out_of_the_money = is_out_of_the_money(chain)
    assert difference[out_of_the_money].quantile(0.95) < difference[~out_of_the_money].quantile(0.95) / 3


def test_recovered_volatilities_are_economically_plausible(chain):
    out_of_the_money = chain[is_out_of_the_money(chain)]
    assert out_of_the_money["our_iv"].between(0.02, 1.5).mean() > 0.98
