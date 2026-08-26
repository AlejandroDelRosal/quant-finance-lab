from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .black_scholes import price, vega, intrinsic_value

MIN_VOLATILITY = 1e-4
MAX_VOLATILITY = 5.0


class NoSolution(Exception):
    """The quoted price admits no Black-Scholes implied volatility."""


def implied_volatility(
    market_price: float,
    option_type: str,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float = 0.0,
    tolerance: float = 1e-8,
    max_iterations: int = 60,
) -> float:
    """Invert Black-Scholes for volatility.

    Newton-Raphson converges quadratically because vega is available in closed
    form, but it is unreliable far from the money, where vega approaches zero
    and a Newton step can jump outside any sensible volatility range. The
    solver therefore falls back to Brent's method on a bracketed interval,
    which cannot diverge because option price is strictly increasing in
    volatility.
    """
    lower_bound = intrinsic_value(option_type, spot, strike, maturity, rate, dividend_yield)
    if market_price < lower_bound - 1e-10:
        raise NoSolution("quoted price is below intrinsic value, which violates no-arbitrage")

    def pricing_error(volatility: float) -> float:
        return price(option_type, spot, strike, maturity, rate, volatility, dividend_yield) - market_price

    volatility = 0.25
    for _ in range(max_iterations):
        error = pricing_error(volatility)
        if abs(error) < tolerance:
            return volatility
        sensitivity = vega(spot, strike, maturity, rate, volatility, dividend_yield)
        if sensitivity < 1e-10:
            break
        step = error / sensitivity
        candidate = volatility - step
        if not (MIN_VOLATILITY < candidate < MAX_VOLATILITY):
            break
        volatility = candidate

    if pricing_error(MIN_VOLATILITY) * pricing_error(MAX_VOLATILITY) > 0:
        raise NoSolution("no volatility in the searchable range reproduces this price")
    return float(brentq(pricing_error, MIN_VOLATILITY, MAX_VOLATILITY, xtol=tolerance))


def implied_volatility_or_nan(market_price, option_type, spot, strike, maturity, rate, dividend_yield=0.0):
    try:
        return implied_volatility(market_price, option_type, spot, strike, maturity, rate, dividend_yield)
    except (NoSolution, ValueError):
        return np.nan
