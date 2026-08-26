from __future__ import annotations

import numpy as np
from scipy.stats import norm


def d1_d2(spot, strike, maturity, rate, volatility, dividend_yield=0.0):
    variance = volatility * np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity) / variance
    return d1, d1 - variance


def call_price(spot, strike, maturity, rate, volatility, dividend_yield=0.0):
    d1, d2 = d1_d2(spot, strike, maturity, rate, volatility, dividend_yield)
    return spot * np.exp(-dividend_yield * maturity) * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)


def put_price(spot, strike, maturity, rate, volatility, dividend_yield=0.0):
    d1, d2 = d1_d2(spot, strike, maturity, rate, volatility, dividend_yield)
    return strike * np.exp(-rate * maturity) * norm.cdf(-d2) - spot * np.exp(-dividend_yield * maturity) * norm.cdf(-d1)


def price(option_type: str, spot, strike, maturity, rate, volatility, dividend_yield=0.0):
    if option_type == "call":
        return call_price(spot, strike, maturity, rate, volatility, dividend_yield)
    if option_type == "put":
        return put_price(spot, strike, maturity, rate, volatility, dividend_yield)
    raise ValueError(f"unknown option type: {option_type}")


def vega(spot, strike, maturity, rate, volatility, dividend_yield=0.0):
    """Sensitivity of price to volatility. Identical for calls and puts, since
    they differ only by the deterministic put-call parity terms."""
    d1, _ = d1_d2(spot, strike, maturity, rate, volatility, dividend_yield)
    return spot * np.exp(-dividend_yield * maturity) * norm.pdf(d1) * np.sqrt(maturity)


def intrinsic_value(option_type: str, spot, strike, maturity, rate, dividend_yield=0.0):
    """Lower no-arbitrage bound on the option price."""
    forward = spot * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    if option_type == "call":
        return np.maximum(forward - discounted_strike, 0.0)
    return np.maximum(discounted_strike - forward, 0.0)
