import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def hedge_ratio(price_a, price_b) -> float:
    model = sm.OLS(price_a, sm.add_constant(price_b)).fit()
    return float(model.params.iloc[1])


def spread(price_a, price_b, beta: float):
    return price_a - beta * price_b


def engle_granger_test(price_a, price_b):
    """Engle & Granger 1987, Econometrica 55(2), 251-276.

    Two-step test: regress A on B, then test the OLS residual for a unit
    root with the Augmented Dickey-Fuller test. A stationary residual
    means A and B share a common stochastic trend (are cointegrated).
    """
    beta = hedge_ratio(price_a, price_b)
    resid = spread(price_a, price_b, beta)
    adf_stat, p_value, *_ = adfuller(resid, autolag="AIC")
    return beta, resid, adf_stat, p_value


def half_life(spread_series) -> float:
    """Ornstein-Uhlenbeck mean-reversion speed from an AR(1) fit on the
    spread (Chan, Algorithmic Trading, 2013, ch. 2)."""
    lagged = spread_series.shift(1).dropna()
    delta = spread_series.diff().dropna()
    lagged = lagged.loc[delta.index]
    model = sm.OLS(delta, sm.add_constant(lagged)).fit()
    theta = model.params.iloc[1]
    if theta >= 0:
        return np.inf
    return float(-np.log(2) / theta)
