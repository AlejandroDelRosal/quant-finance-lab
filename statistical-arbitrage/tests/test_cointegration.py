import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_prices, train_test_split
from src.cointegration import engle_granger_test, half_life, spread, hedge_ratio


def test_adf_detects_a_known_stationary_process():
    rng = np.random.default_rng(0)
    n = 2000
    ou = np.zeros(n)
    theta, sigma = 0.05, 1.0
    for i in range(1, n):
        ou[i] = ou[i - 1] - theta * ou[i - 1] + sigma * rng.standard_normal()
    series = pd.Series(ou)
    _beta, _resid, _stat, p_value = engle_granger_test(series, series.shift(1).fillna(0))
    assert p_value < 0.05


def test_real_visa_mastercard_pair_is_cointegrated_on_training_period():
    """Visa and Mastercard, a near-duopoly in card payment networks, share
    a stable long-run price relationship over 2021-2024."""
    prices = load_prices()
    train, _test = train_test_split(prices)
    _beta, _resid, _stat, p_value = engle_granger_test(train["V"], train["MA"])
    assert p_value < 0.01


def test_real_koke_pepsi_pair_is_not_cointegrated():
    """Control case: superficially similar consumer-staples businesses do
    not automatically imply a statistically stable price relationship."""
    prices = load_prices()
    train, _test = train_test_split(prices)
    _beta, _resid, _stat, p_value = engle_granger_test(train["KO"], train["PEP"])
    assert p_value > 0.05


def test_half_life_of_cointegrated_spread_is_positive_and_bounded():
    prices = load_prices()
    train, _test = train_test_split(prices)
    beta = hedge_ratio(train["V"], train["MA"])
    spread_series = spread(train["V"], train["MA"], beta)
    hl = half_life(spread_series)
    assert 0 < hl < len(train)
