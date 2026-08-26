from __future__ import annotations

import numpy as np
import pandas as pd


def fit_smile(log_moneyness: np.ndarray, implied_vol: np.ndarray) -> np.ndarray:
    """Quadratic fit in log-moneyness: level, slope (skew), curvature (smile)."""
    return np.polyfit(log_moneyness, implied_vol, 2)


FIT_BAND = 0.30


def smile_statistics(chain: pd.DataFrame, band: float = FIT_BAND) -> pd.DataFrame:
    """Per-maturity skew and curvature of the volatility smile.

    Black-Scholes assumes one constant volatility for every strike, which
    predicts slope and curvature of exactly zero at every maturity.

    The quadratic is fitted only within `band` of at-the-money. Over the full
    quoted range, which reaches far into the wings, a parabola is a poor
    description and the coefficients would depend mostly on how deep the
    listed strikes happen to go rather than on the shape near the money.
    """
    rows = []
    for maturity, group in chain.groupby("maturity"):
        slice_ = group[group["log_moneyness"].abs() <= band]
        if len(slice_) < 8:
            continue
        curvature, slope, level = fit_smile(slice_["log_moneyness"].to_numpy(), slice_["our_iv"].to_numpy())
        rows.append(
            {
                "maturity": maturity,
                "n_quotes": len(slice_),
                "atm_level": level,
                "skew": slope,
                "curvature": curvature,
                "iv_range": group["our_iv"].max() - group["our_iv"].min(),
            }
        )
    return pd.DataFrame(rows).sort_values("maturity").reset_index(drop=True)


def term_structure(chain: pd.DataFrame, moneyness_window: float = 0.02) -> pd.DataFrame:
    """At-the-money implied volatility as a function of maturity."""
    near_the_money = chain[chain["log_moneyness"].abs() <= moneyness_window]
    grouped = near_the_money.groupby("maturity")["our_iv"].agg(["mean", "count"])
    return grouped[grouped["count"] >= 2].reset_index()
