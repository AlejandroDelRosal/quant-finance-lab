import pandas as pd

from src.cointegration import engle_granger_test


def screen_pairs(prices: pd.DataFrame, candidate_pairs, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for ticker_a, ticker_b in candidate_pairs:
        beta, _resid, adf_stat, p_value = engle_granger_test(prices[ticker_a], prices[ticker_b])
        rows.append(
            {
                "pair": f"{ticker_a}-{ticker_b}",
                "hedge_ratio": beta,
                "adf_statistic": adf_stat,
                "p_value": p_value,
                "cointegrated": p_value < alpha,
            }
        )
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
