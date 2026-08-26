from __future__ import annotations

import datetime as dt
import pathlib

import numpy as np
import pandas as pd
import yfinance as yf

from .implied_vol import implied_volatility_or_nan

DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "option_chain.csv"

# Screening thresholds. Illiquid contracts carry stale or nonsensical quotes
# that would otherwise dominate the fitted surface.
MIN_OPEN_INTEREST = 10
MAX_RELATIVE_SPREAD = 0.25
MIN_MID_PRICE = 0.05


def select_expirations(available: tuple[str, ...], count: int) -> list[str]:
    """Sample expirations evenly across the whole listed range.

    Taking the first N would return only weeklies clustered in the next month,
    which cannot show a term structure. Spreading the sample out reaches the
    monthly and longer-dated contracts as well.
    """
    if len(available) <= count:
        return list(available)
    positions = np.linspace(0, len(available) - 1, count).round().astype(int)
    return [available[position] for position in dict.fromkeys(positions)]


def download_chain(ticker: str = "SPY", max_expirations: int = 14) -> pd.DataFrame:
    security = yf.Ticker(ticker)
    spot = float(security.history(period="1d")["Close"].iloc[-1])
    today = dt.date.today()

    frames = []
    for expiry in select_expirations(security.options, max_expirations):
        chain = security.option_chain(expiry)
        for option_type, quotes in (("call", chain.calls), ("put", chain.puts)):
            frame = quotes[["strike", "bid", "ask", "impliedVolatility", "openInterest", "volume"]].copy()
            frame["option_type"] = option_type
            frame["expiry"] = expiry
            frame["maturity"] = (dt.date.fromisoformat(expiry) - today).days / 365.0
            frames.append(frame)

    chain = pd.concat(frames, ignore_index=True)
    chain["spot"] = spot
    return chain


def is_out_of_the_money(chain: pd.DataFrame) -> pd.Series:
    return ((chain["option_type"] == "call") & (chain["strike"] > chain["spot"])) | (
        (chain["option_type"] == "put") & (chain["strike"] < chain["spot"])
    )


def screen(chain: pd.DataFrame, out_of_the_money_only: bool = True) -> pd.DataFrame:
    """Keep only quotes that can carry reliable volatility information.

    Beyond the liquidity filters, this restricts to out-of-the-money contracts.
    A deep in-the-money option is almost entirely intrinsic value, so its vega
    is near zero and its price barely responds to volatility: inverting it
    amplifies quote noise into implausible implied volatilities.
    """
    chain = chain.copy()
    chain["mid"] = (chain["bid"] + chain["ask"]) / 2
    chain["relative_spread"] = (chain["ask"] - chain["bid"]) / chain["mid"].replace(0, np.nan)

    liquid = (
        (chain["maturity"] > 1 / 365)
        & (chain["bid"] > 0)
        & (chain["mid"] >= MIN_MID_PRICE)
        & (chain["openInterest"] >= MIN_OPEN_INTEREST)
        & (chain["relative_spread"] <= MAX_RELATIVE_SPREAD)
    )
    if out_of_the_money_only:
        liquid &= is_out_of_the_money(chain)
    return chain[liquid].reset_index(drop=True)


def add_implied_volatility(chain: pd.DataFrame, rate: float, dividend_yield: float) -> pd.DataFrame:
    chain = chain.copy()
    chain["our_iv"] = [
        implied_volatility_or_nan(row.mid, row.option_type, row.spot, row.strike, row.maturity, rate, dividend_yield)
        for row in chain.itertuples()
    ]
    chain["log_moneyness"] = np.log(chain["strike"] / chain["spot"])
    return chain.dropna(subset=["our_iv"]).reset_index(drop=True)


def load_cached_chain(path: pathlib.Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)
