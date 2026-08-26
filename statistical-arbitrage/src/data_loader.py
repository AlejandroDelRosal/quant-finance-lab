import pathlib

import pandas as pd

DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "prices_raw.csv"


def load_prices(path: pathlib.Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def train_test_split(series, train_fraction: float = 0.6):
    split = int(len(series) * train_fraction)
    return series.iloc[:split], series.iloc[split:]
