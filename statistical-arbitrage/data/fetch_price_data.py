import pathlib

import yfinance as yf

CANDIDATE_PAIRS = [
    ("XOM", "CVX"), ("V", "MA"), ("JPM", "BAC"), ("HD", "LOW"), ("PG", "CL"),
    ("GS", "MS"), ("KO", "PEP"), ("MCD", "YUM"), ("UNH", "CI"), ("COST", "WMT"),
]
TICKERS = sorted({ticker for pair in CANDIDATE_PAIRS for ticker in pair})
PERIOD = "5y"
OUTPUT_PATH = pathlib.Path(__file__).parent / "prices_raw.csv"


def fetch() -> None:
    prices = yf.download(TICKERS, period=PERIOD, progress=False)["Close"].dropna()
    prices.to_csv(OUTPUT_PATH)


if __name__ == "__main__":
    fetch()
