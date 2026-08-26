"""Cache the liquidity-screened chain, keeping both in and out of the money
quotes so the put-call parity check has matched pairs to work with."""
from volsurface.chain import download_chain, screen, add_implied_volatility, DATA_PATH

RATE = 0.042
DIVIDEND_YIELD = 0.012

chain = download_chain("SPY", max_expirations=14)
liquid = screen(chain, out_of_the_money_only=False)
enriched = add_implied_volatility(liquid, RATE, DIVIDEND_YIELD)
enriched.to_csv(DATA_PATH, index=False)
print(f"downloaded {len(chain)}, liquid {len(liquid)}, solved {len(enriched)}")
