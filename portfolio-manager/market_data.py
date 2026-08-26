import yfinance as yf

def get_latest_prices(tickers:list):
    prices={}
    for ticker in tickers:
        dat=yf.Ticker(ticker)
        price=dat.info.get('currentPrice')
        prices[ticker]=price
    return prices
