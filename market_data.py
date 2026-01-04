import yfinance as yf

def get_latest_prices(tickers:list):
    tickers_diccionary={}
    for ticker in tickers:
        dat=yf.Ticker(ticker)
        price=dat.info.get('currentPrice')
        tickers_diccionary[ticker]=price
    return tickers_diccionary
