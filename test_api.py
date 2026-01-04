import yfinance as yf

dat= yf.Ticker('MSFT')

info=dat.info

print(f'El Precio Actual Es: {info.get('currentPrice')}')