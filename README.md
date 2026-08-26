# Quant Finance Lab

Quantitative finance modeling: risk valuation, derivatives pricing, algorithmic backtesting, and market sentiment analysis. Part of the [portfolio](https://github.com/AlejandroDelRosal/Portafolio) of Manuel Alejandro Del Rosal.

## Contents

- [`implied-volatility-surface/`](implied-volatility-surface/): recovering the volatility surface from live SPY option quotes by numerical inversion of Black-Scholes, measuring a skew that is negative at every maturity against a model that predicts zero
- [`statistical-arbitrage/`](statistical-arbitrage/): cointegration screening across ten real equity pairs and an out-of-sample backtest of the strongest candidate (Engle-Granger test, Ornstein-Uhlenbeck half-life)
- `quantitative-finance/`
  - `VaR_with_gbm_and_jp.py`: Value at Risk and CVaR using Geometric Brownian Motion and Merton Jump-Diffusion
  - `options_pricing_models.py`: Black-Scholes vs Monte Carlo for European options
  - `financial_time_series_analysis.py`: SMA crossover backtesting, S&P 500 vs VIX correlation
- `financial-nlp-sentiment/`: VADER-based sentiment engine over financial news headlines (CNBC, Reuters, The Guardian)
- `portfolio-manager/`: real-time portfolio manager (OOP, `yfinance` integration, weighted average cost basis)

## Roadmap

- [ ] Multi-strategy backtesting with risk metrics (Sharpe ratio, drawdown)
- [ ] Exotic derivatives pricing
- [ ] Connect the sentiment engine as a signal for the portfolio manager
