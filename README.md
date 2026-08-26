# Quant Finance Lab

Modelado cuantitativo de finanzas: valuación de riesgo, pricing de derivados, backtesting algorítmico y NLP de sentimiento de mercado. Parte del [portafolio híbrido](https://github.com/AlejandroDelRosal/Portafolio) de Manuel Alejandro Del Rosal.

## Contenido actual

- **`quantitative-finance/`**
  - `VaR_with_gbm_and_jp.py` — Value at Risk y CVaR vía Geometric Brownian Motion y Merton Jump-Diffusion
  - `options_pricing_models.py` — Black-Scholes vs Monte Carlo para opciones europeas
  - `Financial time series analysis.py` — backtesting de cruce de medias móviles (SMA), correlación S&P 500 vs VIX
- **`financial-nlp-sentiment/`** — motor de sentimiento con VADER sobre titulares financieros (CNBC, Reuters, The Guardian)
- **`finance_engine.py`, `main.py`, `market_data.py`** — gestor de portafolio en tiempo real (OOP, integración con `yfinance`, costo promedio ponderado)

## Roadmap

- [ ] Backtesting multi-estrategia con métricas de riesgo (Sharpe, drawdown)
- [ ] Pricing de derivados exóticos
- [ ] Conectar el sentiment engine como señal del portfolio manager

## Convenciones

`main` protegido, todo por PR, conventional commits, CI con lint + tests. Ver [CHANGELOG.md](CHANGELOG.md).

## Licencia

MIT — ver [LICENSE](LICENSE).
