import numpy as np
import pandas as pd


def rolling_zscore(series, window: int = 20):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def generate_positions(zscore, entry: float = 2.0, exit: float = 0.5):
    """Gatev, Goetzmann & Rouwenhorst 2006, Rev. Financ. Stud. 19(3), 797-827."""
    position = np.zeros(len(zscore))
    current = 0
    for i, z in enumerate(zscore):
        if np.isnan(z):
            position[i] = current
            continue
        if current == 0:
            if z > entry:
                current = -1
            elif z < -entry:
                current = 1
        elif abs(z) < exit:
            current = 0
        position[i] = current
    return pd.Series(position, index=zscore.index)


def backtest_spread(spread_series, positions, cost_bps: float = 5.0):
    spread_return = spread_series.diff()
    turnover = positions.diff().abs().fillna(0)
    cost = turnover * (cost_bps / 1e4) * spread_series.abs()
    pnl = positions.shift(1).fillna(0) * spread_return - cost
    equity = pnl.cumsum()
    return pnl, equity


def performance_summary(pnl):
    equity = pnl.cumsum()
    sharpe = np.sqrt(252) * pnl.mean() / pnl.std() if pnl.std() > 0 else 0.0
    running_max = equity.cummax()
    drawdown = equity - running_max
    return {
        "total_pnl": float(equity.iloc[-1]),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "n_trades": int((pnl != 0).sum()),
    }
