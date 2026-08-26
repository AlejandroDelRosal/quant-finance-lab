import pathlib

import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_prices, train_test_split
from src.cointegration import hedge_ratio, spread, half_life, engle_granger_test
from src.screening import screen_pairs
from src.backtest import rolling_zscore, generate_positions, backtest_spread, performance_summary
from data.fetch_price_data import CANDIDATE_PAIRS

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"


def plot_screening_results(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2b6cb0" if c else "#a0aec0" for c in results["cointegrated"]]
    ax.barh(results["pair"], -np.log10(results["p_value"]), color=colors)
    ax.axvline(-np.log10(0.05), color="#c05621", linestyle="--", label="p = 0.05 threshold")
    ax.set_xlabel("-log10(p-value), Engle-Granger cointegration test")
    ax.set_title("Cointegration screen across 10 candidate pairs")
    ax.legend()
    fig.savefig(RESULTS_DIR / "cointegration_screen.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spread_and_signal(spread_test, zscore_test, positions):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(spread_test.index, spread_test, color="#2b6cb0")
    axes[0].set_ylabel("Spread: V - beta * MA")
    axes[0].set_title("Out-of-sample spread and trading signal (V-MA)")

    axes[1].plot(zscore_test.index, zscore_test, color="#2b6cb0")
    axes[1].axhline(2.0, color="#c05621", linestyle="--")
    axes[1].axhline(-2.0, color="#c05621", linestyle="--")
    axes[1].fill_between(zscore_test.index, 0, 3.5 * positions, color="#2b6cb0", alpha=0.15, step="mid")
    axes[1].set_ylabel("Z-score / position")
    axes[1].set_xlabel("Date")

    fig.savefig(RESULTS_DIR / "spread_and_signal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curve(equity, benchmark_v, benchmark_ma):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(equity.index, equity, label="Pairs trade (out-of-sample)", color="#2b6cb0")
    ax.plot(benchmark_v.index, benchmark_v - benchmark_v.iloc[0], label="Buy and hold V", color="#a0aec0", linewidth=1)
    ax.plot(benchmark_ma.index, benchmark_ma - benchmark_ma.iloc[0], label="Buy and hold MA", color="#c05621", linewidth=1)
    ax.set_ylabel("Cumulative P&L ($ per share)")
    ax.set_title("Out-of-sample performance: cointegration pairs trade vs buy and hold")
    ax.legend()
    fig.savefig(RESULTS_DIR / "equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    prices = load_prices()
    train, test = train_test_split(prices)

    results = screen_pairs(train, CANDIDATE_PAIRS)
    plot_screening_results(results)
    print("Cointegration screen (training period, 2021-2024):")
    print(results.to_string(index=False))

    beta = hedge_ratio(train["V"], train["MA"])
    spread_train = spread(train["V"], train["MA"], beta)
    hl = half_life(spread_train)
    print(f"\nSelected pair: V-MA, hedge ratio = {beta:.3f}, half-life = {hl:.1f} trading days")

    spread_test = spread(test["V"], test["MA"], beta)
    zscore_test = rolling_zscore(spread_test, window=20)
    positions = generate_positions(zscore_test, entry=2.0, exit=0.5)
    pnl, equity = backtest_spread(spread_test, positions, cost_bps=5.0)
    summary = performance_summary(pnl)

    plot_spread_and_signal(spread_test, zscore_test, positions)
    plot_equity_curve(equity, test["V"], test["MA"])

    print(f"\nOut-of-sample backtest, {len(test)} trading days:")
    print(f"  Total P&L: {summary['total_pnl']:.2f} $/share")
    print(f"  Sharpe ratio: {summary['sharpe_ratio']:.2f}")
    print(f"  Max drawdown: {summary['max_drawdown']:.2f} $/share")
    print(f"  Number of position changes: {summary['n_trades']}")


if __name__ == "__main__":
    main()
