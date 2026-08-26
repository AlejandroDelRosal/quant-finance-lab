"""Build the volatility surface from the cached chain and report what it shows."""

from __future__ import annotations

import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

from volsurface.chain import load_cached_chain, is_out_of_the_money
from volsurface.smile import smile_statistics, term_structure, fit_smile, FIT_BAND

FIGURES = pathlib.Path(__file__).parent / "figures"


def plot_surface(chain):
    x = chain["log_moneyness"].to_numpy()
    y = chain["maturity"].to_numpy()
    z = chain["our_iv"].to_numpy()

    grid_x, grid_y = np.meshgrid(
        np.linspace(np.percentile(x, 1), np.percentile(x, 99), 90),
        np.linspace(y.min(), np.percentile(y, 99), 90),
    )
    grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")

    figure = plt.figure(figsize=(11, 7))
    axes = figure.add_subplot(projection="3d")
    axes.plot_surface(grid_x, grid_y, grid_z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.9)
    axes.set_xlabel("log moneyness  ln(K/S)")
    axes.set_ylabel("maturity (years)")
    axes.set_zlabel("implied volatility")
    axes.set_title(f"SPY implied volatility surface, {len(chain)} out-of-the-money quotes")
    axes.view_init(elev=22, azim=-125)
    figure.savefig(FIGURES / "volatility_surface.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_smiles(chain, maturities):
    figure, axes = plt.subplots(figsize=(8, 5))
    colors = cm.viridis(np.linspace(0, 0.9, len(maturities)))
    for maturity, color in zip(maturities, colors):
        slice_ = chain[chain["maturity"] == maturity].sort_values("log_moneyness")
        axes.plot(slice_["log_moneyness"], slice_["our_iv"], "o", markersize=3, color=color, alpha=0.6)
        near = slice_[slice_["log_moneyness"].abs() <= FIT_BAND]
        curve = np.poly1d(fit_smile(near["log_moneyness"].to_numpy(), near["our_iv"].to_numpy()))
        grid = np.linspace(-FIT_BAND, FIT_BAND, 200)
        axes.plot(grid, curve(grid), color=color, label=f"{maturity * 365:.0f} days")

    axes.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    axes.set_xlabel("log moneyness  ln(K/S)")
    axes.set_ylabel("implied volatility")
    axes.set_title("Volatility smile: Black-Scholes predicts a flat line at every maturity")
    axes.set_xlim(-1.2, 0.7)
    axes.legend(title="maturity")
    figure.savefig(FIGURES / "volatility_smile.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_term_structure(structure):
    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.plot(structure["maturity"] * 365, structure["mean"], "o-", color="#2b6cb0")
    axes.set_xlabel("maturity (days)")
    axes.set_ylabel("at-the-money implied volatility")
    axes.set_title("Term structure of at-the-money volatility")
    figure.savefig(FIGURES / "term_structure.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    full_chain = load_cached_chain()
    chain = full_chain[is_out_of_the_money(full_chain)].reset_index(drop=True)

    print(f"Quotes surviving the liquidity and moneyness screen: {len(chain)}")
    print(f"Spot: {chain['spot'].iloc[0]:.2f}")
    print(f"Maturities: {chain['maturity'].nunique()}, from {chain['maturity'].min() * 365:.0f} to "
          f"{chain['maturity'].max() * 365:.0f} days")

    difference = (chain["our_iv"] - chain["impliedVolatility"]).abs()
    print(f"\nAgreement with the data provider's own implied volatility:")
    print(f"  median absolute difference {difference.median():.5f}, 95th percentile {difference.quantile(0.95):.5f}")

    in_the_money = full_chain[~is_out_of_the_money(full_chain)]
    itm_difference = (in_the_money["our_iv"] - in_the_money["impliedVolatility"]).abs()
    print(f"  the same comparison on in-the-money quotes: 95th percentile {itm_difference.quantile(0.95):.5f}")

    statistics = smile_statistics(chain)
    print(f"\nPer-maturity smile fit (Black-Scholes predicts skew = curvature = 0):")
    print(statistics.head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSkew is negative at {(statistics['skew'] < 0).mean():.0%} of maturities, "
          f"median {statistics['skew'].median():.3f}")
    print(f"Curvature is positive at {(statistics['curvature'] > 0).mean():.0%} of maturities, "
          f"median {statistics['curvature'].median():.3f}")
    print(f"Median implied volatility range within a single maturity: {statistics['iv_range'].median():.3f}")

    plot_surface(chain)
    plot_smiles(chain, statistics.nlargest(4, "n_quotes")["maturity"].sort_values().tolist())
    structure = term_structure(chain)
    if len(structure) > 1:
        plot_term_structure(structure)
        print(f"\nAt-the-money volatility runs from {structure['mean'].iloc[0]:.3f} at "
              f"{structure['maturity'].iloc[0] * 365:.0f} days to {structure['mean'].iloc[-1]:.3f} at "
              f"{structure['maturity'].iloc[-1] * 365:.0f} days")


if __name__ == "__main__":
    main()
