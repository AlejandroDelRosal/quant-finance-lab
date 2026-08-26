'''
Project: Quantitative Risk Analysis: VaR & CVaR via Hybrid Stochastic Simulation
Description: Evaluates financial risk exposure by estimating Value-at-Risk (VaR) 
and Expected Shortfall (CVaR) using Monte Carlo simulations under Geometric 
Brownian Motion and Jump-Diffusion models.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Global Parameters & Data Preparation
data_url = 'https://hilpisch.com/tr_eikon_eod_data.csv'
price_history = pd.read_csv(data_url, index_col=0, parse_dates=True)
asset_list = ['AAPL.O', 'MSFT.O', 'AMZN.O']

time_horizon = 30 / 365.0
num_paths = 10000
confidence_levels = [0.01, 0.1, 1.0, 2.5, 5.0, 10.0]

# 2. Geometric Brownian Motion (GBM) Model
def evaluate_continuous_risk(ticker):
    """
    Simulates price paths using Geometric Brownian Motion (GBM), 
    plots the return distribution, and calculates both VaR and CVaR.
    """
    price_series = price_history[ticker]
    current_price = price_series.iloc[-1]
    log_returns = np.log(price_series / price_series.shift(1)).dropna()

    # Base parameter calibration
    annual_return = log_returns.mean() * 252
    annual_volatility = log_returns.std() * np.sqrt(252)

    # Vectorized Monte Carlo Simulation
    gaussian_noise = np.random.standard_normal(num_paths)
    future_prices = current_price * np.exp(
        (annual_return - 0.5 * annual_volatility**2) * time_horizon +
        annual_volatility * np.sqrt(time_horizon) * gaussian_noise
    )

    # Sort absolute returns for percentile scoring
    absolute_returns = np.sort(future_prices - current_price)

    # Distribution Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(absolute_returns, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Expected Absolute Return')
    plt.ylabel('Frequency')
    plt.title(f'Simulated Distribution (GBM) for {ticker}')
    plt.grid(axis='y', alpha=0.5)
    plt.show()

    # Calculate and print VaR & CVaR (Expected Shortfall)
    var_values = stats.scoreatpercentile(absolute_returns, confidence_levels)

    print(f"\nGBM Results for {ticker}")
    print(f"{'Confidence Level':>18} | {'Value-at-Risk':>15} | {'Expected Shortfall':>20}")
    print("-" * 60)
    for prob, var_val in zip(confidence_levels, var_values):
        # CVaR is the average of all returns worse than the VaR
        cvar_val = absolute_returns[absolute_returns <= var_val].mean()
        print(f"{100 - prob:>17.2f}% | {-var_val:>15.3f} | {-cvar_val:>20.3f}")

    return absolute_returns

# 3. Jump-Diffusion (JD) Model
def evaluate_jump_risk(ticker):
    """
    Simulates price paths using Merton's Jump-Diffusion model, capturing
    abnormal market shocks. Plots distribution and calculates VaR & CVaR.
    """
    price_series = price_history[ticker]
    current_price = price_series.iloc[-1]
    log_returns = np.log(price_series / price_series.shift(1)).dropna()

    annual_volatility = log_returns.std() * np.sqrt(252)
    annual_return = log_returns.mean() * 252

    # Jump detection and calibration
    jump_threshold = 3 * log_returns.std()
    jump_events = log_returns[np.abs(log_returns) > jump_threshold]

    jump_rate = len(jump_events) / (len(log_returns) / 252)
    jump_mean = jump_events.mean()
    jump_volatility = jump_events.std()

    # Drift adjustment due to jumps
    drift_compensation = jump_rate * (np.exp(jump_mean + 0.5 * jump_volatility**2) - 1)
    adjusted_return = annual_return - drift_compensation

    # Discretization setup
    steps = 100
    delta_t = time_horizon / steps

    price_matrix = np.zeros((steps + 1, num_paths))
    price_matrix[0] = current_price

    # Random variable generation
    z1 = np.random.standard_normal((steps + 1, num_paths))
    z2 = np.random.standard_normal((steps + 1, num_paths))
    poisson_dist = np.random.poisson(jump_rate * delta_t, (steps + 1, num_paths))

    # Iterative stochastic evolution
    for t in range(1, steps + 1):
        continuous_comp = np.exp((adjusted_return - 0.5 * annual_volatility**2) * delta_t + annual_volatility * np.sqrt(delta_t) * z1[t])
        jump_comp = (np.exp(jump_mean + jump_volatility * z2[t]) - 1) * poisson_dist[t]
        
        price_matrix[t] = price_matrix[t-1] * (continuous_comp + jump_comp)
        price_matrix[t] = np.maximum(price_matrix[t], 0) # Prevent negative prices

    absolute_returns = np.sort(price_matrix[-1] - current_price)

    # Distribution Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(absolute_returns, bins=50, color='salmon', edgecolor='black')
    plt.xlabel('Expected Absolute Return')
    plt.ylabel('Frequency')
    plt.title(f'Simulated Distribution (Jump-Diffusion) for {ticker}')
    plt.grid(axis='y', alpha=0.5)
    plt.show()

    # Calculate and print VaR & CVaR (Expected Shortfall)
    var_values = stats.scoreatpercentile(absolute_returns, confidence_levels)

    print(f"\nJump-Diffusion Results for {ticker}")
    print(f"{'Confidence Level':>18} | {'Value-at-Risk':>15} | {'Expected Shortfall':>20}")
    print("-" * 60)
    for prob, var_val in zip(confidence_levels, var_values):
        cvar_val = absolute_returns[absolute_returns <= var_val].mean()
        print(f"{100 - prob:>17.2f}% | {-var_val:>15.3f} | {-cvar_val:>20.3f}")

    return absolute_returns

# 4. Final Comparative Plot
def plot_risk_profiles(gbm_returns, jd_returns, ticker):
    """
    Plots a direct comparison of the VaR risk profiles across all 
    percentiles for both the continuous and jump-diffusion models.
    """
    percentile_range = list(np.arange(0.0, 10.1, 0.1))
    var_gbm_arr = stats.scoreatpercentile(gbm_returns, percentile_range)
    var_jd_arr = stats.scoreatpercentile(jd_returns, percentile_range)

    plt.figure(figsize=(10, 6))
    plt.plot(percentile_range, var_gbm_arr, color='navy', linewidth=2.0, linestyle='--', label='Continuous Process (GBM)')
    plt.plot(percentile_range, var_jd_arr, color='darkred', linewidth=2.0, label='Jump Process (JD)')
    plt.legend(loc='lower right')
    plt.title(f'VaR Risk Profile Comparison: {ticker}')
    plt.xlabel('Tail Probability [%]')
    plt.ylabel('Maximum Capital Exposure (VaR)')
    plt.ylim(ymax=0.0)
    plt.grid(True, alpha=0.3)
    plt.show()

# 5. Main Execution Block
if __name__ == '__main__':
    for asset in asset_list:
        print(f"\n{'='*60}\nInitiating Risk Evaluation for: {asset}\n{'='*60}")
        gbm_results = evaluate_continuous_risk(asset)
        jd_results = evaluate_jump_risk(asset)
        plot_risk_profiles(gbm_results, jd_results, asset)
