'''
Project: Derivatives Pricing Engine: Monte Carlo vs. Analytical Black-Scholes
Description: Estimates European Call and Put option prices using Monte Carlo 
simulations (GBM) and benchmarks them against the closed-form Black-Scholes model.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

# 1. Data Import & Parameter Setup
# Note: Ensure 'AAPL_stock.csv' is in your working directory.
stock_data = pd.read_csv('AAPL_stock.csv', index_col=0, parse_dates=True)

prices = stock_data['Price']
log_returns = np.log(prices / prices.shift(1)).dropna()

initial_price = prices.iloc[-1]    
drift = log_returns.mean()
volatility = log_returns.std()

time_to_maturity = 1.0          # Time to expiration in years (T)
time_step = 1 / 252.0           # Daily steps assuming 252 trading days (dt)
num_simulations = 500           # Number of Monte Carlo paths
strike_price = 110.0            # Strike price (K)
risk_free_rate = 0.0413         # Risk-free interest rate (r)

# 2. Geometric Brownian Motion (GBM) Engine
def generate_gbm_paths(S0, mu, sigma, T, dt, paths):
    """
    Simulates underlying asset price paths using Geometric Brownian Motion.
    Returns the time vector and a matrix of simulated price trajectories.
    """
    steps = int(T / dt)
    time_vector = np.linspace(0, T, steps)
    price_matrix = np.zeros((steps, paths))
    price_matrix[0, :] = S0
    
    for i in range(1, steps):
        wiener_increment = np.random.normal(0, np.sqrt(dt), size=paths)
        price_matrix[i, :] = price_matrix[i-1, :] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * wiener_increment)
    
    return time_vector, price_matrix

# 3. Black-Scholes Analytical Model
def calculate_black_scholes(S0, K, T, r, sigma):
    """
    Calculates the theoretical price of European Call and Put options 
    using the standard Black-Scholes-Merton formula.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
    return call_price, put_price

# 4. Execute Simulation & Visualization
time_vector, simulated_paths = generate_gbm_paths(
    initial_price, drift, volatility, time_to_maturity, time_step, num_simulations
)

plt.figure(figsize=(10, 6))
# Plotting paths with lower opacity (alpha) to see the density clearly
plt.plot(time_vector, simulated_paths, linewidth=0.5, alpha=0.6)
# Adding a horizontal line to visualize the Strike Price
plt.axhline(y=strike_price, color='red', linestyle='--', linewidth=2, label=f'Strike Price (K={strike_price})')

plt.xlabel("Time to Maturity (Years)")
plt.ylabel("Simulated Asset Price")
plt.title(f"Monte Carlo GBM Trajectories ({num_simulations} Simulations)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 5. Pricing Calculation & Comparison
final_prices = simulated_paths[-1, :]

# Monte Carlo Pricing (Discounted expected payoffs)
call_payoffs = np.maximum(final_prices - strike_price, 0)
put_payoffs = np.maximum(strike_price - final_prices, 0)

mc_call_price = np.mean(call_payoffs) * np.exp(-risk_free_rate * time_to_maturity)
mc_put_price = np.mean(put_payoffs) * np.exp(-risk_free_rate * time_to_maturity)

# Black-Scholes Pricing
bs_call_price, bs_put_price = calculate_black_scholes(
    initial_price, strike_price, time_to_maturity, risk_free_rate, volatility
)

# 6. Output Results
print(f"\nOption Pricing Results (Strike: {strike_price})")
print("-" * 60)
print(f"{'Pricing Method':<20} | {'Call Option':<15} | {'Put Option':<15}")
print("-" * 60)
print(f"{'Monte Carlo (GBM)':<20} | {mc_call_price:<15.4f} | {mc_put_price:<15.4f}")
print(f"{'Black-Scholes':<20} | {bs_call_price:<15.4f} | {bs_put_price:<15.4f}")
print("-" * 60)
