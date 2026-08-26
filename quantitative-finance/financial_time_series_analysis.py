'''
Project: Quantitative Market Analysis & Trading Strategy Backtester
Description: Automates data extraction, SMA crossover strategy backtesting, 
and statistical correlation analysis (SPX vs VIX) using pandas and numpy.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import warnings

# Configure plotting style and suppress pandas future warnings
plt.style.use('ggplot')
warnings.simplefilter(action='ignore', category=FutureWarning)

# 1. Global Parameters
DATA_URL = 'https://hilpisch.com/tr_eikon_eod_data.csv'

# 2. Technical Analysis & Backtesting Engine
def analyze_sma_strategy(data_url, ticker='AMZN.O', short_window=42, long_window=252):
    """
    Downloads historical data, calculates short and long SMAs, generates 
    trading signals, and calculates the cumulative returns of the strategy.
    """
    # Load and clean data
    raw_data = pd.read_csv(data_url, index_col=0, parse_dates=True)
    df = pd.DataFrame(raw_data[ticker]).dropna()
    
    # Calculate Log Returns
    df['Log_Returns'] = np.log(df[ticker] / df[ticker].shift(1))
    
    # Calculate Simple Moving Averages (SMA)
    df['SMA_Short'] = df[ticker].rolling(window=short_window).mean()
    df['SMA_Long'] = df[ticker].rolling(window=long_window).mean()
    df.dropna(inplace=True)
    
    # Generate Trading Signals (1 for Long, -1 for Short)
    df['Position'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, -1)
    
    # THE PLUS: Calculate Strategy Returns (Shifted position * log returns)
    df['Strategy_Returns'] = df['Position'].shift(1) * df['Log_Returns']
    df.dropna(inplace=True)
    
    # Visualization: Price and SMAs
    fig, ax1 = plt.subplots(figsize=(10, 6))
    df[[ticker, 'SMA_Short', 'SMA_Long']].plot(ax=ax1, lw=1.5)
    ax1.set_title(f'Technical Analysis: SMA Crossover Strategy ({ticker})')
    ax1.set_ylabel('Price')
    
    # Visualization: Strategy Performance (Buy & Hold vs SMA Strategy)
    fig, ax2 = plt.subplots(figsize=(10, 6))
    cumulative_returns = df[['Log_Returns', 'Strategy_Returns']].cumsum().apply(np.exp)
    cumulative_returns.columns = ['Buy & Hold', 'SMA Strategy']
    cumulative_returns.plot(ax=ax2, lw=2.0)
    ax2.set_title(f'Backtest Performance: Buy & Hold vs SMA Strategy ({ticker})')
    ax2.set_ylabel('Cumulative Returns')
    
    plt.show()
    return df

# 3. Market Correlation Analysis
def analyze_market_correlation(data_url, index_ticker='.SPX', vol_ticker='.VIX'):
    """
    Analyzes the relationship between an equity index and a volatility index 
    using OLS regression and rolling correlation.
    """
    raw_data = pd.read_csv(data_url, index_col=0, parse_dates=True)
    df = raw_data[[index_ticker, vol_ticker]].dropna()
    
    # Calculate log returns
    returns = np.log(df / df.shift(1)).dropna()
    
    # OLS Regression
    regression_coeffs = np.polyfit(returns[index_ticker], returns[vol_ticker], deg=1)
    
    # Visualization: Scatter Plot with Regression Line
    fig, ax1 = plt.subplots(figsize=(10, 6))
    returns.plot(kind='scatter', x=index_ticker, y=vol_ticker, alpha=0.5, ax=ax1)
    ax1.plot(returns[index_ticker], np.polyval(regression_coeffs, returns[index_ticker]), 'r', lw=2)
    ax1.set_title(f'OLS Regression: {index_ticker} vs {vol_ticker} Returns')
    
    # Visualization: 252-Day Rolling Correlation
    fig, ax2 = plt.subplots(figsize=(10, 6))
    rolling_corr = returns[index_ticker].rolling(window=252).corr(returns[vol_ticker])
    rolling_corr.plot(ax=ax2, color='navy')
    
    # Plot static overall correlation as a baseline
    static_corr = returns.corr().iloc[0, 1]
    ax2.axhline(static_corr, color='red', linestyle='--', label=f'Static Correlation: {static_corr:.2f}')
    
    ax2.set_title(f'252-Day Rolling Correlation ({index_ticker} & {vol_ticker})')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.legend()
    
    plt.show()

# 4. Utility Algorithms
def check_magazine(magazine, note):
    """
    Verifies if a ransom note can be formed using the words from a magazine.
    Optimized using Python's collections.Counter for O(N) time complexity.
    """
    mag_count = Counter(magazine)
    note_count = Counter(note)
    
    # Subtracting note requirements from available magazine words
    # If the note requires more of any word than the magazine has, it fails
    if note_count - mag_count == Counter():
        print('Yes: The note can be created.')
        return True
    else:
        print('No: Insufficient words in the magazine.')
        return False

# 5. Execution Block
if __name__ == '__main__':
    print("="*60)
    print("Executing Quantitative Analysis Engine...")
    print("="*60)
    
    # Run SMA Backtest for Amazon
    print("\n1. Running SMA Crossover Backtest for AMZN.O...")
    strategy_data = analyze_sma_strategy(DATA_URL, ticker='AMZN.O')
    
    # Run SPX vs VIX Correlation
    print("\n2. Running Market Correlation Analysis (SPX vs VIX)...")
    analyze_market_correlation(DATA_URL)
    
    # Test the Magazine Algorithm
    print("\n3. Testing Magazine/Note Algorithm...")
    mag_words = "give me one grand today night".split()
    note_words = "give one grand today".split()
    check_magazine(mag_words, note_words)
