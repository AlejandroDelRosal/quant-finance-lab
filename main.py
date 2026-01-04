import matplotlib.pyplot as plt
from finance_engine import Portfolio
from market_data import get_latest_prices

#MAIN EXECUTION BLOCK
if __name__ == "__main__":

    # 1. Define current market prices 
    tickers=['AAPL','GOOGL','TSLA','AMZN']
    market_prices=get_latest_prices(tickers)
   

    # 2. Initialize the portfolio (starts empty)
    my_portfolio = Portfolio()

    # 3. Attempt to Load previous data
    try:
        # We load from 'my_data.json'
        my_portfolio.load_portafolio("my_data.json")
        print(f"Data loaded successfully.")
        print(f"Available Balance: ${my_portfolio.balance:.2f}")
    except FileNotFoundError:
        print("No previous save file found.")
        my_portfolio.balance = 10000.0 # Welcome gift for new users

    # 4. Display current inventory
    print("\nINVENTORY")
    chart_labels = []
    chart_values = []

    for asset in my_portfolio.assets:
        # Calculate current market value
        current_price = market_prices.get(asset.ticker, 0) # Use 0 if price not found
        market_value = asset.shares * current_price
        
        # Save data for the chart
        chart_labels.append(asset.ticker)
        chart_values.append(market_value)
        
        print(f"{asset.ticker}: {asset.shares} shares | Value: ${market_value:.2f}")

    # 5. Visualization 
    if len(chart_values) > 0:
        plt.figure(figsize=(6, 6))
        plt.pie(chart_values, labels=chart_labels, autopct='%1.1f%%')
        plt.title("Portfolio Distribution")
        plt.show()
    else:
        print("\nNo shares to plot.")

    # 6. Save before exiting (Safety)
    my_portfolio.save_portafolio("my_data.json")
    print("\nData saved")