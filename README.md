# Stock Portfolio Manager 

A robust, Object-Oriented Python application to manage a financial portfolio. This tool tracks cash balance, stock inventory, and calculates realized gains/losses using **Average Cost Basis** logic. It integrates with the `yfinance` API to provide real-time valuation and uses `matplotlib` for portfolio visualization.

## Key Features

* **Object-Oriented Design:** Modular architecture using `Asset` and `Portfolio` classes.
* **Real-Time Data:** Fetches live market prices using the Yahoo Finance API (`yfinance`).
* **Trade Logic:** Implements sophisticated buy/sell algorithms, updating average cost and calculating PnL (Profit and Loss).
* **Data Persistence:** Saves and loads portfolio state (cash + assets) using JSON serialization.
* **Visualization:** Generates a dynamic pie chart of current asset allocation.

## Tech Stack

* **Language:** Python 3.x
* **Libraries:**
    * `yfinance` (Market Data)
    * `matplotlib` (Data Visualization)
    * `json` (Standard Library for Persistence)

## 📂 Project Structure

```text
├── main.py              # Entry point: Handles execution, data fetching, and UI
├── finance_engine.py    # Core logic: Contains Asset and Portfolio classes
├── requirements.txt     # Dependency list
├── my_data.json         # (Auto-generated) Database for persistence
└── README.md            # Project documentation