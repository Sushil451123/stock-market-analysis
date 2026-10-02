"""
Stock Market Technical & Risk Analysis Engine
Author: Financial Data Analytics Portfolio
Description: Analyzes equities, generates technical indicators, and calculates portfolio risk metrics.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf


def fetch_stock_data(tickers, start_date="2023-01-01", end_date=None):
    """Download historical adjusted close prices from Yahoo Finance."""
    print(f"[INFO] Fetching historical data for: {tickers}...")
    df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)[
        "Close"
    ]
    df = df.dropna()
    return df


def calculate_technical_indicators(series):
    """Compute SMA 20, SMA 50, RSI (14), and Bollinger Bands."""
    data = pd.DataFrame({"Close": series})

    # Simple Moving Averages
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()

    # Bollinger Bands
    rolling_std = data["Close"].rolling(window=20).std()
    data["BB_Upper"] = data["SMA_20"] + (rolling_std * 2)
    data["BB_Lower"] = data["SMA_20"] - (rolling_std * 2)

    # Relative Strength Index (RSI - 14)
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    return data


def compute_risk_metrics(returns_df, benchmark_col, risk_free_rate=0.06):
    """Calculate Annualized Return, Volatility, Sharpe Ratio, and Beta."""
    trading_days = 252
    metrics = {}

    benchmark_ret = returns_df[benchmark_col]

    for col in returns_df.columns:
        ann_return = returns_df[col].mean() * trading_days
        ann_vol = returns_df[col].std() * np.sqrt(trading_days)
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else 0

        # Beta against benchmark
        covariance = np.cov(returns_df[col], benchmark_ret)[0][1]
        benchmark_var = np.var(benchmark_ret)
        beta = covariance / benchmark_var if benchmark_var != 0 else 1.0

        # Historical Value at Risk (95% confidence)
        var_95 = np.percentile(returns_df[col], 5)

        metrics[col] = {
            "Annualized Return": f"{ann_return * 100:.2f}%",
            "Annualized Volatility": f"{ann_vol * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Beta vs Benchmark": f"{beta:.2f}",
            "Daily VaR (95%)": f"{var_95 * 100:.2f}%",
        }

    return pd.DataFrame(metrics).T


def generate_plots(prices, single_stock_tech, stock_symbol):
    """Generate and save technical and comparative charts."""
    os.makedirs("output", exist_ok=True)

    # 1. Technical Indicators Plot
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(single_stock_tech.index, single_stock_tech["Close"], label="Close Price", color="black", lw=1.5)
    ax1.plot(single_stock_tech.index, single_stock_tech["SMA_20"], label="20-Day SMA", color="blue", ls="--")
    ax1.plot(single_stock_tech.index, single_stock_tech["SMA_50"], label="50-Day SMA", color="orange", ls="--")
    ax1.fill_between(
        single_stock_tech.index,
        single_stock_tech["BB_Upper"],
        single_stock_tech["BB_Lower"],
        color="gray",
        alpha=0.15,
        label="Bollinger Bands",
    )
    ax1.set_title(f"Technical Analysis: {stock_symbol} (Price & Trends)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # RSI Subplot
    ax2.plot(single_stock_tech.index, single_stock_tech["RSI"], color="purple", lw=1.5, label="RSI (14)")
    ax2.axhline(70, color="red", linestyle="--", alpha=0.7, label="Overbought (70)")
    ax2.axhline(30, color="green", linestyle="--", alpha=0.7, label="Oversold (30)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Relative Strength Index (RSI)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/technical_indicators.png", dpi=300)
    plt.close()
    print("[INFO] Saved plot: output/technical_indicators.png")

    # 2. Cumulative Return Comparison
    cumulative_returns = (1 + prices.pct_change().dropna()).cumprod() - 1
    plt.figure(figsize=(12, 6))
    for col in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[col] * 100, label=col, lw=2)
    plt.title("Comparative Cumulative Returns (%)")
    plt.ylabel("Return (%)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/cumulative_returns.png", dpi=300)
    plt.close()
    print("[INFO] Saved plot: output/cumulative_returns.png")


if __name__ == "__main__":
    # Selected Large Cap Equities + Nifty 50 Benchmark (^NSEI)
    symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "^NSEI"]
    primary_stock = "RELIANCE.NS"

    prices = fetch_stock_data(symbols)
    daily_returns = prices.pct_change().dropna()

    # Risk Metrics Table
    risk_summary = compute_risk_metrics(daily_returns, benchmark_col="^NSEI")
    print("\n" + "=" * 55)
    print("           PORTFOLIO & RISK METRICS SUMMARY      ")
    print("=" * 55)
    print(risk_summary.to_string())

    # Technical Indicators for Primary Stock
    tech_data = calculate_technical_indicators(prices[primary_stock])

    # Visualizations
    generate_plots(prices, tech_data, primary_stock)
    print("\n[SUCCESS] Stock market analytics engine finished execution.")