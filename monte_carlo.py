import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime


# Function 1: Fetch Stock Data
def fetch_stock_data(ticker, start_date):
    """
    Fetches historical stock data from Yahoo Finance with the most recent available date.
    """
    end_date = datetime.today().strftime("%Y-%m-%d")  # Today's date in YYYY-MM-DD format
    stock = yf.download(ticker, start=start_date, end=end_date)

    if stock.empty:
        raise ValueError(f"❌ No data found for {ticker}. Check the ticker symbol and date range.")

    # Extract 'Close' price properly (handle MultiIndex case)
    if isinstance(stock.columns, pd.MultiIndex):
        return stock[("Close", ticker)]
    elif "Close" in stock.columns:
        return stock["Close"]
    else:
        raise KeyError("No 'Close' price column found in the data!")


# Function 2: Calculate Drift & Volatility
def calculate_drift_volatility(prices):
    returns = prices.pct_change().dropna()
    mu = returns.mean() * 252  # Annualized Drift
    sigma = returns.std() * np.sqrt(252)  # Annualized Volatility
    return mu, sigma

# Function 3: Monte Carlo Simulation (GBM)
def monte_carlo_simulation(S0, mu, sigma, start_date, T=1, simulations=1000):
    dt = 1/252  # Daily time step
    N = int(T/dt)  # Number of steps
    np.random.seed(42)  # Fix randomness for reproducibility

    # Generate proper dates for the simulation
    time = pd.date_range(start=start_date, periods=N, freq='B')  # 'B' for business days

    # Initialize price paths
    paths = np.zeros((N, simulations))
    paths[0] = S0

    for t in range(1, N):
        Z = np.random.standard_normal(simulations)  # Random normal shocks
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return time, paths

# Function 4: Plot Simulations vs. Real Data
def plot_simulations(time, paths, real_stock_prices, ticker):
    plt.figure(figsize=(10,5))

    # Monte Carlo Simulated Paths
    plt.plot(time, paths, alpha=0.05, color="blue")  # Lower opacity for clarity

    # Compute & Plot Confidence Intervals
    lower_bound = np.percentile(paths, 5, axis=1)
    upper_bound = np.percentile(paths, 95, axis=1)
    plt.fill_between(time, lower_bound, upper_bound, color="blue", alpha=0.2, label="90% Confidence Interval")

    # Plot Actual Stock Prices
    plt.plot(real_stock_prices.index, real_stock_prices, label=f"Actual {ticker} Prices", color="red")

    plt.title(f"Monte Carlo Simulations vs. {ticker} Prices")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    
    plt.savefig("results/monte_carlo_simulation.png", dpi=300)
    plt.show()


# Function 5: Estimate Probability of Hitting Target Price
def estimate_target_probability(paths, target_price):
    hit_target = np.any(paths >= target_price, axis=0)  # Count paths that reached target
    probability = np.mean(hit_target)  # Calculate probability
    return probability

def compute_confidence_intervals(paths):
    """
    Computes the expected stock price and confidence intervals from Monte Carlo simulations.
    Returns expected price, 5% lower bound, and 95% upper bound.
    """
    expected_price = np.mean(paths[-1])
    lower_bound = np.percentile(paths[-1], 5)  # 5% confidence lower bound
    upper_bound = np.percentile(paths[-1], 95)  # 95% confidence upper bound

    print(f"\n📊 Expected AAPL Price in 1 Year: ${expected_price:.2f}")
    print(f"🔻 5% Confidence Interval: ${lower_bound:.2f} - ${upper_bound:.2f}")

    return expected_price, lower_bound, upper_bound

def save_simulation_results(paths, time, filename="results/monte_carlo_simulation.csv"):
    """
    Saves the Monte Carlo simulation paths to a CSV file.
    Each column represents a simulated stock price path over time.
    """
    df_simulation = pd.DataFrame(paths, index=time)
    df_simulation.to_csv(filename)

    print(f"\n📊 Monte Carlo Simulation results saved to: {filename}")


# Main Execution
if __name__ == "__main__":
    # Define Parameters
    ticker = "AAPL"
    start_date = "2023-01-01"
    target_price = 200  # Price target

    # ✅ Fetch the most up-to-date stock data
    real_stock_prices = fetch_stock_data(ticker, start_date)
    S0 = real_stock_prices[-1]  # Latest stock price

    # Step 2: Compute drift & volatility
    mu, sigma = calculate_drift_volatility(real_stock_prices)

    print(f"\n📊 Estimated Annualized Drift (μ): {mu:.4f}")
    print(f"📊 Estimated Annualized Volatility (σ): {sigma:.4f}")

    # Step 3: Run Monte Carlo Simulation
    T = 2  # Simulate 2 years ahead instead of 1
    time, paths = monte_carlo_simulation(S0, mu, sigma, start_date, T)

    # Step 4: Compute Confidence Intervals
    expected_price, lower_bound, upper_bound = compute_confidence_intervals(paths)

    # Step 5: Save Simulation Results
    save_simulation_results(paths, time)

    # Step 6: Plot Results
    plot_simulations(time, paths, real_stock_prices, ticker)

    # Step 7: Estimate probability of reaching target price
    probability = estimate_target_probability(paths, target_price)
    print(f"\n📊 Probability of {ticker} reaching ${target_price} in 6 months: {probability:.2%}")

    print("\n✅ Done!")





