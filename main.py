import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# -----------------------------
# 1. Portfolio settings
# -----------------------------

tickers = ["AAPL", "MSFT", "SPY"]
weights = np.array([0.3, 0.3, 0.4])

initial_investment = 10000
years = 10
simulations = 5000
trading_days = 252

# -----------------------------
# 2. Download historical prices
# -----------------------------

data = yf.download(tickers, start="2015-01-01")["Close"]

# Daily returns
daily_returns = data.pct_change().dropna()

# Average daily returns and covariance
mean_returns = daily_returns.mean()
cov_matrix = daily_returns.cov()

# -----------------------------
# 3. Monte Carlo simulation
# -----------------------------

days = years * trading_days
portfolio_results = np.zeros((days, simulations))

for sim in range(simulations):
    simulated_returns = np.random.multivariate_normal(
        mean_returns,
        cov_matrix,
        days
    )

    portfolio_daily_returns = simulated_returns @ weights

    portfolio_values = initial_investment * np.cumprod(1 + portfolio_daily_returns)

    portfolio_results[:, sim] = portfolio_values

# -----------------------------
# 4. Results
# -----------------------------

final_values = portfolio_results[-1, :]

mean_final = np.mean(final_values)
median_final = np.median(final_values)
percentile_5 = np.percentile(final_values, 5)
percentile_95 = np.percentile(final_values, 95)
prob_loss = np.mean(final_values < initial_investment)

print("Monte Carlo Portfolio Simulation")
print("--------------------------------")
print(f"Initial investment: ${initial_investment:,.2f}")
print(f"Years: {years}")
print(f"Simulations: {simulations}")
print()
print(f"Mean final value: ${mean_final:,.2f}")
print(f"Median final value: ${median_final:,.2f}")
print(f"5th percentile: ${percentile_5:,.2f}")
print(f"95th percentile: ${percentile_95:,.2f}")
print(f"Probability of losing money: {prob_loss:.2%}")

# -----------------------------
# 5. Plot simulations
# -----------------------------

plt.figure(figsize=(10, 6))
plt.plot(portfolio_results[:, :100])
plt.title("Monte Carlo Portfolio Simulation")
plt.xlabel("Trading Days")
plt.ylabel("Portfolio Value ($)")
plt.show()

# Final value distribution
plt.figure(figsize=(10, 6))
plt.hist(final_values, bins=50)
plt.title("Distribution of Final Portfolio Values")
plt.xlabel("Final Portfolio Value ($)")
plt.ylabel("Frequency")
plt.show()