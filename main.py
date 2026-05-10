import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# -----------------------------
# 1. User settings
# -----------------------------

tickers = ["AAPL", "MSFT", "SPY"]
weights = np.array([0.3, 0.3, 0.4])

initial_investment = 10000
monthly_contribution = 300

years = 10
simulations = 5000
trading_days = 252
risk_free_rate = 0.04  # 4%

# -----------------------------
# 2. Download historical data
# -----------------------------

data = yf.download(tickers, start="2015-01-01")["Close"]
daily_returns = data.pct_change().dropna()

mean_returns = daily_returns.mean()
cov_matrix = daily_returns.cov()

# -----------------------------
# 3. Portfolio statistics
# -----------------------------

portfolio_daily_return = np.sum(mean_returns * weights)
portfolio_daily_volatility = np.sqrt(weights.T @ cov_matrix @ weights)

annualized_return = (1 + portfolio_daily_return) ** trading_days - 1
annualized_volatility = portfolio_daily_volatility * np.sqrt(trading_days)

sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

# Historical portfolio performance
historical_portfolio_returns = daily_returns @ weights
historical_growth = (1 + historical_portfolio_returns).cumprod()

running_max = historical_growth.cummax()
drawdown = (historical_growth - running_max) / running_max
max_drawdown = drawdown.min()

# -----------------------------
# 4. Monte Carlo simulation
# -----------------------------

days = years * trading_days
portfolio_results = np.zeros((days, simulations))

monthly_contribution_days = 21

for sim in range(simulations):
    simulated_returns = np.random.multivariate_normal(
        mean_returns,
        cov_matrix,
        days
    )

    portfolio_daily_returns = simulated_returns @ weights

    portfolio_value = initial_investment
    values = []

    for day in range(days):
        portfolio_value = portfolio_value * (1 + portfolio_daily_returns[day])

        if day % monthly_contribution_days == 0 and day != 0:
            portfolio_value += monthly_contribution

        values.append(portfolio_value)

    portfolio_results[:, sim] = values

# -----------------------------
# 5. Simulation results
# -----------------------------

final_values = portfolio_results[-1, :]

mean_final = np.mean(final_values)
median_final = np.median(final_values)
percentile_5 = np.percentile(final_values, 5)
percentile_95 = np.percentile(final_values, 95)
prob_loss = np.mean(final_values < initial_investment)

total_contributions = initial_investment + monthly_contribution * years * 12
prob_below_contributions = np.mean(final_values < total_contributions)

# -----------------------------
# 6. Print results
# -----------------------------

print("Monte Carlo Portfolio Simulation")
print("--------------------------------")
print(f"Tickers: {tickers}")
print(f"Weights: {weights}")
print()
print(f"Initial investment: ${initial_investment:,.2f}")
print(f"Monthly contribution: ${monthly_contribution:,.2f}")
print(f"Total amount contributed: ${total_contributions:,.2f}")
print(f"Years: {years}")
print(f"Simulations: {simulations}")
print()
print("Historical Portfolio Statistics")
print("--------------------------------")
print(f"Annualized return: {annualized_return:.2%}")
print(f"Annualized volatility: {annualized_volatility:.2%}")
print(f"Sharpe ratio: {sharpe_ratio:.2f}")
print(f"Max drawdown: {max_drawdown:.2%}")
print()
print("Simulation Results")
print("--------------------------------")
print(f"Mean final value: ${mean_final:,.2f}")
print(f"Median final value: ${median_final:,.2f}")
print(f"5th percentile: ${percentile_5:,.2f}")
print(f"95th percentile: ${percentile_95:,.2f}")
print(f"Probability final value is below initial investment: {prob_loss:.2%}")
print(f"Probability final value is below total contributions: {prob_below_contributions:.2%}")

# -----------------------------
# 7. Plot simulations
# -----------------------------

plt.figure(figsize=(10, 6))
plt.plot(portfolio_results[:, :100])
plt.title("Monte Carlo Portfolio Simulation")
plt.xlabel("Trading Days")
plt.ylabel("Portfolio Value ($)")
plt.show()

# -----------------------------
# 8. Plot final value distribution
# -----------------------------

plt.figure(figsize=(10, 6))
plt.hist(final_values, bins=50)
plt.axvline(median_final, linestyle="--", label="Median")
plt.axvline(percentile_5, linestyle="--", label="5th percentile")
plt.axvline(percentile_95, linestyle="--", label="95th percentile")
plt.title("Distribution of Final Portfolio Values")
plt.xlabel("Final Portfolio Value ($)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# -----------------------------
# 9. Plot historical drawdown
# -----------------------------

plt.figure(figsize=(10, 6))
plt.plot(drawdown)
plt.title("Historical Portfolio Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.show()