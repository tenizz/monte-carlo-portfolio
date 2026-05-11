import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# -----------------------------
# 1. User settings
# -----------------------------

tickers_input = input("Enter stock tickers separated by commas, example AAPL,MSFT,SPY: ")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

weights_input = input("Enter portfolio weights separated by commas, example 0.3,0.3,0.4: ")
weights = np.array([float(weight.strip()) for weight in weights_input.split(",")])

if len(tickers) != len(weights):
    raise ValueError("Number of tickers must match number of weights.")

if not np.isclose(weights.sum(), 1):
    raise ValueError("Weights must add up to 1. Example: 0.3 + 0.3 + 0.4 = 1")

initial_investment = float(input("Enter initial investment amount: "))
monthly_contribution = float(input("Enter monthly contribution amount: "))

years = int(input("Enter number of years to simulate: "))
simulations = int(input("Enter number of simulations, example 5000: "))

trading_days = 252
risk_free_rate = 0.04  # 4%
inflation_rate = 0.03 # 3%

print()
print("Return assumption mode:")
print("1 = Historical normal simulation")
print("2 = Conservative 8% return simulation")
print("3 = Bootstrap historical simulation")

mode = input("Choose mode, 1, 2, or 3: ")

# -----------------------------
# 2. Download historical data
# -----------------------------

data = yf.download(tickers, start="2015-01-01")["Close"]
daily_returns = data.pct_change().dropna()

mean_returns = daily_returns.mean()
cov_matrix = daily_returns.cov()

if mode == "2":
    conservative_annual_return = 0.08
    conservative_daily_return = (1 + conservative_annual_return) ** (1 / trading_days) - 1

    mean_returns = pd.Series(
        [conservative_daily_return] * len(tickers),
        index=tickers
    )

    print()
    print("Using conservative mode:")
    print("Expected annual return set to 8%")
elif mode == "3":
    print()
    print("Using bootstrap historical simulation mode")
else:
    print()
    print("Using historical normal simulation mode")

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
# CAPM Beta vs SPY
# -----------------------------

benchmark = yf.download(
    "SPY",
    start="2015-01-01"
)["Close"]

benchmark_returns = benchmark.pct_change().dropna()

aligned_returns = pd.concat(
    [historical_portfolio_returns, benchmark_returns],
    axis=1
).dropna()

aligned_returns.columns = ["Portfolio", "SPY"]

cov_with_market = aligned_returns.cov().loc["Portfolio", "SPY"]

market_variance = aligned_returns["SPY"].var()

beta = float(cov_with_market / market_variance)

# -----------------------------
# 4. Monte Carlo simulation
# -----------------------------

days = years * trading_days
portfolio_results = np.zeros((days, simulations))

monthly_contribution_days = 21

for sim in range(simulations):

    if mode == "3":
        sampled_returns = daily_returns.sample(
            n=days,
            replace=True,
            ignore_index=True
        )

        portfolio_daily_returns = sampled_returns.to_numpy() @ weights

    else:
        simulated_returns = np.random.multivariate_normal(
            mean_returns,
            cov_matrix,
            days
        )

        portfolio_daily_returns = simulated_returns @ weights

    portfolio_value = initial_investment
    values = []

    for day in range(days):
        daily_growth = 1 + float(portfolio_daily_returns[day])

        # Prevent impossible return below -100%
        daily_growth = max(0, daily_growth)

        portfolio_value = portfolio_value * daily_growth

        if day % monthly_contribution_days == 0 and day != 0:
            portfolio_value += monthly_contribution

        values.append(portfolio_value)

    portfolio_results[:, sim] = values

# -----------------------------
# 5. Simulation results
# -----------------------------

final_values = portfolio_results[-1, :]

mean_final = float(np.mean(final_values))
median_final = float(np.median(final_values))
real_median_final = median_final / ((1 + inflation_rate) ** years)
# -----------------------------
# Stress Test Scenario
# -----------------------------

crash_size = 0.30

stressed_median = median_final * (1 - crash_size)

percentile_5 = float(np.percentile(final_values, 5))
percentile_95 = float(np.percentile(final_values, 95))

cvar_5 = float(np.asarray(final_values[final_values <= percentile_5]).mean())
prob_loss = np.mean(final_values < initial_investment)

total_contributions = initial_investment + monthly_contribution * years * 12
prob_below_contributions = np.mean(final_values < total_contributions)

# -----------------------------
# Summary Table
# -----------------------------

summary_table = pd.DataFrame({
    "Metric": [
        "Mean Final Value",
        "Median Final Value",
        "Inflation-Adjusted Median",
        "5th Percentile / VaR 5%",
        "CVaR 5%",
        "95th Percentile",
        "Probability Below Initial Investment",
        "Probability Below Total Contributions"
    ],
    "Value": [
        f"${mean_final:,.2f}",
        f"${median_final:,.2f}",
        f"${real_median_final:,.2f}",
        f"${percentile_5:,.2f}",
        f"${cvar_5:,.2f}",
        f"${percentile_95:,.2f}",
        f"{prob_loss:.2%}",
        f"{prob_below_contributions:.2%}"
    ]
})

# -----------------------------
# Efficient Frontier
# -----------------------------

num_random_portfolios = 10000

frontier_returns = []
frontier_volatility = []
frontier_sharpe = []
frontier_weights = []

for _ in range(num_random_portfolios):

    random_weights = np.random.random(len(tickers))
    random_weights = random_weights / np.sum(random_weights)

    random_daily_return = np.sum(mean_returns * random_weights)

    random_annual_return = (
                                   (1 + random_daily_return) ** trading_days
                           ) - 1

    random_annual_volatility = (
            np.sqrt(random_weights.T @ cov_matrix @ random_weights)
            * np.sqrt(trading_days)
    )

    random_sharpe = (
            (random_annual_return - risk_free_rate)
            / random_annual_volatility
    )

    frontier_returns.append(random_annual_return)
    frontier_volatility.append(random_annual_volatility)
    frontier_sharpe.append(random_sharpe)
    frontier_weights.append(random_weights)

frontier_returns = np.array(frontier_returns)
frontier_volatility = np.array(frontier_volatility)
frontier_sharpe = np.array(frontier_sharpe)

max_sharpe_index = np.argmax(frontier_sharpe)
min_volatility_index = np.argmin(frontier_volatility)

max_sharpe_weights = frontier_weights[max_sharpe_index]
min_volatility_weights = frontier_weights[min_volatility_index]

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
print(f"CAPM beta vs SPY: {beta:.2f}")
print()
print("Simulation Results")
print()
print("--------------------------------")
print(summary_table.to_string(index=False))
print("Stress Test")
print("--------------------------------")
print(f"Median value after immediate 30% crash: ${stressed_median:,.2f}")


print()
print("Efficient Frontier Results")
print("--------------------------------")

print("Maximum Sharpe Ratio Portfolio:")
for ticker, weight in zip(tickers, max_sharpe_weights):
    print(f"{ticker}: {weight:.2%}")

print()

print("Minimum Volatility Portfolio:")
for ticker, weight in zip(tickers, min_volatility_weights):
    print(f"{ticker}: {weight:.2%}")

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
plt.axvline(cvar_5, linestyle=":", label="CVaR 5%")
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

# -----------------------------
# Efficient Frontier Plot
# -----------------------------

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    frontier_volatility,
    frontier_returns,
    c=frontier_sharpe
)

plt.scatter(
    frontier_volatility[max_sharpe_index],
    frontier_returns[max_sharpe_index],
    marker="*",
    s=300,
    label="Max Sharpe Portfolio"
)

plt.scatter(
    frontier_volatility[min_volatility_index],
    frontier_returns[min_volatility_index],
    marker="*",
    s=300,
    label="Min Volatility Portfolio"
)

plt.title("Efficient Frontier")
plt.xlabel("Annualized Volatility")
plt.ylabel("Annualized Return")

plt.colorbar(scatter, label="Sharpe Ratio")

plt.legend()

plt.show()