import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from portfolio_engine import (
    prepare_mean_returns,
    run_monte_carlo_simulation,
    calculate_efficient_frontier,
    calculate_simulation_metrics,
)

# ------------------------------------------------------------------
# 1. User inputs
# ------------------------------------------------------------------

tickers_input = input("Enter stock tickers separated by commas, example AAPL,MSFT,SPY: ")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

weights_input = input("Enter portfolio weights separated by commas, example 0.3,0.3,0.4: ")
weights = np.array([float(w.strip()) for w in weights_input.split(",")])

if len(tickers) != len(weights):
    raise ValueError("Number of tickers must match number of weights.")

if not np.isclose(weights.sum(), 1):
    raise ValueError(f"Weights must add up to 1. Current sum: {weights.sum():.4f}")

initial_investment   = float(input("Enter initial investment amount: "))
monthly_contribution = float(input("Enter monthly contribution amount: "))
years                = int(input("Enter number of years to simulate: "))
simulations          = int(input("Enter number of simulations, example 5000: "))

trading_days   = 252
risk_free_rate = 0.04
inflation_rate = 0.03

print()
print("Return assumption mode:")
print("  1 = Historical normal simulation")
print("  2 = Conservative 8% return simulation")
print("  3 = Bootstrap historical simulation")

mode_input = input("Choose mode (1, 2, or 3): ").strip()

MODE_MAP = {
    "1": "Historical normal simulation",
    "2": "Conservative 8% return simulation",
    "3": "Bootstrap historical simulation",
}

if mode_input not in MODE_MAP:
    raise ValueError(f"Invalid mode '{mode_input}'. Choose 1, 2, or 3.")

mode = MODE_MAP[mode_input]
print(f"\nUsing: {mode}")

# ------------------------------------------------------------------
# 2. Download historical data
# ------------------------------------------------------------------

print("\nDownloading historical market data...")
data = yf.download(tickers, start="2015-01-01")["Close"]

if isinstance(data, pd.Series):
    data = data.to_frame()

data = data.ffill().dropna()
daily_returns = data[tickers].pct_change().dropna()

# ------------------------------------------------------------------
# 3. Historical portfolio statistics
# ------------------------------------------------------------------

mean_returns = prepare_mean_returns(daily_returns, tickers, mode, trading_days)
cov_matrix   = daily_returns.cov()

portfolio_daily_return    = float(np.sum(mean_returns * weights))
portfolio_daily_vol       = float(np.sqrt(weights.T @ cov_matrix @ weights))
annualized_return         = (1 + portfolio_daily_return) ** trading_days - 1
annualized_volatility     = portfolio_daily_vol * np.sqrt(trading_days)
sharpe_ratio              = (annualized_return - risk_free_rate) / annualized_volatility

historical_portfolio_returns = daily_returns @ weights
historical_growth            = (1 + historical_portfolio_returns).cumprod()
running_max                  = historical_growth.cummax()
drawdown                     = (historical_growth - running_max) / running_max
max_drawdown                 = float(drawdown.min())

# CAPM beta vs SPY
print("Downloading SPY benchmark data...")
benchmark         = yf.download("SPY", start="2015-01-01")["Close"]
benchmark_returns = benchmark.pct_change().dropna()

aligned = pd.concat(
    [historical_portfolio_returns, benchmark_returns], axis=1
).dropna()
aligned.columns = ["Portfolio", "SPY"]

beta = float(aligned.cov().loc["Portfolio", "SPY"] / aligned["SPY"].var())

# ------------------------------------------------------------------
# 4. Monte Carlo simulation  (vectorized — no Python loops)
# ------------------------------------------------------------------

print(f"\nRunning {simulations:,} simulations...")

portfolio_results, final_values = run_monte_carlo_simulation(
    daily_returns=daily_returns,
    mean_returns=mean_returns,
    cov_matrix=cov_matrix,
    weights=weights,
    initial_investment=initial_investment,
    monthly_contribution=monthly_contribution,
    years=years,
    simulations=simulations,
    mode=mode,
    trading_days=trading_days,
)

# ------------------------------------------------------------------
# 5. Simulation metrics  (unified with web app)
# ------------------------------------------------------------------

metrics = calculate_simulation_metrics(
    final_values=final_values,
    initial_investment=initial_investment,
    monthly_contribution=monthly_contribution,
    years=years,
    inflation_rate=inflation_rate,
)

# ------------------------------------------------------------------
# 6. Efficient frontier  (vectorized — no Python loops)
# ------------------------------------------------------------------

print("Calculating efficient frontier...")

frontier_df = calculate_efficient_frontier(
    tickers=tickers,
    mean_returns=mean_returns,
    cov_matrix=cov_matrix,
    risk_free_rate=risk_free_rate,
    trading_days=trading_days,
    num_random_portfolios=10000,
)

max_sharpe_row = frontier_df.loc[frontier_df["Sharpe Ratio"].idxmax()]
min_vol_row    = frontier_df.loc[frontier_df["Volatility"].idxmin()]

weight_cols = [f"w_{t}" for t in tickers]

# ------------------------------------------------------------------
# 7. Export CSVs
# ------------------------------------------------------------------

metrics["summary_table"].to_csv("simulation_summary.csv", index=False)
frontier_df.to_csv("efficient_frontier.csv", index=False)
pd.DataFrame({"Final Portfolio Value": final_values}).to_csv(
    "final_values.csv", index=False
)

# ------------------------------------------------------------------
# 8. Print results
# ------------------------------------------------------------------

print()
print("Monte Carlo Portfolio Simulation")
print("=" * 50)
print(f"Tickers:              {tickers}")
print(f"Weights:              {weights}")
print(f"Initial investment:   ${initial_investment:,.2f}")
print(f"Monthly contribution: ${monthly_contribution:,.2f}")
print(f"Total contributed:    ${metrics['total_contributions']:,.2f}")
print(f"Years:                {years}")
print(f"Simulations:          {simulations:,}")
print(f"Mode:                 {mode}")

print()
print("Historical Portfolio Statistics")
print("-" * 50)
print(f"Annualized return:     {annualized_return:.2%}")
print(f"Annualized volatility: {annualized_volatility:.2%}")
print(f"Sharpe ratio:          {sharpe_ratio:.2f}")
print(f"Max drawdown:          {max_drawdown:.2%}")
print(f"CAPM beta vs SPY:      {beta:.2f}")

print()
print("Simulation Results")
print("-" * 50)
print(metrics["summary_table"].to_string(index=False))

print()
print("Efficient Frontier")
print("-" * 50)
print("Maximum Sharpe Ratio Portfolio:")
for ticker in tickers:
    print(f"  {ticker}: {max_sharpe_row[f'w_{ticker}']:.2%}")

print()
print("Minimum Volatility Portfolio:")
for ticker in tickers:
    print(f"  {ticker}: {min_vol_row[f'w_{ticker}']:.2%}")

# ------------------------------------------------------------------
# 9. Plots
# ------------------------------------------------------------------

# Monte Carlo paths
plt.figure(figsize=(10, 6))
plt.plot(portfolio_results[:, :100], linewidth=0.6, alpha=0.5)
plt.title("Monte Carlo Portfolio Simulation")
plt.xlabel("Trading Days")
plt.ylabel("Portfolio Value ($)")
plt.tight_layout()
plt.savefig("monte_carlo_simulation.png", dpi=300, bbox_inches="tight")
plt.show()

# Final value distribution
plt.figure(figsize=(10, 6))
plt.hist(final_values, bins=50, edgecolor="none", alpha=0.75)
plt.axvline(metrics["median_final"],  linestyle="--", label="Median")
plt.axvline(metrics["percentile_5"],  linestyle="--", label="5th Percentile")
plt.axvline(metrics["percentile_95"], linestyle="--", label="95th Percentile")
plt.axvline(metrics["cvar_5"],        linestyle=":",  label="CVaR 5%")
plt.title("Distribution of Final Portfolio Values")
plt.xlabel("Final Portfolio Value ($)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("final_value_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Historical drawdown
plt.figure(figsize=(10, 6))
plt.plot(drawdown, color="crimson")
plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color="crimson")
plt.title("Historical Portfolio Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.tight_layout()
plt.savefig("historical_drawdown.png", dpi=300, bbox_inches="tight")
plt.show()

# Efficient frontier
fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(
    frontier_df["Volatility"],
    frontier_df["Return"],
    c=frontier_df["Sharpe Ratio"],
    cmap="viridis",
    alpha=0.5,
    s=10,
)

ax.scatter(
    max_sharpe_row["Volatility"],
    max_sharpe_row["Return"],
    marker="*", s=300, zorder=5, label="Max Sharpe",
)

ax.scatter(
    min_vol_row["Volatility"],
    min_vol_row["Return"],
    marker="*", s=300, zorder=5, label="Min Volatility",
)

plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")
ax.set_title("Efficient Frontier")
ax.set_xlabel("Annualized Volatility")
ax.set_ylabel("Annualized Return")
ax.legend()
plt.tight_layout()
plt.savefig("efficient_frontier.png", dpi=300, bbox_inches="tight")
plt.show()