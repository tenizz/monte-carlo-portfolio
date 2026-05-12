import numpy as np
import pandas as pd

# Monte Carlo function

def run_monte_carlo_simulation(
        daily_returns,
        mean_returns,
        cov_matrix,
        weights,
        initial_investment,
        monthly_contribution,
        years,
        simulations,
        mode,
        trading_days=252
):
    days = years * trading_days
    portfolio_results = np.zeros((days, simulations))

    for sim in range(simulations):

        if mode == "Bootstrap historical simulation":
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
            daily_growth = max(0, daily_growth)

            portfolio_value *= daily_growth

            if day % 21 == 0 and day != 0:
                portfolio_value += monthly_contribution

            values.append(portfolio_value)

        portfolio_results[:, sim] = values

    final_values = portfolio_results[-1, :]

    return portfolio_results, final_values

# Efficient Frontier
def calculate_efficient_frontier(
        tickers,
        mean_returns,
        cov_matrix,
        risk_free_rate=0.04,
        trading_days=252,
        num_random_portfolios=1000
):
    frontier_returns = []
    frontier_volatility = []
    frontier_sharpe = []

    for _ in range(num_random_portfolios):
        random_weights = np.random.random(len(tickers))
        random_weights /= np.sum(random_weights)

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

    frontier_df = pd.DataFrame({
        "Volatility": frontier_volatility,
        "Return": frontier_returns,
        "Sharpe Ratio": frontier_sharpe
    })

    return frontier_df