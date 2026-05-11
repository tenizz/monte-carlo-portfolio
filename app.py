import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
@st.cache_data
def download_price_data(tickers):
    data = yf.download(tickers, start="2015-01-01")["Close"]
    return data

st.set_page_config(
    page_title="Monte Carlo Portfolio Simulator",
    layout="wide"
)

st.title("Monte Carlo Portfolio Simulator")
st.write("Analyze portfolio risk, return, downside scenarios, and future outcomes.")

# Sidebar inputs
st.sidebar.header("Portfolio Inputs")

tickers_input = st.sidebar.text_input(
    "Stock tickers",
    value="AAPL,MSFT,SPY"
)

weights_input = st.sidebar.text_input(
    "Portfolio weights",
    value="0.3,0.3,0.4"
)

initial_investment = st.sidebar.number_input(
    "Initial investment",
    min_value=0.0,
    value=10000.0,
    step=500.0
)

monthly_contribution = st.sidebar.number_input(
    "Monthly contribution",
    min_value=0.0,
    value=300.0,
    step=50.0
)

years = st.sidebar.slider(
    "Years to simulate",
    min_value=1,
    max_value=40,
    value=10
)

simulations = st.sidebar.slider(
    "Number of simulations",
    min_value=1000,
    max_value=20000,
    value=5000,
    step=1000
)

mode = st.sidebar.selectbox(
    "Simulation mode",
    [
        "Historical normal simulation",
        "Conservative 8% return simulation",
        "Bootstrap historical simulation"
    ]
)

run_button = st.sidebar.button("Run Simulation")

if run_button:
    st.success("Inputs captured successfully.")

    st.write("### Selected Portfolio")
    st.write(f"Tickers: {tickers_input}")
    st.write(f"Weights: {weights_input}")
    st.write(f"Initial investment: ${initial_investment:,.2f}")
    st.write(f"Monthly contribution: ${monthly_contribution:,.2f}")
    st.write(f"Years: {years}")
    st.write(f"Simulations: {simulations}")
    st.write(f"Mode: {mode}")

    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
    weights = np.array([float(weight.strip()) for weight in weights_input.split(",")])

    if len(tickers) != len(weights):
        st.error("Number of tickers must match number of weights.")
        st.stop()

    if not np.isclose(weights.sum(), 1):
        st.error("Portfolio weights must add up to 1.")
        st.stop()

    with st.spinner("Downloading historical market data..."):
        data = download_price_data(tickers)

    if data.empty:
        st.error("No price data found. Check your ticker symbols.")
        st.stop()

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.dropna()

    missing_tickers = [ticker for ticker in tickers if ticker not in data.columns]

    if missing_tickers:
        st.error(f"Missing data for: {', '.join(missing_tickers)}")
        st.stop()

    daily_returns = data.pct_change().dropna()

    if daily_returns.empty:
        st.error("Not enough historical data to calculate returns.")
        st.stop()

    st.success("Market data downloaded successfully.")

    st.write("### Historical Price Data")
    st.line_chart(data)

    st.write("### Daily Returns Preview")
    st.dataframe(daily_returns.tail())
    # Monte Carlo Engine
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    trading_days = 252

    if mode == "Conservative 8% return simulation":
        conservative_annual_return = 0.08

        conservative_daily_return = (
                                            (1 + conservative_annual_return) ** (1 / trading_days)
                                    ) - 1

        mean_returns = pd.Series(
            [conservative_daily_return] * len(tickers),
            index=tickers
        )

    days = years * trading_days

    portfolio_results = np.zeros((days, simulations))

    for sim in range(simulations):

        if mode == "Bootstrap historical simulation":

            sampled_returns = daily_returns.sample(
                n=days,
                replace=True,
                ignore_index=True
            )

            portfolio_daily_returns = (
                    sampled_returns.to_numpy() @ weights
            )

        else:

            simulated_returns = np.random.multivariate_normal(
                mean_returns,
                cov_matrix,
                days
            )

            portfolio_daily_returns = (
                    simulated_returns @ weights
            )

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

    median_final = float(np.median(final_values))
    mean_final = float(np.mean(final_values))
    percentile_5 = float(np.percentile(final_values, 5))
    percentile_95 = float(np.percentile(final_values, 95))
    summary_table = pd.DataFrame({
        "Metric": [
            "Mean Final Value",
            "Median Final Value",
            "5th Percentile",
            "95th Percentile"
        ],
        "Value": [
            f"${mean_final:,.2f}",
            f"${median_final:,.2f}",
            f"${percentile_5:,.2f}",
            f"${percentile_95:,.2f}"
        ]
    })

    st.write("## Simulation Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Median Final Value", f"${median_final:,.2f}")
        st.metric("5th Percentile", f"${percentile_5:,.2f}")

    with col2:
        st.metric("Mean Final Value", f"${mean_final:,.2f}")
        st.metric("95th Percentile", f"${percentile_95:,.2f}")

        st.write("## Simulation Charts")

    # Monte Carlo simulation paths
    simulation_chart_data = pd.DataFrame(
        portfolio_results[:, :30]
    )

    st.write("### Monte Carlo Portfolio Paths")
    st.line_chart(simulation_chart_data)

    # Final value distribution
    final_values_table = pd.DataFrame({
        "Final Portfolio Value": final_values
    })

    st.write("### Distribution of Final Portfolio Values")
    st.write(final_values_table.describe())
    st.write("## Efficient Frontier")

    num_random_portfolios = 1000

    frontier_returns = []
    frontier_volatility = []
    frontier_sharpe = []

    for _ in range(num_random_portfolios):

        random_weights = np.random.random(len(tickers))
        random_weights /= np.sum(random_weights)

        random_daily_return = np.sum(
            mean_returns * random_weights
        )

        random_annual_return = (
                                       (1 + random_daily_return) ** trading_days
                               ) - 1

        random_annual_volatility = (
                np.sqrt(
                    random_weights.T @ cov_matrix @ random_weights
                ) * np.sqrt(trading_days)
        )

        random_sharpe = (
                (random_annual_return - 0.04)
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

    st.write("### Efficient Frontier Scatter Plot")

    st.scatter_chart(
        frontier_df,
        x="Volatility",
        y="Return",
        color="Sharpe Ratio"
    )

st.write("## Download Results")

summary_csv = summary_table.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Simulation Summary CSV",
    data=summary_csv,
    file_name="simulation_summary.csv",
    mime="text/csv"
)

frontier_csv = frontier_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Efficient Frontier CSV",
    data=frontier_csv,
    file_name="efficient_frontier.csv",
    mime="text/csv"
)

final_values_csv = final_values_table.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Final Values CSV",
    data=final_values_csv,
    file_name="final_values.csv",
    mime="text/csv"
)