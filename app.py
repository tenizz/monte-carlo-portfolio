import streamlit as st

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
    import numpy as np
    import pandas as pd
    import yfinance as yf

    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
    weights = np.array([float(weight.strip()) for weight in weights_input.split(",")])

    if len(tickers) != len(weights):
        st.error("Number of tickers must match number of weights.")
        st.stop()

    if not np.isclose(weights.sum(), 1):
        st.error("Portfolio weights must add up to 1.")
        st.stop()

    with st.spinner("Downloading historical market data..."):
        data = yf.download(tickers, start="2015-01-01")["Close"]
        daily_returns = data.pct_change().dropna()

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
        portfolio_results[:, :100]
    )

    st.write("### Monte Carlo Portfolio Paths")
    st.line_chart(simulation_chart_data)

    # Final value distribution
    final_values_table = pd.DataFrame({
        "Final Portfolio Value": final_values
    })

    st.write("### Distribution of Final Portfolio Values")
    st.bar_chart(
        final_values_table["Final Portfolio Value"].value_counts(bins=50).sort_index()
    )