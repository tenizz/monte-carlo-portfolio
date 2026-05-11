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