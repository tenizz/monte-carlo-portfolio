import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from portfolio_engine import (
    prepare_mean_returns,
    run_monte_carlo_simulation,
    calculate_efficient_frontier,
    calculate_simulation_metrics,
)

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Monte Carlo Portfolio Simulator",
    layout="wide"
)

st.title("Monte Carlo Portfolio Simulator")
st.caption("Analyze portfolio risk, return, downside scenarios, and future outcomes.")


# ------------------------------------------------------------------
# Cached data download
# ------------------------------------------------------------------

@st.cache_data
def download_price_data(tickers_tuple):
    # Accepts a tuple so it is hashable for st.cache_data
    tickers = list(tickers_tuple)
    data = yf.download(tickers, start="2015-01-01")["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.ffill().dropna()
    return data


# ------------------------------------------------------------------
# Sidebar inputs
# ------------------------------------------------------------------

st.sidebar.header("Portfolio Inputs")

tickers_input = st.sidebar.text_input(
    "Stock tickers (comma-separated)",
    value="AAPL,MSFT,SPY"
)

weights_input = st.sidebar.text_input(
    "Portfolio weights (must sum to 1)",
    value="0.3,0.3,0.4"
)

initial_investment = st.sidebar.number_input(
    "Initial investment ($)",
    min_value=0.0,
    value=10000.0,
    step=500.0
)

monthly_contribution = st.sidebar.number_input(
    "Monthly contribution ($)",
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
        "Bootstrap historical simulation",
    ]
)

run_button = st.sidebar.button("Run Simulation", type="primary")

# ------------------------------------------------------------------
# Main logic — only runs when button is pressed
# ------------------------------------------------------------------

if run_button:

    # --- Parse and validate inputs ---
    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    try:
        weights = np.array([float(w.strip()) for w in weights_input.split(",")])
    except ValueError:
        st.error("Weights must be numbers separated by commas.")
        st.stop()

    if len(tickers) != len(weights):
        st.error("Number of tickers must match number of weights.")
        st.stop()

    if not np.isclose(weights.sum(), 1):
        st.error(f"Weights must sum to 1. Current sum: {weights.sum():.4f}")
        st.stop()

    # --- Download data ---
    with st.spinner("Downloading historical market data..."):
        data = download_price_data(tuple(tickers))

    if data.empty:
        st.error("No price data found. Check your ticker symbols.")
        st.stop()

    missing = [t for t in tickers if t not in data.columns]
    if missing:
        st.error(f"No data found for: {', '.join(missing)}")
        st.stop()

    daily_returns = data[tickers].pct_change().dropna()

    if daily_returns.empty:
        st.error("Not enough historical data to calculate returns.")
        st.stop()

    # --- Prepare simulation inputs (mode logic lives in engine) ---
    trading_days = 252
    mean_returns = prepare_mean_returns(daily_returns, tickers, mode, trading_days)
    cov_matrix   = daily_returns.cov()

    # --- Run simulation ---
    with st.spinner(f"Running {simulations:,} simulations..."):
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

    # --- Compute metrics (full set, unified with CLI) ---
    metrics = calculate_simulation_metrics(
        final_values=final_values,
        initial_investment=initial_investment,
        monthly_contribution=monthly_contribution,
        years=years,
    )

    # --- Efficient frontier ---
    with st.spinner("Calculating efficient frontier..."):
        frontier_df = calculate_efficient_frontier(
            tickers=tickers,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=0.04,
            trading_days=trading_days,
            num_random_portfolios=3000,
        )

    st.success("Simulation complete.")

    # ------------------------------------------------------------------
    # Results — organised into tabs
    # ------------------------------------------------------------------

    tab_results, tab_charts, tab_frontier, tab_data = st.tabs([
        "Results", "Charts", "Efficient Frontier", "Raw Data & Export"
    ])

    # ── Tab 1: Results ──────────────────────────────────────────────
    with tab_results:

        st.subheader("Simulation Summary")
        st.caption(f"Mode: {mode} · {simulations:,} simulations · {years} years")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Median Final Value",   f"${metrics['median_final']:,.0f}")
            st.metric("Inflation-Adj. Median", f"${metrics['real_median']:,.0f}")
            st.metric("Mean Final Value",      f"${metrics['mean_final']:,.0f}")

        with col2:
            st.metric("95th Percentile",  f"${metrics['percentile_95']:,.0f}")
            st.metric("5th Percentile",   f"${metrics['percentile_5']:,.0f}")
            st.metric("CVaR 5%",          f"${metrics['cvar_5']:,.0f}")

        with col3:
            st.metric("Prob. of Loss",               f"{metrics['prob_loss']:.1%}")
            st.metric("Prob. Below Total Invested",  f"{metrics['prob_below_contrib']:.1%}")
            st.metric("Median After 30% Crash",      f"${metrics['stressed_median']:,.0f}")

        st.divider()
        st.caption(
            f"Total amount contributed over {years} years: "
            f"${metrics['total_contributions']:,.0f}"
        )

        st.subheader("Full Summary Table")
        st.dataframe(metrics["summary_table"], use_container_width=True, hide_index=True)

    # ── Tab 2: Charts ───────────────────────────────────────────────
    with tab_charts:

        st.subheader("Historical Price Data")
        st.line_chart(data[tickers])

        st.subheader("Monte Carlo Simulation Paths (first 50 shown)")
        st.line_chart(
            pd.DataFrame(
                portfolio_results[:, :50],
                columns=[f"Sim {i+1}" for i in range(50)]
            )
        )

        st.subheader("Distribution of Final Portfolio Values")
        st.dataframe(
            pd.DataFrame({"Final Portfolio Value ($)": final_values}).describe(),
            use_container_width=True
        )

    # ── Tab 3: Efficient Frontier ───────────────────────────────────
    with tab_frontier:

        st.subheader("Efficient Frontier")
        st.caption("Each point is a randomly sampled portfolio. Color = Sharpe ratio.")

        st.scatter_chart(
            frontier_df,
            x="Volatility",
            y="Return",
            color="Sharpe Ratio",
        )

        best = frontier_df.loc[frontier_df["Sharpe Ratio"].idxmax()]
        low_vol = frontier_df.loc[frontier_df["Volatility"].idxmin()]

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Max Sharpe — Return",     f"{best['Return']:.2%}")
            st.metric("Max Sharpe — Volatility", f"{best['Volatility']:.2%}")
            st.metric("Max Sharpe Ratio",        f"{best['Sharpe Ratio']:.2f}")
        with col_b:
            st.metric("Min Vol — Return",     f"{low_vol['Return']:.2%}")
            st.metric("Min Vol — Volatility", f"{low_vol['Volatility']:.2%}")
            st.metric("Min Vol Sharpe Ratio", f"{low_vol['Sharpe Ratio']:.2f}")

    # ── Tab 4: Raw Data & Export ────────────────────────────────────
    with tab_data:

        st.subheader("Daily Returns Preview")
        st.dataframe(daily_returns.tail(10), use_container_width=True)

        st.subheader("Download Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                label="📥 Simulation Summary",
                data=metrics["summary_table"].to_csv(index=False).encode("utf-8"),
                file_name="simulation_summary.csv",
                mime="text/csv",
            )

        with col2:
            st.download_button(
                label="📥 Efficient Frontier",
                data=frontier_df.to_csv(index=False).encode("utf-8"),
                file_name="efficient_frontier.csv",
                mime="text/csv",
            )

        with col3:
            st.download_button(
                label="📥 Final Portfolio Values",
                data=pd.DataFrame({"Final Portfolio Value": final_values})
                .to_csv(index=False)
                .encode("utf-8"),
                file_name="final_values.csv",
                mime="text/csv",
                )