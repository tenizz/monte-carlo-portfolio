import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from portfolio_engine import (
    prepare_mean_returns,
    run_monte_carlo_simulation,
    calculate_efficient_frontier,
    calculate_simulation_metrics,
    fit_garch,
    run_garch_simulation,
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
    tickers = list(tickers_tuple)
    raw = yf.download(tickers, start="2015-01-01", progress=False, auto_adjust=True)

    if raw.empty:
        return pd.DataFrame()

    # yfinance 1.x returns MultiIndex columns (field, ticker) for multi-ticker downloads
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw["Close"]
    else:
        # single ticker returns flat columns
        data = raw[["Close"]] if "Close" in raw.columns else raw

    if isinstance(data, pd.Series):
        data = data.to_frame()
        data.columns = tickers

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
# Run simulation — only when button pressed, store in session_state
# so results survive tab switches and download button clicks
# ------------------------------------------------------------------

if run_button:

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

    trading_days = 252
    mean_returns = prepare_mean_returns(daily_returns, tickers, mode, trading_days)
    cov_matrix   = daily_returns.cov()

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

    metrics = calculate_simulation_metrics(
        final_values=final_values,
        initial_investment=initial_investment,
        monthly_contribution=monthly_contribution,
        years=years,
    )

    with st.spinner("Calculating efficient frontier..."):
        frontier_df = calculate_efficient_frontier(
            tickers=tickers,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=0.04,
            trading_days=trading_days,
            num_random_portfolios=3000,
        )

    # ------------------------------------------------------------------
    # GARCH(1,1) — fit on historical portfolio returns, then simulate
    # ------------------------------------------------------------------

    portfolio_returns_hist = daily_returns @ weights

    with st.spinner("Fitting GARCH(1,1) model..."):
        garch_params = fit_garch(portfolio_returns_hist)

    with st.spinner("Running GARCH simulation..."):
        garch_results, garch_final_values = run_garch_simulation(
            garch_params=garch_params,
            initial_investment=initial_investment,
            monthly_contribution=monthly_contribution,
            years=years,
            simulations=simulations,
            trading_days=trading_days,
        )

    garch_metrics = calculate_simulation_metrics(
        final_values=garch_final_values,
        initial_investment=initial_investment,
        monthly_contribution=monthly_contribution,
        years=years,
    )

    # Build CSV bytes once now — not inside download buttons —
    # so clicking a button doesn't trigger recomputation
    summary_csv  = metrics["summary_table"].to_csv(index=False).encode("utf-8")
    frontier_csv = frontier_df.drop(
        columns=[c for c in frontier_df.columns if c.startswith("w_")]
    ).to_csv(index=False).encode("utf-8")
    values_csv   = pd.DataFrame(
        {"Final Portfolio Value": final_values}
    ).to_csv(index=False).encode("utf-8")

    # Store everything — persists across all reruns until next Run click
    st.session_state["results"] = {
        "tickers":             tickers,
        "mode":                mode,
        "years":               years,
        "simulations":         simulations,
        "data":                data,
        "daily_returns":       daily_returns,
        "portfolio_results":   portfolio_results,
        "final_values":        final_values,
        "metrics":             metrics,
        "frontier_df":         frontier_df,
        "garch_params":        garch_params,
        "garch_results":       garch_results,
        "garch_final_values":  garch_final_values,
        "garch_metrics":       garch_metrics,
        "summary_csv":         summary_csv,
        "frontier_csv":        frontier_csv,
        "values_csv":          values_csv,
    }

# ------------------------------------------------------------------
# Render UI — reads only from session_state
# Survives every rerun: tab clicks, download clicks, anything
# ------------------------------------------------------------------

if "results" not in st.session_state:
    st.info("Configure your portfolio in the sidebar and press **Run Simulation**.")
    st.stop()

r = st.session_state["results"]
m = r["metrics"]

st.success("Simulation complete.")

tab_results, tab_charts, tab_frontier, tab_garch, tab_data = st.tabs([
    "Results", "Charts", "Efficient Frontier", "GARCH", "Raw Data & Export"
])

# ── Tab 1: Results ───────────────────────────────────────────────
with tab_results:

    st.subheader("Simulation Summary")
    st.caption(
        f"Mode: {r['mode']} · {r['simulations']:,} simulations · {r['years']} years"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Median Final Value",    f"${m['median_final']:,.0f}")
        st.metric("Inflation-Adj. Median", f"${m['real_median']:,.0f}")
        st.metric("Mean Final Value",      f"${m['mean_final']:,.0f}")

    with col2:
        st.metric("95th Percentile", f"${m['percentile_95']:,.0f}")
        st.metric("5th Percentile",  f"${m['percentile_5']:,.0f}")
        st.metric("CVaR 5%",         f"${m['cvar_5']:,.0f}")

    with col3:
        st.metric("Prob. of Loss",              f"{m['prob_loss']:.1%}")
        st.metric("Prob. Below Total Invested", f"{m['prob_below_contrib']:.1%}")
        st.metric("Median After 30% Crash",     f"${m['stressed_median']:,.0f}")

    st.divider()
    st.caption(
        f"Total contributed over {r['years']} years: "
        f"${m['total_contributions']:,.0f}"
    )
    st.subheader("Full Summary Table")
    st.dataframe(m["summary_table"], use_container_width=True, hide_index=True)

# ── Tab 2: Charts ────────────────────────────────────────────────
with tab_charts:

    st.subheader("Historical Price Data")

    # Normalise to 100 so tickers with different prices are comparable
    price_data = r["data"][r["tickers"]]
    normalised = price_data / price_data.iloc[0] * 100

    fig_price = px.line(
        normalised,
        labels={"value": "Normalised Value (base 100)", "index": "Date"},
        title="Historical Price (Normalised to 100)",
    )
    fig_price.update_traces(line=dict(width=1.5))
    fig_price.update_layout(legend_title_text="Ticker", hovermode="x unified")
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Monte Carlo Simulation Paths")

    # Downsample: target ~500 points per path to keep the browser fast
    total_days = r["portfolio_results"].shape[0]
    step       = max(1, total_days // 500)
    n_paths    = 40
    paths      = r["portfolio_results"][::step, :n_paths]

    fig_mc = go.Figure()
    for i in range(n_paths):
        fig_mc.add_trace(go.Scatter(
            y=paths[:, i],
            mode="lines",
            line=dict(width=0.6),
            opacity=0.4,
            showlegend=False,
            hoverinfo="skip",
        ))

    # Overlay median path
    median_path = np.median(r["portfolio_results"][::step, :], axis=1)
    fig_mc.add_trace(go.Scatter(
        y=median_path,
        mode="lines",
        line=dict(width=2, color="white", dash="dash"),
        name="Median",
    ))

    fig_mc.update_layout(
        title=f"{n_paths} Simulation Paths (1-in-{step} days shown)",
        xaxis_title="Sampled Trading Days",
        yaxis_title="Portfolio Value ($)",
        hovermode="x",
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    st.subheader("Distribution of Final Portfolio Values")

    fv = r["final_values"]
    m  = r["metrics"]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=fv, nbinsx=60, name="Simulations", opacity=0.75))

    for value, label, color in [
        (m["median_final"],  "Median",         "white"),
        (m["percentile_5"],  "5th Percentile", "red"),
        (m["percentile_95"], "95th Percentile","green"),
        (m["cvar_5"],        "CVaR 5%",        "orange"),
    ]:
        fig_hist.add_vline(
            x=value,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="top",
        )

    fig_hist.update_layout(
        title="Distribution of Final Portfolio Values",
        xaxis_title="Final Portfolio Value ($)",
        yaxis_title="Frequency",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Tab 3: Efficient Frontier ─────────────────────────────────────
with tab_frontier:

    st.subheader("Efficient Frontier")
    st.caption("Each point is a randomly sampled portfolio. Color = Sharpe ratio.")

    fdf     = r["frontier_df"]
    best    = fdf.loc[fdf["Sharpe Ratio"].idxmax()]
    low_vol = fdf.loc[fdf["Volatility"].idxmin()]

    fig_ef = go.Figure()

    fig_ef.add_trace(go.Scatter(
        x=fdf["Volatility"],
        y=fdf["Return"],
        mode="markers",
        marker=dict(
            color=fdf["Sharpe Ratio"],
            colorscale="Viridis",
            size=4,
            opacity=0.6,
            colorbar=dict(title="Sharpe Ratio"),
        ),
        name="Portfolios",
        hovertemplate=(
            "Return: %{y:.2%}<br>"
            "Volatility: %{x:.2%}<extra></extra>"
        ),
    ))

    for row, label, color in [
        (best,    "Max Sharpe",    "gold"),
        (low_vol, "Min Volatility","cyan"),
    ]:
        fig_ef.add_trace(go.Scatter(
            x=[row["Volatility"]],
            y=[row["Return"]],
            mode="markers",
            marker=dict(symbol="star", size=18, color=color),
            name=label,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Return: %{y:.2%}<br>"
                "Volatility: %{x:.2%}<extra></extra>"
            ),
        ))

    fig_ef.update_layout(
        title="Efficient Frontier",
        xaxis_title="Annualised Volatility",
        yaxis_title="Annualised Return",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Max Sharpe — Return",     f"{best['Return']:.2%}")
        st.metric("Max Sharpe — Volatility", f"{best['Volatility']:.2%}")
        st.metric("Max Sharpe Ratio",        f"{best['Sharpe Ratio']:.2f}")
    with col_b:
        st.metric("Min Vol — Return",     f"{low_vol['Return']:.2%}")
        st.metric("Min Vol — Volatility", f"{low_vol['Volatility']:.2%}")
        st.metric("Min Vol Sharpe Ratio", f"{low_vol['Sharpe Ratio']:.2f}")

# ── Tab 4: GARCH ─────────────────────────────────────────────────
with tab_garch:

    gp = r["garch_params"]
    gm = r["garch_metrics"]

    st.subheader("GARCH(1,1) Model")
    st.caption(
        "Volatility is modelled as time-varying: large moves tend to "
        "cluster together. GARCH captures this; constant-vol simulation does not."
    )

    # ── Parameters ──────────────────────────────────────────────────
    st.markdown("#### Fitted Parameters")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ω (omega)",   f"{gp['omega']:.6f}",
                  help="Base variance — the floor volatility reverts to")
        st.metric("AIC", f"{gp['aic']:.1f}")
    with col2:
        st.metric("α (alpha)",   f"{gp['alpha']:.4f}",
                  help="ARCH term — sensitivity to yesterday's shock")
        st.metric("BIC", f"{gp['bic']:.1f}")
    with col3:
        st.metric("β (beta)",    f"{gp['beta']:.4f}",
                  help="GARCH term — persistence of yesterday's volatility")
    with col4:
        st.metric("α + β (persistence)", f"{gp['persistence']:.4f}",
                  help="Near 1 = long volatility memory; above 1 = non-stationary")
        st.metric("Long-run Annual Vol",  f"{gp['long_run_vol_annual']:.2%}",
                  help="Unconditional volatility the model reverts to")

    st.divider()

    # ── Historical conditional volatility ───────────────────────────
    st.markdown("#### Historical Conditional Volatility (Annualised)")
    st.caption(
        "Spikes show volatility clustering — periods of turbulence followed "
        "by more turbulence. A constant-vol model misses this entirely."
    )

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=gp["cond_vol_annual"].index,
        y=gp["cond_vol_annual"].values,
        mode="lines",
        line=dict(width=1, color="#00b4d8"),
        name="Conditional Volatility",
        fill="tozeroy",
        fillcolor="rgba(0,180,216,0.15)",
    ))
    fig_vol.add_hline(
        y=gp["long_run_vol_annual"],
        line_dash="dash",
        line_color="orange",
        annotation_text="Long-run vol",
        annotation_position="bottom right",
    )
    fig_vol.update_layout(
        xaxis_title="Date",
        yaxis_title="Annualised Volatility",
        yaxis_tickformat=".0%",
        hovermode="x unified",
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    st.divider()

    # ── GARCH vs Standard simulation comparison ──────────────────────
    st.markdown("#### GARCH vs Standard Simulation")

    # Downsampled paths for both
    total_days = r["garch_results"].shape[0]
    step    = max(1, total_days // 500)
    n_paths = 30

    fig_compare = go.Figure()

    # Standard paths
    std_paths = r["portfolio_results"][::step, :n_paths]
    for i in range(n_paths):
        fig_compare.add_trace(go.Scatter(
            y=std_paths[:, i],
            mode="lines",
            line=dict(width=0.5, color="#4cc9f0"),
            opacity=0.3,
            showlegend=(i == 0),
            name="Standard (constant vol)",
            hoverinfo="skip",
            legendgroup="standard",
        ))

    # GARCH paths
    garch_paths = r["garch_results"][::step, :n_paths]
    for i in range(n_paths):
        fig_compare.add_trace(go.Scatter(
            y=garch_paths[:, i],
            mode="lines",
            line=dict(width=0.5, color="#f72585"),
            opacity=0.3,
            showlegend=(i == 0),
            name="GARCH (time-varying vol)",
            hoverinfo="skip",
            legendgroup="garch",
        ))

    # Median overlays
    std_median   = np.median(r["portfolio_results"][::step, :], axis=1)
    garch_median = np.median(r["garch_results"][::step, :],     axis=1)

    fig_compare.add_trace(go.Scatter(
        y=std_median, mode="lines",
        line=dict(width=2.5, color="#4cc9f0", dash="dash"),
        name="Standard median", legendgroup="standard",
    ))
    fig_compare.add_trace(go.Scatter(
        y=garch_median, mode="lines",
        line=dict(width=2.5, color="#f72585", dash="dash"),
        name="GARCH median", legendgroup="garch",
    ))

    fig_compare.update_layout(
        title="Simulation Paths: Standard vs GARCH",
        xaxis_title="Sampled Trading Days",
        yaxis_title="Portfolio Value ($)",
        hovermode="x",
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # ── Distribution comparison ──────────────────────────────────────
    st.markdown("#### Final Value Distribution: Standard vs GARCH")
    st.caption(
        "GARCH typically produces fatter tails — more extreme outcomes "
        "in both directions — because volatility can spike suddenly."
    )

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=r["final_values"],
        nbinsx=60, opacity=0.6,
        name="Standard", marker_color="#4cc9f0",
    ))
    fig_dist.add_trace(go.Histogram(
        x=r["garch_final_values"],
        nbinsx=60, opacity=0.6,
        name="GARCH", marker_color="#f72585",
    ))
    fig_dist.update_layout(
        barmode="overlay",
        xaxis_title="Final Portfolio Value ($)",
        yaxis_title="Frequency",
        title="Final Value Distributions",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── Side-by-side metrics ─────────────────────────────────────────
    st.markdown("#### Key Metrics: Standard vs GARCH")

    col_std, col_g = st.columns(2)
    std_m = r["metrics"]

    with col_std:
        st.markdown("**Standard Simulation**")
        st.metric("Median Final Value", f"${std_m['median_final']:,.0f}")
        st.metric("CVaR 5%",            f"${std_m['cvar_5']:,.0f}")
        st.metric("Prob. of Loss",       f"{std_m['prob_loss']:.1%}")
        st.metric("95th Percentile",     f"${std_m['percentile_95']:,.0f}")

    with col_g:
        st.markdown("**GARCH Simulation**")
        st.metric("Median Final Value", f"${gm['median_final']:,.0f}",
                  delta=f"{gm['median_final'] - std_m['median_final']:+,.0f}")
        st.metric("CVaR 5%",            f"${gm['cvar_5']:,.0f}",
                  delta=f"{gm['cvar_5'] - std_m['cvar_5']:+,.0f}")
        st.metric("Prob. of Loss",       f"{gm['prob_loss']:.1%}")
        st.metric("95th Percentile",     f"${gm['percentile_95']:,.0f}",
                  delta=f"{gm['percentile_95'] - std_m['percentile_95']:+,.0f}")

# ── Tab 5: Raw Data & Export ──────────────────────────────────────
with tab_data:

    st.subheader("Daily Returns Preview")
    st.dataframe(r["daily_returns"].tail(10), use_container_width=True)

    st.subheader("Download Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="📥 Simulation Summary",
            data=r["summary_csv"],
            file_name="simulation_summary.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            label="📥 Efficient Frontier",
            data=r["frontier_csv"],
            file_name="efficient_frontier.csv",
            mime="text/csv",
        )
    with col3:
        st.download_button(
            label="📥 Final Portfolio Values",
            data=r["values_csv"],
            file_name="final_values.csv",
            mime="text/csv",
        )