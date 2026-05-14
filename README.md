# Monte Carlo Portfolio Simulator

A quantitative portfolio analysis tool built in Python that combines Monte Carlo simulation, GARCH volatility modelling, and modern portfolio theory to estimate future portfolio outcomes, measure downside risk, and optimise asset allocation.

**[Live Demo →](https://monte-carlo-portfolio.streamlit.app/)** *https://monte-carlo-portfolio.streamlit.app/*

---

## Overview

This project started as a command-line simulation script and evolved into a deployed, interactive web application. It is structured around a modular engine that separates all mathematical logic from the UI layer, making it straightforward to extend with new models.

The simulator supports multiple stocks, customisable portfolio weights, monthly contributions, and three distinct return assumption modes. It produces risk metrics, volatility analysis, and portfolio optimisation in a single run.

---

## Features

### Monte Carlo Simulation
- Three simulation modes:
    - **Historical normal** — fits a multivariate normal distribution to historical returns using the full covariance matrix to preserve cross-asset correlations
    - **Conservative** — overrides the mean with a fixed 8% annual return assumption while retaining historical covariance structure
    - **Bootstrap** — samples historical return vectors with replacement, preserving the empirical return distribution including fat tails
- Monthly contribution support with mathematically exact compounding (vectorised using `np.cumprod` and `np.cumsum`)
- Configurable investment horizon (1–40 years) and simulation count (1,000–20,000)

### GARCH(1,1) Volatility Modelling
- Fits a GARCH(1,1) model to historical portfolio returns via Maximum Likelihood Estimation using the `arch` library
- Parameters: ω (base variance), α (ARCH term / shock sensitivity), β (GARCH term / volatility persistence)
- Runs a full Monte Carlo simulation under time-varying volatility, with the day loop vectorised across all simulations simultaneously
- Compares GARCH vs standard simulation: final value distributions, simulation paths, and key risk metrics side by side
- Displays historical conditional volatility to visualise volatility clustering

### Risk Metrics
- **VaR** (Value at Risk) at the 5th percentile
- **CVaR** (Conditional Value at Risk / Expected Shortfall) — expected loss in the worst 5% of outcomes
- **Sharpe ratio** — risk-adjusted return relative to a 4% risk-free rate
- **Maximum drawdown** — largest peak-to-trough decline in historical portfolio value
- **CAPM beta** — portfolio sensitivity vs SPY benchmark
- **Inflation-adjusted median** — real purchasing power of the median outcome at 3% annual inflation
- **Stress test** — median portfolio value after an immediate 30% market crash
- **Probability of loss** — percentage of simulations ending below initial investment
- **Probability below total contributions** — percentage of simulations that fail to recover invested capital

### Portfolio Optimisation — Efficient Frontier
- Generates 3,000+ random portfolio weight combinations (fully vectorised with `np.einsum`)
- Identifies the **Maximum Sharpe Ratio** portfolio and **Minimum Volatility** portfolio
- Interactive scatter plot coloured by Sharpe ratio

### Engineering
- **Vectorised engine** — eliminates Python loops; all simulation paths computed via NumPy matrix operations
- **Modular architecture** — `portfolio_engine.py` contains all mathematical logic; `app.py` is UI only
- **Session state persistence** — Streamlit results survive tab switches and download button clicks
- **Downsampled Plotly charts** — simulation paths downsampled to ~500 points per path for fast browser rendering
- **CLI version** — `main.py` runs the full simulation from the terminal with Matplotlib outputs

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3 |
| Numerical computing | NumPy, SciPy |
| Data manipulation | pandas |
| Market data | yFinance |
| Volatility modelling | arch |
| Visualisation | Plotly, Matplotlib |
| Web application | Streamlit |
| Deployment | Streamlit Cloud |
| Version control | Git / GitHub |

---

## Project Structure

```
├── portfolio_engine.py   # All simulation and analytics logic
│   ├── prepare_mean_returns()
│   ├── run_monte_carlo_simulation()
│   ├── calculate_efficient_frontier()
│   ├── calculate_simulation_metrics()
│   ├── fit_garch()
│   └── run_garch_simulation()
│
├── app.py                # Streamlit web application (UI only)
├── main.py               # Command-line interface with Matplotlib charts
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/monte-carlo-portfolio
cd monte-carlo-portfolio
pip install -r requirements.txt
```

**Run the web app locally:**
```bash
streamlit run app.py
```

**Run the CLI version:**
```bash
python main.py
```

---

## Example Usage

```
Enter stock tickers: AAPL,MSFT,SPY
Enter portfolio weights: 0.3,0.3,0.4
Enter initial investment: 10000
Enter monthly contribution: 500
Enter years to simulate: 20
Enter number of simulations: 5000
Choose mode (1, 2, or 3): 1
```

**Example output (illustrative):**

| Metric | Value |
|---|---|
| Median Final Value | $187,432 |
| Inflation-Adjusted Median | $103,891 |
| CVaR 5% | $41,205 |
| Sharpe Ratio | 0.84 |
| Max Drawdown | -34.2% |
| CAPM Beta vs SPY | 1.12 |

---

## Key Concepts

**Monte Carlo Simulation** estimates future outcomes by running thousands of randomised scenarios based on historical return distributions. Each simulation represents one possible future portfolio path.

**GARCH(1,1)** (Generalised Autoregressive Conditional Heteroskedasticity) models volatility as time-varying rather than constant. The variance at time *t* is:

σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁

This captures volatility clustering — periods of high volatility tend to be followed by more high volatility — which standard normal simulations miss entirely.

**CVaR** (Expected Shortfall) measures the expected portfolio value in the worst 5% of simulated outcomes. It is a more informative downside risk measure than VaR because it captures the severity of tail losses, not just their threshold.

**Efficient Frontier** maps the trade-off between return and volatility across all possible portfolio weight combinations, identifying the allocation with the best risk-adjusted return (Maximum Sharpe) and the least volatile allocation.

---

## Disclaimer

This project is for educational and analytical purposes only. It does not constitute financial advice.