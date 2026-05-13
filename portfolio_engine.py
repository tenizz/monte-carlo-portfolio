import numpy as np
import pandas as pd

CONSERVATIVE_ANNUAL_RETURN = 0.08


def prepare_mean_returns(daily_returns, tickers, mode, trading_days=252):
    """
    Return the mean_returns Series appropriate for the chosen simulation mode.

    Keeps mode-specific logic in the engine, not the UI layer.
    """
    if mode == "Conservative 8% return simulation":
        daily_rate = (1 + CONSERVATIVE_ANNUAL_RETURN) ** (1 / trading_days) - 1
        return pd.Series([daily_rate] * len(tickers), index=tickers)
    return daily_returns.mean()


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
    """
    Run a fully vectorized Monte Carlo portfolio simulation.

    Eliminates all Python-level loops by generating the entire
    (days × simulations) return matrix at once and computing
    portfolio paths using NumPy cumulative products.

    The monthly contribution math:
        Let G[t] = cumprod of daily growth factors up to day t.
        A contribution added at day k compounds to G[t]/G[k] by day t.
        So total portfolio value at day t is:
            V[t] = G[t] * (initial_investment
                           + monthly_contribution * Σ_{k≤t} 1/G[k])
        where the sum is over contribution days only.
        This is computed in one pass using np.cumsum.

    Returns
    -------
    portfolio_results : np.ndarray, shape (days, simulations)
    final_values      : np.ndarray, shape (simulations,)
    """
    days = years * trading_days
    rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # 1. Generate all returns at once: shape (days, simulations)
    # ------------------------------------------------------------------

    if mode == "Bootstrap historical simulation":
        # Sample day indices with replacement across all simulations.
        # Each column is one simulation; cross-asset correlations are
        # preserved within a day because we sample whole rows.
        idx = rng.integers(0, len(daily_returns), size=(days, simulations))
        returns_matrix = daily_returns.to_numpy()[idx]   # (days, sims, assets)
        portfolio_returns = returns_matrix @ weights      # (days, sims)

    else:
        # Draw from the joint multivariate normal for all days and
        # simulations in a single call.
        raw = rng.multivariate_normal(
            mean=mean_returns.to_numpy(),
            cov=cov_matrix.to_numpy(),
            size=(days, simulations)
        )                                                 # (days, sims, assets)
        portfolio_returns = raw @ weights                 # (days, sims)

    # ------------------------------------------------------------------
    # 2. Daily growth factors, clipped so portfolio never goes negative
    # ------------------------------------------------------------------

    growth_factors = np.maximum(0.0, 1.0 + portfolio_returns)  # (days, sims)

    # ------------------------------------------------------------------
    # 3. Cumulative compound growth: G[t] = prod of g[0..t]
    # ------------------------------------------------------------------

    G = np.cumprod(growth_factors, axis=0)               # (days, sims)

    # ------------------------------------------------------------------
    # 4. Monthly contributions (vectorized)
    #
    # Contribution days match the original logic: every 21 trading days,
    # starting at day 21 (not day 0), added *after* that day's growth.
    #
    # Build a sparse (days, sims) matrix that holds 1/G[k] at each
    # contribution day k, then cumsum along the day axis. At any day t,
    # cumsum_inv_G[t] = Σ_{k≤t, k is contrib day} 1/G[k].
    # ------------------------------------------------------------------

    if monthly_contribution > 0:
        contrib_days = np.arange(21, days, 21)           # [21, 42, 63, ...]

        inv_G_sparse = np.zeros((days, simulations))
        inv_G_sparse[contrib_days] = 1.0 / G[contrib_days]  # (days, sims)

        cumsum_inv_G = np.cumsum(inv_G_sparse, axis=0)   # (days, sims)

        portfolio_results = G * (
                initial_investment + monthly_contribution * cumsum_inv_G
        )
    else:
        portfolio_results = G * initial_investment

    final_values = portfolio_results[-1, :]

    return portfolio_results, final_values


def calculate_efficient_frontier(
        tickers,
        mean_returns,
        cov_matrix,
        risk_free_rate=0.04,
        trading_days=252,
        num_random_portfolios=1000
):
    """
    Generate random portfolios to approximate the efficient frontier.

    Vectorized: all random weight sets are sampled and annualized
    in one pass with no Python loop.
    """
    rng = np.random.default_rng()
    n = len(tickers)

    # Sample all random weight sets at once: shape (num_portfolios, n_assets)
    raw_weights = rng.random((num_random_portfolios, n))
    all_weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)

    # Daily returns for each portfolio: shape (num_portfolios,)
    daily_ret = all_weights @ mean_returns.to_numpy()
    annual_ret = (1 + daily_ret) ** trading_days - 1

    # Annualized volatility: sqrt(w' Σ w) * sqrt(252)
    # Efficient: compute w' Σ in one matrix multiply, then dot with w row-wise
    wS = all_weights @ cov_matrix.to_numpy()              # (portfolios, assets)
    variance = np.einsum("ij,ij->i", wS, all_weights)     # (portfolios,)
    annual_vol = np.sqrt(variance) * np.sqrt(trading_days)

    sharpe = (annual_ret - risk_free_rate) / annual_vol

    frontier_df = pd.DataFrame({
        "Volatility": annual_vol,
        "Return": annual_ret,
        "Sharpe Ratio": sharpe,
    })

    # Store per-ticker weights so callers can retrieve optimal allocations
    for i, ticker in enumerate(tickers):
        frontier_df[f"w_{ticker}"] = all_weights[:, i]

    return frontier_df


def calculate_simulation_metrics(
        final_values,
        initial_investment,
        monthly_contribution,
        years,
        inflation_rate=0.03
):
    """
    Compute summary statistics from final simulation values.

    Includes CVaR, inflation-adjusted median, probability metrics,
    and stress test — unified so both CLI and web app use the same engine.
    """
    mean_final     = float(np.mean(final_values))
    median_final   = float(np.median(final_values))
    percentile_5   = float(np.percentile(final_values, 5))
    percentile_95  = float(np.percentile(final_values, 95))

    # CVaR: expected value in the worst 5% of outcomes
    tail_mask = final_values <= percentile_5
    cvar_5 = float(np.mean(final_values[tail_mask]))

    # Inflation-adjusted median (real purchasing power)
    real_median = median_final / ((1 + inflation_rate) ** years)

    # Probability of loss metrics
    total_contributions = initial_investment + monthly_contribution * years * 12
    prob_loss            = float(np.mean(final_values < initial_investment))
    prob_below_contrib   = float(np.mean(final_values < total_contributions))

    # Stress test: immediate 30% crash applied to median
    stressed_median = median_final * 0.70

    summary_table = pd.DataFrame({
        "Metric": [
            "Mean Final Value",
            "Median Final Value",
            "Inflation-Adjusted Median",
            "5th Percentile (VaR 5%)",
            "CVaR 5%",
            "95th Percentile",
            "Probability Below Initial Investment",
            "Probability Below Total Contributions",
            "Median After 30% Crash (Stress Test)",
        ],
        "Value": [
            f"${mean_final:,.2f}",
            f"${median_final:,.2f}",
            f"${real_median:,.2f}",
            f"${percentile_5:,.2f}",
            f"${cvar_5:,.2f}",
            f"${percentile_95:,.2f}",
            f"{prob_loss:.2%}",
            f"{prob_below_contrib:.2%}",
            f"${stressed_median:,.2f}",
        ]
    })

    return {
        "mean_final":          mean_final,
        "median_final":        median_final,
        "real_median":         real_median,
        "percentile_5":        percentile_5,
        "percentile_95":       percentile_95,
        "cvar_5":              cvar_5,
        "prob_loss":           prob_loss,
        "prob_below_contrib":  prob_below_contrib,
        "stressed_median":     stressed_median,
        "total_contributions": total_contributions,
        "summary_table":       summary_table,
    }


# ------------------------------------------------------------------
# GARCH(1,1)
# ------------------------------------------------------------------

def fit_garch(portfolio_returns):
    """
    Fit a GARCH(1,1) model to a portfolio return series.

    GARCH(1,1) variance equation:
        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

        ω (omega)  — base variance; floor that volatility reverts to
        α (alpha)  — ARCH term; sensitivity to yesterday's shock
        β (beta)   — GARCH term; persistence of yesterday's volatility
        α + β      — total persistence; near 1 = long volatility memory

    The arch library requires percentage returns for numerical stability,
    so inputs are scaled ×100 before fitting and converted back after.

    Parameters
    ----------
    portfolio_returns : array-like
        Daily portfolio returns in decimal form (e.g. 0.012 for 1.2%).

    Returns
    -------
    dict with fitted parameters, diagnostics, and conditional volatility series.
    """
    from arch import arch_model

    # Scale to % for numerical stability in the optimizer
    r_pct = np.asarray(portfolio_returns) * 100

    model  = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="Normal")
    result = model.fit(disp="off")

    omega = float(result.params["omega"])       # in %² units
    alpha = float(result.params["alpha[1]"])
    beta  = float(result.params["beta[1]"])

    # arch changed the mean parameter name between versions: "Const" → "mu"
    mean_key = "mu" if "mu" in result.params.index else "Const"
    mu = float(result.params[mean_key])         # daily mean in % units

    persistence   = alpha + beta
    # Long-run (unconditional) variance, converted back to decimal
    long_run_vol_daily  = np.sqrt(omega / (1 - persistence)) / 100
    long_run_vol_annual = long_run_vol_daily * np.sqrt(252)

    # Last conditional variance in decimal² (seed for simulation)
    last_cond_vol_pct = float(result.conditional_volatility.iloc[-1])
    last_var_decimal  = (last_cond_vol_pct / 100) ** 2

    # Conditional volatility series annualised for charting
    cond_vol_annual = (result.conditional_volatility / 100) * np.sqrt(252)

    return {
        "omega":               omega,
        "alpha":               alpha,
        "beta":                beta,
        "mu":                  mu / 100,           # back to decimal
        "persistence":         persistence,
        "long_run_vol_annual": long_run_vol_annual,
        "last_var_decimal":    last_var_decimal,
        "cond_vol_annual":     pd.Series(
            cond_vol_annual.values,
            index=result.conditional_volatility.index
        ),
        "aic":                 result.aic,
        "bic":                 result.bic,
    }


def run_garch_simulation(
        garch_params,
        initial_investment,
        monthly_contribution,
        years,
        simulations,
        trading_days=252
):
    """
    Monte Carlo simulation using GARCH(1,1) time-varying volatility.

    Because σ²_t depends on r_{t-1} (previous day's actual return),
    the day axis must remain sequential — it cannot be collapsed into
    a single cumprod like the constant-vol engine.

    The outer loop runs over days (≤ 10,080 iterations for 40 years),
    but every operation inside is a NumPy ufunc over the full
    simulations vector, so wall-clock time stays acceptable.

    Random innovations z ~ N(0,1) are pre-generated in one shot as a
    (days, simulations) matrix before the loop starts.

    Parameters
    ----------
    garch_params : dict
        Output of fit_garch().
    initial_investment, monthly_contribution, years, simulations : numeric
    trading_days : int

    Returns
    -------
    portfolio_results : np.ndarray, shape (days, simulations)
    final_values      : np.ndarray, shape (simulations,)
    """
    days = years * trading_days
    rng  = np.random.default_rng()

    # GARCH parameters in decimal units
    omega = garch_params["omega"] / 10000       # ω was fitted on % returns → ÷ 10000
    alpha = garch_params["alpha"]
    beta  = garch_params["beta"]
    mu    = garch_params["mu"]                  # daily mean return in decimal

    # Pre-generate all standard-normal innovations: shape (days, simulations)
    z = rng.standard_normal((days, simulations))

    # Seed each simulation's variance with the last observed conditional variance
    sigma2 = np.full(simulations, garch_params["last_var_decimal"])

    portfolio_values  = np.full(simulations, float(initial_investment))
    portfolio_results = np.zeros((days, simulations))

    for t in range(days):
        # Daily return: r_t = μ + σ_t · z_t  (vectorized over all simulations)
        r = mu + np.sqrt(sigma2) * z[t]
        r = np.maximum(-1.0, r)                 # floor: portfolio can't go below 0

        portfolio_values = portfolio_values * (1.0 + r)

        if t % 21 == 0 and t != 0:
            portfolio_values += monthly_contribution

        portfolio_results[t] = portfolio_values

        # GARCH variance update: σ²_t+1 = ω + α·r²_t + β·σ²_t
        sigma2 = omega + alpha * r ** 2 + beta * sigma2

    return portfolio_results, portfolio_results[-1]