"""
utils/05_utils.py

Data loading, return computation, and plotting helpers.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


# ---------------------------------------------------------------
# Return helpers
# ---------------------------------------------------------------

def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Log or simple returns from a price panel."""
    if method == "log":
        return np.log(prices / prices.shift(1))
    return prices.pct_change()


def generate_sample_data(
    n_assets: int = 6,
    n_obs: int = 1500,
    seed: int = 42,
    start: str = "2015-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synthetic price panel + macro predictors for testing.

    Returns
    -------
    prices : (n_obs x n_assets) DataFrame
    macro  : (n_obs x 4) DataFrame with fake macro predictors
             [yield_slope, vix_change, credit_spread, momentum]
    """
    rng = np.random.default_rng(seed)
    assets = [f"ASSET_{i:02d}" for i in range(n_assets)]
    dates  = pd.bdate_range(start=start, periods=n_obs)

    # Two-regime process: calm and stressed
    regime = np.zeros(n_obs, dtype=int)
    for t in range(1, n_obs):
        if regime[t-1] == 0:
            regime[t] = int(rng.random() < 0.02)   # 2% chance of entering stress
        else:
            regime[t] = int(rng.random() < 0.90)   # 90% chance of staying stressed

    mu_calm   = rng.uniform(0.0002, 0.0008, n_assets)
    mu_stress = rng.uniform(-0.001, 0.0001, n_assets)
    sig_calm  = rng.uniform(0.008, 0.015, n_assets)
    sig_stress = rng.uniform(0.018, 0.035, n_assets)

    mkt_calm   = rng.normal(0.0003, 0.008, n_obs)
    mkt_stress = rng.normal(-0.001, 0.020, n_obs)
    betas = rng.uniform(0.4, 1.4, n_assets)

    daily_rets = np.zeros((n_obs, n_assets))
    for t in range(n_obs):
        if regime[t] == 0:
            daily_rets[t] = mu_calm + betas * mkt_calm[t] + rng.normal(0, sig_calm)
        else:
            daily_rets[t] = mu_stress + betas * mkt_stress[t] + rng.normal(0, sig_stress)

    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(daily_rets, axis=0)),
        index=dates, columns=assets,
    )

    # Macro predictors: loosely correlated with regime
    yield_slope   = -0.5 * regime + rng.normal(0, 0.3, n_obs)
    vix_change    =  1.2 * regime + rng.normal(0, 0.5, n_obs)
    credit_spread =  0.8 * regime + rng.normal(0, 0.4, n_obs)
    momentum      = -0.6 * regime + rng.normal(0, 0.5, n_obs)

    macro = pd.DataFrame({
        "yield_slope":   yield_slope,
        "vix_change":    vix_change,
        "credit_spread": credit_spread,
        "momentum":      momentum,
        "regime":        regime,   # keep true regime for reference (not used in model)
    }, index=dates)

    return prices, macro


# ---------------------------------------------------------------
# Performance metrics (standalone, used by backtest module)
# ---------------------------------------------------------------

ANN = 252


def annualised_return(r: pd.Series) -> float:
    return float((1 + r).prod() ** (ANN / len(r)) - 1)


def annualised_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(ANN))


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    excess = r - rf / ANN
    return float(excess.mean() / excess.std() * np.sqrt(ANN)) if excess.std() > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def calmar(r: pd.Series) -> float:
    mdd = abs(max_drawdown(r))
    return annualised_return(r) / mdd if mdd > 0 else np.nan


# ---------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------

def plot_tearsheet(
    results: dict,
    title: str = "Strategy Tearsheet",
    save_path: str | None = None,
) -> None:
    """
    Multi-strategy tearsheet.
    results : dict of {label: pd.Series of daily returns}
    """
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(3, 1, hspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    colors = ["#1a5276", "#922b21", "#1e8449", "#6c3483", "#b9770e"]

    for (label, r), color in zip(results.items(), colors):
        eq = (1 + r).cumprod()
        dd = eq / eq.cummax() - 1
        roll_sh = r.rolling(63).apply(
            lambda x: x.mean() / x.std() * np.sqrt(ANN) if x.std() > 0 else np.nan,
            raw=True,
        )
        ax1.plot(eq.index, eq,    label=label, color=color, linewidth=1.3)
        ax2.plot(dd.index, dd,    label=label, color=color, linewidth=1.0, alpha=0.85)
        ax3.plot(roll_sh.index, roll_sh, label=label, color=color, linewidth=1.0)

    ax1.set_title("Cumulative Return",        fontsize=10, fontweight="bold")
    ax2.set_title("Drawdown",                 fontsize=10, fontweight="bold")
    ax3.set_title("Rolling 63-day Sharpe",    fontsize=10, fontweight="bold")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax3.axhline(0, color="black", linewidth=0.5, linestyle="--")

    for ax in [ax1, ax2, ax3]:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()
