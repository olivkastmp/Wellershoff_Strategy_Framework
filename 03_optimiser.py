"""
optimisation/03_optimiser.py

Portfolio optimisers: mean-variance, minimum volatility, risk parity.

These are called by the backtest engine to translate raw signal weights
into optimised weights at each rebalancing date.

The optimiser is optional — strategies can also use fixed/rule-based weights
without going through optimisation. To skip optimisation, set
optimiser=None in the backtest engine.

All optimisers use scipy.optimize.minimize with L-BFGS-B and enforce:
  - long-only (no shorts)
  - fully invested (weights sum to 1)
  - optional per-asset min/max constraints
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class Optimiser:
    """
    Portfolio optimiser with pluggable objective.

    Parameters
    ----------
    method         : "min_vol" | "mean_variance" | "risk_parity"
    lookback       : rolling window (days) for estimating cov/returns (default 126)
    risk_aversion  : lambda for mean-variance (higher = more risk averse, default 1.0)
    min_weight     : lower bound per asset (default 0.0 = long only)
    max_weight     : upper bound per asset (default 1.0)
    """

    def __init__(
        self,
        method: str = "min_vol",
        lookback: int = 126,
        risk_aversion: float = 1.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> None:
        if method not in ("min_vol", "mean_variance", "risk_parity"):
            raise ValueError(f"Unknown method: '{method}'")
        self.method        = method
        self.lookback      = lookback
        self.risk_aversion = risk_aversion
        self.min_weight    = min_weight
        self.max_weight    = max_weight

    def optimise(
        self,
        returns: pd.DataFrame,
        signal_weights: pd.Series | None = None,
    ) -> pd.Series:
        """
        Compute optimal weights given historical returns.

        Parameters
        ----------
        returns        : historical return DataFrame (rows=dates, cols=assets)
        signal_weights : optional initial guess / constraint from strategy signal
                         if provided, assets with zero signal weight are excluded

        Returns
        -------
        pd.Series : portfolio weights
        """
        # restrict to assets with non-zero signal weight if provided
        if signal_weights is not None:
            active = signal_weights[signal_weights > 0].index
            returns = returns[active]

        n = returns.shape[1]
        if n == 0:
            return pd.Series(dtype=float)

        # use lookback window
        r = returns.iloc[-self.lookback:]
        mu  = r.mean().values * 252
        cov = r.cov().values  * 252

        w0     = np.ones(n) / n
        bounds = [(self.min_weight, self.max_weight)] * n
        cons   = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

        if self.method == "min_vol":
            def objective(w):
                return float(w @ cov @ w)

        elif self.method == "mean_variance":
            lam = self.risk_aversion
            def objective(w):
                return float(lam * (w @ cov @ w) - w @ mu)

        else:   # risk_parity
            def objective(w):
                port_var = w @ cov @ w
                # risk contributions should be equal
                rc = (w * (cov @ w)) / (port_var + 1e-12)
                target = np.ones(n) / n
                return float(np.sum((rc - target) ** 2))

        res = minimize(
            objective, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-10, "maxiter": 500},
        )

        if not res.success:
            # fallback to equal weight on failure
            w = np.ones(n) / n
        else:
            w = res.x
            w = np.clip(w, 0, None)
            w = w / w.sum()

        return pd.Series(w, index=returns.columns)

    def __repr__(self) -> str:
        return (f"Optimiser(method={self.method}, lookback={self.lookback}, "
                f"risk_aversion={self.risk_aversion})")
