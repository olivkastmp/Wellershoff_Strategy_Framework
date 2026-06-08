"""
backtest/04_backtest.py

Rolling backtest engine.

Wires together:
  1. ProbitRegimeModel  → predicted regime probabilities
  2. AllocationStrategy → target weights given regime prob
  3. Optimiser          → refined weights (optional)
  4. Performance metrics and tearsheet

The engine loops day-by-day, which is slower than vectorised approaches
but makes the strategy/optimiser interaction explicit and easy to modify.

Transaction costs are applied on changes in portfolio weight (one-way).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Output container for a single backtest run."""
    portfolio_returns: pd.Series
    equity_curve:      pd.Series
    weights:           pd.DataFrame
    turnover:          pd.Series
    tc_drag:           pd.Series
    regime_probs:      pd.Series
    params:            dict = field(default_factory=dict)


class BacktestEngine:
    """
    Rolling backtest engine.

    Parameters
    ----------
    prices           : wide-format price DataFrame (dates x assets)
    macro            : macro predictor DataFrame (same index as prices)
    strategy         : AllocationStrategy instance
    regime_model     : ProbitRegimeModel instance (optional; if None, p_t = 0)
    optimiser        : Optimiser instance (optional; if None, uses raw strategy weights)
    transaction_cost_bps : one-way TC in basis points (default 5)
    min_history      : minimum days before backtest starts (probit burn-in)
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame,
        strategy,
        regime_model=None,
        optimiser=None,
        transaction_cost_bps: float = 5.0,
        min_history: int = 252,
    ) -> None:
        self.prices    = prices
        self.macro     = macro
        self.strategy  = strategy
        self.regime_model = regime_model
        self.optimiser = optimiser
        self.tc        = transaction_cost_bps / 10_000
        self.min_history = min_history

    def run(self) -> BacktestResult:
        """Execute the backtest and return results."""

        # --- Step 1: fit probit model and get regime probabilities ---
        if self.regime_model is not None:
            print("Fitting probit regime model...")
            regime_probs = self.regime_model.fit_predict()
        else:
            regime_probs = pd.Series(0.0, index=self.prices.index)

        # --- Step 2: compute returns ---
        returns = np.log(self.prices / self.prices.shift(1))

        # --- Step 3: align all series ---
        idx  = self.prices.index.intersection(regime_probs.dropna().index)
        idx  = idx[idx >= self.prices.index[self.min_history]]

        assets = list(self.strategy.base_weights.keys())
        # filter to assets that exist in prices
        assets = [a for a in assets if a in self.prices.columns]

        # --- Step 4: day-by-day loop ---
        weight_records = []
        port_returns   = []
        prev_weights   = pd.Series(0.0, index=assets)

        for date in idx:
            p_t = float(regime_probs.get(date, 0.0))
            if np.isnan(p_t):
                p_t = 0.0

            # get target weights from strategy
            target_w = self.strategy.get_weights(date, regime_prob=p_t)
            if target_w is None:
                target_w = prev_weights.copy()

            target_w = target_w.reindex(assets, fill_value=0.0)

            # optionally refine with optimiser
            if self.optimiser is not None:
                hist_ret = returns.loc[:date, assets].dropna()
                if len(hist_ret) >= self.optimiser.lookback:
                    opt_w = self.optimiser.optimise(hist_ret, signal_weights=target_w)
                    target_w = opt_w.reindex(assets, fill_value=0.0)

            # normalise
            total = target_w.abs().sum()
            if total > 0:
                target_w = target_w / total

            # portfolio return: use yesterday's weights on today's return
            day_ret = returns.loc[date, assets].fillna(0.0)
            port_ret = float((prev_weights * day_ret).sum())

            # transaction cost: one-way on weight changes
            turnover = float((target_w - prev_weights).abs().sum())
            tc_cost  = turnover * self.tc
            net_ret  = port_ret - tc_cost

            port_returns.append({
                "date":      date,
                "gross_ret": port_ret,
                "tc_cost":   tc_cost,
                "net_ret":   net_ret,
                "turnover":  turnover,
                "regime_p":  p_t,
            })
            weight_records.append({"date": date, **target_w.to_dict()})
            prev_weights = target_w.copy()

        # --- Step 5: assemble results ---
        df_ret = pd.DataFrame(port_returns).set_index("date")
        df_w   = pd.DataFrame(weight_records).set_index("date")

        net_ret   = df_ret["net_ret"]
        equity    = (1 + net_ret).cumprod()

        return BacktestResult(
            portfolio_returns = net_ret,
            equity_curve      = equity,
            weights           = df_w,
            turnover          = df_ret["turnover"],
            tc_drag           = df_ret["tc_cost"],
            regime_probs      = df_ret["regime_p"],
            params = {
                "mode":    self.strategy.mode,
                "tc_bps":  self.tc * 10_000,
                "assets":  assets,
                "start":   str(idx[0].date()),
                "end":     str(idx[-1].date()),
            },
        )


def performance_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """
    Build a summary performance table from multiple backtest results.

    Parameters
    ----------
    results : dict of {label: BacktestResult}

    Returns
    -------
    pd.DataFrame with one row per strategy
    """
    import importlib
    utils = importlib.import_module("utils.05_utils")

    rows = []
    for label, res in results.items():
        r  = res.portfolio_returns
        eq = res.equity_curve
        rows.append({
            "Strategy":      label,
            "Ann. Return":   f"{utils.annualised_return(r)*100:.2f}%",
            "Ann. Vol":      f"{utils.annualised_vol(r)*100:.2f}%",
            "Sharpe":        f"{utils.sharpe(r):.3f}",
            "Max DD":        f"{utils.max_drawdown(r)*100:.2f}%",
            "Calmar":        f"{utils.calmar(r):.3f}",
            "Avg Turnover":  f"{res.turnover.mean()*100:.1f}%/day",
            "Avg TC (bps)":  f"{res.tc_drag.mean()*10_000:.2f}",
        })
    return pd.DataFrame(rows).set_index("Strategy")
