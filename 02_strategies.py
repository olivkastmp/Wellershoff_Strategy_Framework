"""
strategies/02_strategies.py

SAA, TAA, and DAA strategy classes.

All strategies share the same interface:
    strategy.get_weights(date, prices, regime_prob) -> pd.Series

SAA (Strategic Asset Allocation)
---------------------------------
Fixed target weights, rebalanced at a set frequency. Regime probability
is ignored. Benchmark for comparison.

TAA (Tactical Asset Allocation)
---------------------------------
Starts from SAA weights but applies a tilt proportional to (p_t - 0.5).
When p_t > 0.5 (stress likely), defensive assets get overweighted and
risky assets get underweighted. The tilt magnitude is controlled by
`tilt_strength` (default 0.5).

DAA (Dynamic Asset Allocation)
---------------------------------
Fully regime-driven. Maintains two target portfolios:
  - risk_on_weights  : used when p_t < threshold (calm regime)
  - risk_off_weights : used when p_t >= threshold (stress regime)

Between the two thresholds the weights are linearly interpolated, so
the transition is smooth rather than a hard switch.

Usage
-----
    strategy = AllocationStrategy(
        mode         = "TAA",          # "SAA" | "TAA" | "DAA"
        base_weights = {...},          # strategic weights dict {asset: weight}
        asset_classes = {...},         # {asset: "risky" | "defensive"}
        rebalance_freq = "M",
        tilt_strength  = 0.5,          # TAA only
        stress_threshold = 0.6,        # DAA only
        calm_threshold   = 0.4,        # DAA only
        risk_off_weights = {...},      # DAA only
    )
    weights = strategy.get_weights(date, regime_prob=p_t)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal


MODES = Literal["SAA", "TAA", "DAA"]


class AllocationStrategy:
    """
    Unified interface for SAA, TAA, and DAA strategies.

    Parameters
    ----------
    mode            : "SAA", "TAA", or "DAA"
    base_weights    : dict {asset_name: weight}, must sum to 1
    asset_classes   : dict {asset_name: "risky" | "defensive"}
                      used by TAA to determine tilt direction
    rebalance_freq  : pandas offset string for rebalancing ('D', 'W', 'M', 'Q')
    tilt_strength   : TAA only — max absolute tilt from base weights (default 0.3)
    stress_threshold: DAA only — p_t above this triggers risk-off portfolio
    calm_threshold  : DAA only — p_t below this triggers risk-on portfolio
    risk_off_weights: DAA only — target weights in stress regime
    """

    def __init__(
        self,
        mode: MODES = "SAA",
        base_weights: dict | None = None,
        asset_classes: dict | None = None,
        rebalance_freq: str = "M",
        tilt_strength: float = 0.3,
        stress_threshold: float = 0.6,
        calm_threshold: float = 0.4,
        risk_off_weights: dict | None = None,
    ) -> None:
        if mode not in ("SAA", "TAA", "DAA"):
            raise ValueError(f"mode must be 'SAA', 'TAA', or 'DAA', got '{mode}'")

        self.mode             = mode
        self.base_weights     = base_weights or {}
        self.asset_classes    = asset_classes or {}
        self.rebalance_freq   = rebalance_freq
        self.tilt_strength    = tilt_strength
        self.stress_threshold = stress_threshold
        self.calm_threshold   = calm_threshold
        self.risk_off_weights = risk_off_weights or base_weights or {}

        self._last_rebalance: pd.Timestamp | None = None
        self._current_weights: pd.Series | None = None

        # validate
        assets = list(self.base_weights.keys())
        total  = sum(self.base_weights.values())
        if assets and abs(total - 1.0) > 1e-6:
            raise ValueError(f"base_weights must sum to 1.0, got {total:.4f}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_weights(
        self,
        date: pd.Timestamp,
        regime_prob: float = 0.0,
    ) -> pd.Series:
        """
        Compute portfolio weights for a given date and regime probability.

        Parameters
        ----------
        date        : current date
        regime_prob : probit-predicted P(stress), in [0, 1]

        Returns
        -------
        pd.Series : portfolio weights indexed by asset name
        """
        if not self._is_rebalance_day(date):
            return self._current_weights  # hold positions between rebalances

        if self.mode == "SAA":
            weights = self._saa_weights()
        elif self.mode == "TAA":
            weights = self._taa_weights(regime_prob)
        else:   # DAA
            weights = self._daa_weights(regime_prob)

        self._current_weights  = weights
        self._last_rebalance   = date
        return weights

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _saa_weights(self) -> pd.Series:
        """Fixed strategic weights — regime probability not used."""
        return pd.Series(self.base_weights)

    def _taa_weights(self, p: float) -> pd.Series:
        """
        Tilt from base weights proportional to (p - 0.5).

        When p > 0.5: overweight defensive, underweight risky.
        When p < 0.5: overweight risky, underweight defensive.
        Tilt magnitude = tilt_strength * |p - 0.5| * 2  (scaled to [0, tilt_strength]).
        """
        base   = pd.Series(self.base_weights)
        tilt   = pd.Series(0.0, index=base.index)
        signal = (p - 0.5) * 2   # in [-1, 1]

        for asset in base.index:
            ac = self.asset_classes.get(asset, "risky")
            if ac == "defensive":
                tilt[asset] = +self.tilt_strength * signal   # increase when stressed
            else:
                tilt[asset] = -self.tilt_strength * signal   # decrease when stressed

        raw = base + tilt
        raw = raw.clip(lower=0)   # no short positions
        return raw / raw.sum()    # renormalise

    def _daa_weights(self, p: float) -> pd.Series:
        """
        Smooth interpolation between risk-on and risk-off portfolios.

        p < calm_threshold   → fully risk-on (base_weights)
        p > stress_threshold → fully risk-off (risk_off_weights)
        between thresholds   → linear blend
        """
        w_on  = pd.Series(self.base_weights)
        w_off = pd.Series(self.risk_off_weights)

        # ensure same index
        all_assets = w_on.index.union(w_off.index)
        w_on  = w_on.reindex(all_assets, fill_value=0.0)
        w_off = w_off.reindex(all_assets, fill_value=0.0)

        if p <= self.calm_threshold:
            alpha = 0.0
        elif p >= self.stress_threshold:
            alpha = 1.0
        else:
            # linear interpolation in the transition zone
            alpha = (p - self.calm_threshold) / (self.stress_threshold - self.calm_threshold)

        raw = (1 - alpha) * w_on + alpha * w_off
        return raw / raw.sum()

    # ------------------------------------------------------------------
    # Rebalancing logic
    # ------------------------------------------------------------------

    def _is_rebalance_day(self, date: pd.Timestamp) -> bool:
        """Check whether today is a rebalancing day."""
        if self._last_rebalance is None:
            return True   # always rebalance on first day

        freq = self.rebalance_freq.upper()

        if freq == "D":
            return True
        elif freq == "W":
            return date.week != self._last_rebalance.week
        elif freq in ("M", "ME"):
            return date.month != self._last_rebalance.month
        elif freq in ("Q", "QE"):
            return date.quarter != self._last_rebalance.quarter
        else:
            # fallback: use pandas period comparison
            return (date.to_period(freq) != self._last_rebalance.to_period(freq))

    def __repr__(self) -> str:
        return (f"AllocationStrategy(mode={self.mode}, "
                f"rebalance={self.rebalance_freq}, "
                f"assets={list(self.base_weights.keys())})")


def _normalise_freq(freq: str) -> str:
    """Translate deprecated pandas offset aliases to current ones."""
    return freq.upper().replace("ME", "ME").replace("QE", "QE")
