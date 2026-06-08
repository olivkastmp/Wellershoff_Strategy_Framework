"""
signals/01_probit.py

Probit regime model.

Estimates the probability of a risk-off / stress regime using macro predictors.
Uses statsmodels Probit (binary outcome, standard normal link).

The predicted probability p_t is the main output consumed by the strategy layer:
  - p_t close to 1  →  high probability of stress regime  →  reduce risk
  - p_t close to 0  →  calm regime likely  →  take risk

Design
------
Fits on an expanding in-sample window and produces OOS predicted probabilities
(no look-ahead). Refit at a user-specified frequency (default: monthly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.discrete.discrete_model import Probit
from statsmodels.tools import add_constant


class ProbitRegimeModel:
    """
    Rolling probit model for binary regime classification.

    Parameters
    ----------
    macro        : DataFrame of macro predictors (rows = dates)
    target_col   : column name of the binary regime indicator (0 = calm, 1 = stress)
    feature_cols : list of predictor column names; if None, uses all except target_col
    min_obs      : minimum IS observations before first fit (default 252)
    refit_freq   : how often to refit, as a pandas offset string ('ME', 'W', 'QE')
    lag          : number of lags to apply to predictors to avoid look-ahead (default 1)
    """

    def __init__(
        self,
        macro: pd.DataFrame,
        target_col: str = "regime",
        feature_cols: list[str] | None = None,
        min_obs: int = 252,
        refit_freq: str = "ME",
        lag: int = 1,
    ) -> None:
        self.macro        = macro.copy()
        self.target_col   = target_col
        self.feature_cols = feature_cols or [c for c in macro.columns if c != target_col]
        self.min_obs      = min_obs
        self.refit_freq   = refit_freq
        self.lag          = lag

        self._fitted_models: dict = {}
        self._prob_series: pd.Series | None = None

    def fit_predict(self) -> pd.Series:
        """
        Expanding-window probit: fit up to t, predict t+1 onwards.

        Returns
        -------
        pd.Series : predicted probabilities (same index as macro, NaN in burn-in).
        """
        y     = self.macro[self.target_col]
        X_raw = self.macro[self.feature_cols]
        X     = X_raw.shift(self.lag)   # lag to avoid look-ahead

        probs = pd.Series(np.nan, index=self.macro.index)

        # use updated pandas offset aliases
        freq        = self.refit_freq.upper()
        refit_dates = X.resample(freq).last().index

        for i, refit_date in enumerate(refit_dates):
            mask_is = self.macro.index <= refit_date
            if mask_is.sum() < self.min_obs:
                continue

            # drop NaN rows (from lag + any missing data)
            X_is_raw = X[mask_is]
            y_is_raw = y[mask_is]
            valid    = X_is_raw.notna().all(axis=1) & y_is_raw.notna()
            X_is     = X_is_raw[valid].values
            y_is     = y_is_raw[valid].values

            if len(np.unique(y_is)) < 2:
                continue   # can't fit probit with only one class

            try:
                X_is_c = add_constant(X_is, has_constant="add")
                result  = Probit(y_is, X_is_c).fit(disp=False, maxiter=200)
                self._fitted_models[refit_date] = result
            except Exception:
                continue

            # OOS window: day after refit_date up to next refit date
            if i + 1 < len(refit_dates):
                next_refit = refit_dates[i + 1]
            else:
                next_refit = self.macro.index[-1]

            oos_mask = (self.macro.index > refit_date) & (self.macro.index <= next_refit)
            if oos_mask.sum() == 0:
                continue

            X_oos_raw = X[oos_mask]
            valid_oos  = X_oos_raw.notna().all(axis=1)
            if valid_oos.sum() == 0:
                continue

            X_oos_c = add_constant(X_oos_raw[valid_oos].values, has_constant="add")
            try:
                preds = result.predict(X_oos_c)
                probs.loc[X_oos_raw[valid_oos].index] = preds
            except Exception:
                pass

        self._prob_series = probs
        return probs

    @property
    def probabilities(self) -> pd.Series:
        if self._prob_series is None:
            raise RuntimeError("Call fit_predict() first.")
        return self._prob_series

    def summary(self, date: pd.Timestamp | None = None) -> str:
        if not self._fitted_models:
            return "No fitted models yet."
        key = date or max(self._fitted_models.keys())
        return str(self._fitted_models[key].summary())

    def plot_probabilities(
        self,
        true_regime: pd.Series | None = None,
        ax=None,
        save_path: str | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        if self._prob_series is None:
            raise RuntimeError("Call fit_predict() first.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
            standalone = True
        else:
            standalone = False

        ax.plot(self._prob_series.index, self._prob_series,
                color="#1a5276", linewidth=1.0, label="P(stress regime)")
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)

        if true_regime is not None:
            ax2 = ax.twinx()
            ax2.fill_between(true_regime.index, true_regime,
                             alpha=0.15, color="#922b21", label="True regime")
            ax2.set_ylabel("True regime", fontsize=8)
            ax2.set_ylim(-0.1, 3)

        ax.set_title("Probit Regime Probability", fontsize=10, fontweight="bold")
        ax.set_ylabel("P(stress)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

        if standalone:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            else:
                plt.show()
            plt.close()
