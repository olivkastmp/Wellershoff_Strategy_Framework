"""
example_run.py

End-to-end demonstration of the asset allocation framework.

Runs SAA, TAA, and DAA on synthetic data and compares performance.
Swap in real price/macro data to use with actual assets.

Run: python example_run.py
"""

import importlib
import warnings
import pandas as pd
import sys
import os

warnings.filterwarnings("ignore")

# --- importlib for numeric-prefixed modules ---
sys.path.insert(0, os.path.dirname(__file__))

utils_mod    = importlib.import_module("utils.05_utils")
probit_mod   = importlib.import_module("signals.01_probit")
strat_mod    = importlib.import_module("strategies.02_strategies")
opt_mod      = importlib.import_module("optimisation.03_optimiser")
backtest_mod = importlib.import_module("backtest.04_backtest")

generate_sample_data = utils_mod.generate_sample_data
ProbitRegimeModel    = probit_mod.ProbitRegimeModel
AllocationStrategy   = strat_mod.AllocationStrategy
Optimiser            = opt_mod.Optimiser
BacktestEngine       = backtest_mod.BacktestEngine
performance_table    = backtest_mod.performance_table
plot_tearsheet       = utils_mod.plot_tearsheet


# ---------------------------------------------------------------
# 1. Generate synthetic data
# ---------------------------------------------------------------

print("Generating synthetic data...")
prices, macro = generate_sample_data(n_assets=6, n_obs=1500, seed=42)
assets = list(prices.columns)
print(f"  {prices.shape[0]} days x {prices.shape[1]} assets")
print(f"  {prices.index[0].date()} to {prices.index[-1].date()}\n")

# ---------------------------------------------------------------
# 2. Define base weights and asset classes
# ---------------------------------------------------------------

# equal-weight base (risk-on portfolio)
base_weights = {a: 1/len(assets) for a in assets}

# label half as risky, half as defensive for TAA tilt
asset_classes = {a: ("risky" if i < len(assets) // 2 else "defensive")
                 for i, a in enumerate(assets)}

# risk-off portfolio: shift to defensive assets
risk_off_weights = {a: (0.05 if asset_classes[a] == "risky" else
                        (1 - 0.05 * sum(1 for ac in asset_classes.values() if ac == "risky"))
                        / sum(1 for ac in asset_classes.values() if ac == "defensive"))
                    for a in assets}

# ---------------------------------------------------------------
# 3. Set up probit model
# ---------------------------------------------------------------

regime_model = ProbitRegimeModel(
    macro        = macro,
    target_col   = "regime",
    feature_cols = ["yield_slope", "vix_change", "credit_spread", "momentum"],
    min_obs      = 252,
    refit_freq   = "ME",
    lag          = 1,
)

# ---------------------------------------------------------------
# 4. Run three strategies
# ---------------------------------------------------------------

optimiser = Optimiser(method="min_vol", lookback=126)

strategies = {
    "SAA": AllocationStrategy(
        mode="SAA", base_weights=base_weights,
        asset_classes=asset_classes, rebalance_freq="ME",
    ),
    "TAA": AllocationStrategy(
        mode="TAA", base_weights=base_weights,
        asset_classes=asset_classes, rebalance_freq="ME",
        tilt_strength=0.3,
    ),
    "DAA": AllocationStrategy(
        mode="DAA", base_weights=base_weights,
        asset_classes=asset_classes, rebalance_freq="W",
        stress_threshold=0.65, calm_threshold=0.35,
        risk_off_weights=risk_off_weights,
    ),
}

results = {}
for name, strategy in strategies.items():
    print(f"Running {name}...")
    engine = BacktestEngine(
        prices               = prices,
        macro                = macro,
        strategy             = strategy,
        regime_model         = regime_model if name != "SAA" else None,
        optimiser            = optimiser,
        transaction_cost_bps = 5.0,
        min_history          = 252,
    )
    results[name] = engine.run()
    print(f"  done — {len(results[name].portfolio_returns)} OOS days\n")

# ---------------------------------------------------------------
# 5. Print performance table
# ---------------------------------------------------------------

print("\n" + "=" * 70)
print("  Performance Summary (synthetic data, 5 bps one-way TC)")
print("=" * 70)
print(performance_table(results).to_string())
print("=" * 70 + "\n")

# ---------------------------------------------------------------
# 6. Tearsheet
# ---------------------------------------------------------------

plot_tearsheet(
    {name: res.portfolio_returns for name, res in results.items()},
    title="SAA vs TAA vs DAA — Synthetic Data",
    save_path="output/tearsheet.png",
)
print("Tearsheet saved to output/tearsheet.png")
