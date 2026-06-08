# Asset Allocation Framework

A modular Python framework for strategic, tactical, and dynamic asset allocation
with probit-based regime signals, mean-variance portfolio optimisation, and
rolling backtesting.

## Structure

```
asset_allocation/
├── signals/
│   └── 01_probit.py        # Probit regime model + predicted probabilities
├── strategies/
│   └── 02_strategies.py    # SAA, TAA, DAA strategy classes
├── optimisation/
│   └── 03_optimiser.py     # Mean-variance, min-vol, risk-parity optimisers
├── backtest/
│   └── 04_backtest.py      # Rolling backtest engine + performance metrics
├── utils/
│   └── 05_utils.py         # Data loading, returns, plotting helpers
└── example_run.py          # End-to-end example on synthetic data
```

## Strategy modes

Switch between strategies by passing `mode` to `AllocationStrategy`:

```python
from strategies.02_strategies import AllocationStrategy  # via importlib

# Strategic: fixed weights, rebalance monthly
strategy = AllocationStrategy(mode="SAA", rebalance_freq="M")

# Tactical: tilt weights based on probit regime signal
strategy = AllocationStrategy(mode="TAA", rebalance_freq="M")

# Dynamic: full weight shift driven by regime probability
strategy = AllocationStrategy(mode="DAA", rebalance_freq="W")
```

## Probit signal

The regime model estimates the probability of a "risk-off" regime using macro
predictors (e.g. yield curve slope, VIX changes, credit spreads). The predicted
probability p_t is then used to scale allocations:

- SAA: ignores p_t entirely
- TAA: tilts +/- from strategic weights proportional to (p_t - 0.5)
- DAA: shifts fully between a risk-on and risk-off portfolio based on p_t

## Quick start

```bash
pip install -r requirements.txt
python example_run.py
```

## Requirements

Python 3.9+, see requirements.txt
