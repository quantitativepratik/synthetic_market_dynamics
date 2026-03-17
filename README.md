# Synthetic Market Dynamics & Predictive Time Series Analysis

A self-contained research project combining volatility modeling with high-frequency
trading simulation. No external financial data required — everything is generated
from the ground up using statistically grounded models.

---

## What this does

### Part 1 — Time Series Modeling

**EGARCH(1,1)** volatility model fitted via maximum likelihood estimation:

- **Asymmetric (leverage) effects** — negative shocks amplify volatility more than
  positive ones, consistent with real equity markets
- Key fitted parameters:
  - `gamma = 0.644` — asymmetry/leverage coefficient
  - `beta ≈ 0.97` — high persistence (volatility clusters)
- **ARIMA(0,0,0)** mean equation (log-returns are approximately white noise)
- **30-day Monte Carlo price forecast** (500 simulated paths) with stable forecast
  sigma ≈ 0.0069
- **Risk metrics** at 95% confidence:
  - VaR ≈ -1.21% (1-day)
  - CVaR ≈ -1.49% (1-day)

### Part 2 — HFT Market Simulation

Synthetic market with **5 million trades** designed to replicate realistic HFT dynamics:

- **Three agent types**: market makers (35%), momentum traders (40%), noise/retail (25%)
- **Realistic trade size distribution**: log-normal mixture + Pareto block-trade tail
- **Intraday U-shaped volume profile**: open/close spikes, midday lull
- **Price impact model**: signed impact + mean reversion
- **Total P&L**: ~301M profit units across all strategies
- **30-minute binning**: 17,568 bins over 45 simulated trading days
- **K-means clustering** (k=6): isolates distinct market regimes by volume, volatility, and order flow imbalance

---

## Project structure

```
synthetic_market_dynamics/
├── src/
│   ├── main.py                  # Pipeline runner — start here
│   ├── time_series_model.py     # ARIMA, EGARCH, VaR/CVaR, forecasting
│   ├── hft_market_simulator.py  # 5M-trade synthetic market + clustering
│   └── visualizations.py       # All 6 output figures
├── outputs/                     # Generated figures (created on run)
├── notebooks/                   # Jupyter exploration notebooks
│   └── exploration.ipynb
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### Requirements

```
numpy>=1.24
pandas>=1.5
matplotlib>=3.6
seaborn>=0.12
scipy>=1.10
scikit-learn>=1.2
```

Optional (for extended functionality):
```
arch>=5.3          # production-grade GARCH/EGARCH
statsmodels>=0.14  # full ARIMA with AIC/BIC
jupyter            # for notebooks
```

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
cd src
python main.py
```

Output figures are saved to `outputs/`. Runtime: approximately **2–4 minutes**
(most of it is the 5M-trade simulation and EGARCH MLE).

---

## Outputs

| File | Description |
|------|-------------|
| `fig1_egarch_fit.png` | Price series, returns, and EGARCH conditional volatility vs. true DGP |
| `fig2_forecast.png` | 30-day Monte Carlo fan chart + volatility term structure |
| `fig3_risk.png` | Return distribution with VaR/CVaR, rolling 21-day risk |
| `fig4_hft_activity.png` | 30-min volume timeline, intraday pattern, OFI distribution |
| `fig5_clusters.png` | K-means cluster analysis (6 market regimes) |
| `fig6_pnl.png` | Cumulative P&L by strategy, distribution, total attribution |

---

## Key technical choices

**Why EGARCH instead of GARCH?**
Standard GARCH(1,1) imposes a symmetric response to positive and negative shocks.
In equity markets, crashes cause larger vol spikes than equivalent-sized rallies.
EGARCH's log-variance formulation naturally captures this via the `gamma` (leverage)
parameter without requiring sign constraints on coefficients.

**Why synthetic data?**
Real tick data is expensive, licensed, and hard to reproduce. Synthetic data with
known DGP lets you verify that the estimation procedure recovers the true parameters —
a crucial sanity check before applying to real data.

**Why k-means on 30-min bins?**
30 minutes is the sweet spot for HFT regime analysis: fine enough to see intraday
structure (open/close effects, lunch lull), coarse enough to average out microstructure
noise. K-means on volume + OFI + volatility naturally separates opening frenzy,
calm midday, closing rush, and extreme event bins.

---

## Extending the project

- **Swap in real data**: replace `generate_price_series()` with a CSV loader
- **Upgrade EGARCH**: install `arch` package and swap in `arch.univariate.EGARCH`
- **Add order book depth**: extend `hft_market_simulator` with bid/ask spread modeling
- **Run walk-forward validation**: split the 1000-day series into rolling train/test windows
- **Add regime switching**: overlay a Markov-switching model on top of EGARCH

---

*Built with numpy, pandas, scipy, scikit-learn, and matplotlib.*
