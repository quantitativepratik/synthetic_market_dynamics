
import os
import time
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(__file__))

from time_series_model import (
    generate_price_series,
    ARIMA,
    EGARCH,
    compute_var_cvar,
    forecast_prices,
)
from hft_market_simulator import (
    run_simulation,
    bin_to_30min,
    cluster_bins,
)
from visualizations import (
    plot_egarch_fit,
    plot_forecast,
    plot_risk_metrics,
    plot_hft_activity,
    plot_clusters,
    plot_pnl_attribution,
)


def banner(msg):
    width = 60
    print("\n" + "═" * width)
    print(f"  {msg}")
    print("═" * width)


def main():
    os.makedirs("outputs", exist_ok=True)
    
    total_start = time.time()
    
    # time series analysis
    banner("PART 1 — Time Series: ARIMA + EGARCH")
    
    print("\n[1/6] Generating synthetic price series (1000 days)...")
    df = generate_price_series(n=1000, seed=42)
    print(f"      Price range: ${df['price'].min():.2f} – ${df['price'].max():.2f}")
    print(f"      Return stats: μ={df['log_ret'].mean()*100:.4f}%, σ={df['log_ret'].std()*100:.4f}%")
    
    print("\n[2/6] Fitting ARIMA(0,0,0) mean equation...")
    arima = ARIMA(p=0, d=0, q=0)
    arima.fit(df["log_ret"])
    print(f"      Drift (μ): {arima.params['mu']:.6f}")
    
    print("\n[3/6] Fitting EGARCH(1,1) volatility model...")
    t0 = time.time()
    egarch = EGARCH()
    egarch.fit(df["log_ret"].values)
    elapsed = time.time() - t0
    
    p = egarch.params
    print(f"      Converged: {egarch.converged}  ({elapsed:.1f}s)")
    print(f"      ω  (omega) = {p['omega']:.4f}")
    print(f"      α  (alpha) = {p['alpha']:.4f}  (ARCH effect / shock magnitude)")
    print(f"      γ  (gamma) = {p['gamma']:.4f}  ← leverage / asymmetry term")
    print(f"      β  (beta)  = {p['beta']:.4f}  (volatility persistence)")
    
    print("\n[4/6] Computing VaR and CVaR (95% confidence)...")
    var, cvar = compute_var_cvar(df["log_ret"].values)
    print(f"      VaR  = {var*100:.2f}%")
    print(f"      CVaR = {cvar*100:.2f}%")
    
    print("\n[5/6] 30-day Monte Carlo price forecast (500 paths)...")
    fc = forecast_prices(df, arima, egarch, steps=30, n_simulations=500)
    print(f"      Mean forecast σ (sigma):  {fc['mean_vol']:.4f}")
    print(f"      Median exit price:        ${fc['median'][-1]:.2f}")
    print(f"      90% prediction interval:  ${fc['lower5'][-1]:.2f} – ${fc['upper95'][-1]:.2f}")
    
    # Save time series figures
    print("\n  Saving figures...")
    plot_egarch_fit(df, egarch)
    plot_forecast(df, fc)
    plot_risk_metrics(df["log_ret"].values, var, cvar)
    
    # hft simulation
    banner("PART 2 — HFT Synthetic Market Simulation")
    
    print("\n[6/6] Running 5-million-trade simulation...")
    t0 = time.time()
    trades = run_simulation(n_trades=5_000_000, seed=42)
    sim_elapsed = time.time() - t0
    print(f"      Simulation time: {sim_elapsed:.1f}s")
    
    total_pnl = trades["pnl"].sum()
    print(f"      Total P&L: {total_pnl:,.0f} units ({total_pnl/1e6:.1f}M)")
    
    # binning
    print("\n  Binning into 30-minute windows...")
    binned = bin_to_30min(trades)
    print(f"      Total 30-min bins: {len(binned):,}")
    print(f"      Max bin volume:    {binned['volume'].max():,}")
    print(f"      Avg bin volume:    {binned['volume'].mean():,.0f}")
    
    #clustering
    print("\n  Fitting k-means (6 clusters)...")
    binned_cl, cluster_summary, km, scaler = cluster_bins(binned, n_clusters=6)
    
    print("\n  Cluster summary (sorted by total volume):")
    display_cols = ["rank_label", "n_bins", "avg_trades", "total_volume", "avg_range", "avg_ofi"]
    available = [c for c in display_cols if c in cluster_summary.columns]
    print(cluster_summary[available].to_string(index=False))
    
    #strategy
    print("\n  Strategy P&L breakdown:")
    pnl_by_strat = trades.groupby("strategy")["pnl"].sum()
    for strat, pnl in pnl_by_strat.items():
        pct = pnl / total_pnl * 100
        print(f"      {strat:15s}: {pnl:>12,.0f} units  ({pct:.1f}%)")
    print(f"      {'TOTAL':15s}: {total_pnl:>12,.0f} units")
    
    # Save HFT figures
    print("\n  Saving figures...")
    plot_hft_activity(binned)
    plot_clusters(binned_cl, cluster_summary)
    plot_pnl_attribution(trades)
    
    #summary
    banner("RESULTS SUMMARY")
    
    total_elapsed = time.time() - total_start
    
    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │         SYNTHETIC MARKET DYNAMICS — RESULTS         │
  ├─────────────────────────────────────────────────────┤
  │  TIME SERIES                                        │
  │    EGARCH gamma (leverage): {p['gamma']:>8.4f}               │
  │    EGARCH beta  (persist.): {p['beta']:>8.4f}               │
  │    30-day forecast sigma:   {fc['mean_vol']:>8.4f}               │
  │    VaR  (95%, 1-day):      {var*100:>7.2f}%                │
  │    CVaR (95%, 1-day):      {cvar*100:>7.2f}%                │
  ├─────────────────────────────────────────────────────┤
  │  HFT SIMULATION                                     │
  │    Total trades:          {len(trades):>10,}               │
  │    Total P&L:             {total_pnl:>10,.0f} units         │
  │    30-min bins:           {len(binned):>10,}               │
  │    K-means clusters:               6               │
  ├─────────────────────────────────────────────────────┤
  │  Outputs: 6 figures saved to outputs/               │
  │  Total runtime: {total_elapsed:.1f}s                           │
  └─────────────────────────────────────────────────────┘
""")
    
    return {
        "df_prices":       df,
        "arima":           arima,
        "egarch":          egarch,
        "var":             var,
        "cvar":            cvar,
        "forecast":        fc,
        "trades":          trades,
        "binned":          binned_cl,
        "cluster_summary": cluster_summary,
    }


if __name__ == "__main__":
    results = main()
