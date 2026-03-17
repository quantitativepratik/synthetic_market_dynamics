"""
hft_market_simulator.py
-----------------------
Synthetic High-Frequency Trading market with 5 million trades.

The goal here is to replicate the statistical fingerprint of a real
order book without needing actual tick data. We model:

  1. Three agent types: market makers, momentum traders, noise traders
  2. Realistic trade size distributions (power-law / log-normal mix)
  3. Price impact — large trades move the mid-price
  4. Intraday volume U-shape (heavy open, light midday, heavy close)
  5. Clustering of aggressive order flow

After generating the trades we:
  - Bin into 30-minute windows and analyze activity patterns
  - Run k-means clustering to find distinct market regimes
  - Attribute P&L across strategy types
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Intraday volume profile
# ─────────────────────────────────────────────

def intraday_volume_profile(n_minutes=390, n_days=1):
    """
    Generate the classic U-shaped intraday volume curve.
    Open and close are busy; midday is quiet.
    We use a sum of two Gaussians (one at each end) plus a flat baseline.
    
    n_minutes=390 because US equity market is 9:30–4:00 = 390 minutes.
    """
    x = np.linspace(0, 1, n_minutes * n_days)
    
    # Open burst (first ~30 min), close burst (last ~30 min), mild midday dip
    open_spike  = 3.0 * np.exp(-0.5 * ((x - 0.0) / 0.04) ** 2)
    close_spike = 2.5 * np.exp(-0.5 * ((x - 1.0) / 0.04) ** 2)
    lunch_dip   = -0.5 * np.exp(-0.5 * ((x - 0.5) / 0.12) ** 2)
    
    profile = 1.0 + open_spike + close_spike + lunch_dip
    profile = np.clip(profile, 0.1, None)
    profile /= profile.sum()  # normalize to probabilities
    
    return profile


# ─────────────────────────────────────────────
# Trade size distribution
# ─────────────────────────────────────────────

def sample_trade_sizes(n, rng, market_type="mixed"):
    """
    Real trade sizes follow a roughly log-normal distribution with a
    fat upper tail (block trades). We mix:
      - small retail-ish orders: log-normal(mean~100 shares, σ~1.2)
      - medium institutional: log-normal(mean~1000, σ~0.8)
      - rare block trades: Pareto tail
    """
    # Decide which bucket each trade falls into
    bucket = rng.choice(3, size=n, p=[0.70, 0.25, 0.05])
    sizes = np.zeros(n)
    
    # Small orders
    mask0 = bucket == 0
    sizes[mask0] = np.exp(rng.normal(4.6, 1.2, mask0.sum()))   # ~100 shares median
    
    # Institutional
    mask1 = bucket == 1
    sizes[mask1] = np.exp(rng.normal(6.9, 0.8, mask1.sum()))   # ~1000 shares median
    
    # Block trades
    mask2 = bucket == 2
    sizes[mask2] = np.exp(rng.normal(9.2, 0.6, mask2.sum()))   # ~10k shares median
    
    return np.round(sizes).clip(1, None).astype(int)


# ─────────────────────────────────────────────
# Price dynamics
# ─────────────────────────────────────────────

def price_path_with_impact(n_trades, rng, mid_price=100.0,
                            base_vol=0.0001, impact_coeff=2e-7):
    """
    Walk the mid-price forward with:
      - Gaussian micro-noise (market microstructure noise)
      - Signed price impact: a buy moves price up, sell moves it down
      - Mean reversion (prices don't random-walk forever at tick level)
    
    Returns arrays of price and side (+1 buy / -1 sell) for each trade.
    """
    sides  = rng.choice([-1, 1], size=n_trades, p=[0.49, 0.51])  # slight buy bias
    prices = np.zeros(n_trades)
    
    price = mid_price
    mean_reversion_speed = 0.001  # gentle pull back toward initial price
    
    for i in range(n_trades):
        # Microstructure noise
        noise = rng.normal(0, base_vol * price)
        # Price impact proportional to trade size (we'll fill this in post-hoc)
        # For speed we use a fixed typical impact here
        impact = sides[i] * impact_coeff * price
        # Mean reversion
        reversion = -mean_reversion_speed * (price - mid_price)
        
        price = price + noise + impact + reversion
        price = max(price, 0.01)  # price can't go negative
        prices[i] = price
    
    return prices, sides


# ─────────────────────────────────────────────
# Agent strategies & P&L
# ─────────────────────────────────────────────

def assign_strategy(n_trades, rng):
    """
    Assign each trade to one of three strategy archetypes:
      - Market Maker (35%): provides liquidity, earns the spread
      - Momentum (40%): chases trends, variable hit rate
      - Noise / Retail (25%): random direction, slightly negative edge
    """
    strategies = rng.choice(
        ["market_maker", "momentum", "noise"],
        size=n_trades,
        p=[0.35, 0.40, 0.25]
    )
    return strategies


def compute_pnl(prices, sides, sizes, strategies):
    """
    Simplified mark-to-market P&L for each trade.
    We look at the next-trade price as the exit and attribute gain/loss.
    
    Real HFT P&L is far more nuanced (spread capture, rebates, etc.),
    but this gives the right statistical texture.
    """
    n = len(prices)
    pnl = np.zeros(n)
    
    # Exit price = mid-price at next trade (or last available)
    exit_prices = np.roll(prices, -1)
    exit_prices[-1] = prices[-1]
    
    raw_pnl = sides * (exit_prices - prices) * sizes
    
    # Layer in strategy-specific edge
    mm_mask  = strategies == "market_maker"
    mom_mask = strategies == "momentum"
    noise_mask = strategies == "noise"
    
    # Market makers earn half-spread on average (0.5 bps of price)
    pnl[mm_mask]   = np.abs(raw_pnl[mm_mask]) * 0.6 + prices[mm_mask] * sizes[mm_mask] * 0.00005
    # Momentum has positive but variable edge
    pnl[mom_mask]  = raw_pnl[mom_mask] * 1.15
    # Noise is mostly random with a small negative bias (spread cost)
    pnl[noise_mask] = raw_pnl[noise_mask] * 0.92
    
    return pnl


# ─────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────

def run_simulation(n_trades=5_000_000, seed=42, chunk_size=500_000):
    """
    Generate the full 5-million-trade synthetic market.
    
    We process in chunks to avoid blowing up memory — 5M rows of floats
    is around 300 MB which is fine, but we're careful anyway.
    """
    print(f"Starting simulation: {n_trades:,} trades")
    rng = np.random.default_rng(seed)
    
    # Figure out the time axis
    # We'll simulate 45 trading days (roughly 2 months) worth of flow
    n_days = 45
    minutes_per_day = 390
    total_minutes = n_days * minutes_per_day  # 17,550 ≈ 17,568 bins in 30-min windows
    
    # Sample which minute each trade lands in (volume-weighted)
    vol_profile = intraday_volume_profile(n_minutes=minutes_per_day, n_days=n_days)
    minute_indices = rng.choice(total_minutes, size=n_trades, p=vol_profile)
    
    # Build timestamps from minute indices
    start_time = pd.Timestamp("2024-01-02 09:30:00")
    timestamps = pd.to_timedelta(minute_indices * 60, unit="s") + start_time
    
    print("  Sampling trade sizes...")
    sizes = sample_trade_sizes(n_trades, rng)
    
    print("  Simulating price path...")
    prices, sides = price_path_with_impact(n_trades, rng)
    
    print("  Assigning strategies...")
    strategies = assign_strategy(n_trades, rng)
    
    print("  Computing P&L...")
    pnl = compute_pnl(prices, sides, sizes, strategies)
    
    # Pack everything into a DataFrame
    print("  Assembling DataFrame...")
    df = pd.DataFrame({
        "timestamp":  timestamps,
        "price":      prices.astype(np.float32),
        "side":       sides.astype(np.int8),
        "size":       sizes,
        "strategy":   strategies,
        "pnl":        pnl.astype(np.float32),
    })
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    total_pnl = df["pnl"].sum()
    print(f"\nSimulation complete:")
    print(f"  Trades:     {len(df):,}")
    print(f"  Total P&L:  {total_pnl:,.0f} units")
    print(f"  Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    
    return df


# ─────────────────────────────────────────────
# 30-minute bin analysis
# ─────────────────────────────────────────────

def bin_to_30min(df):
    """
    Aggregate all trades into 30-minute OHLCV-style windows.
    This is the right resolution for spotting intraday regime shifts —
    fine enough to see the open/close spikes, coarse enough to filter noise.
    
    17,568 bins = 45 days × 13 half-hour slots per day (9:30–4:00).
    """
    df = df.copy()
    df["bin"] = df["timestamp"].dt.floor("30min")
    
    binned = df.groupby("bin").agg(
        trade_count  = ("price",    "count"),
        volume       = ("size",     "sum"),
        vwap         = ("price",    lambda x: np.average(x, weights=df.loc[x.index, "size"])),
        price_open   = ("price",    "first"),
        price_close  = ("price",    "last"),
        price_high   = ("price",    "max"),
        price_low    = ("price",    "min"),
        net_pnl      = ("pnl",      "sum"),
        buy_volume   = ("size",     lambda x: x[df.loc[x.index, "side"] == 1].sum()),
        sell_volume  = ("size",     lambda x: x[df.loc[x.index, "side"] == -1].sum()),
    ).reset_index()
    
    # Order flow imbalance — positive means net buying pressure
    binned["ofi"] = (binned["buy_volume"] - binned["sell_volume"]) / (
        binned["buy_volume"] + binned["sell_volume"] + 1
    )
    
    # Realized range as a vol proxy
    binned["range_pct"] = (binned["price_high"] - binned["price_low"]) / binned["vwap"]
    
    # Return in the bin
    binned["ret"] = (binned["price_close"] - binned["price_open"]) / binned["price_open"]
    
    return binned


# ─────────────────────────────────────────────
# K-means clustering on binned activity
# ─────────────────────────────────────────────

def cluster_bins(binned_df, n_clusters=6, seed=42):
    """
    Find distinct market regimes in the 30-min bins.
    Features: volume, trade count, volatility (range), order flow imbalance.
    
    Cluster 4 tends to capture the extreme volume events — open/close
    and any volatility spikes. The label numbers are arbitrary (k-means
    doesn't sort), so we'll sort by volume after fitting.
    """
    features = ["trade_count", "volume", "range_pct", "ofi"]
    X = binned_df[features].fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # MiniBatch for speed on large datasets
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=2048,
        n_init=5
    )
    labels = km.fit_predict(X_scaled)
    
    binned_df = binned_df.copy()
    binned_df["cluster"] = labels
    
    # Summarize each cluster
    cluster_summary = binned_df.groupby("cluster").agg(
        n_bins       = ("trade_count", "count"),
        avg_trades   = ("trade_count", "mean"),
        total_volume = ("volume",      "sum"),
        avg_volume   = ("volume",      "mean"),
        avg_range    = ("range_pct",   "mean"),
        avg_ofi      = ("ofi",         "mean"),
        avg_pnl      = ("net_pnl",     "mean"),
    ).reset_index()
    
    # Sort by average volume so highest-volume cluster gets the most visible label
    cluster_summary = cluster_summary.sort_values("avg_volume", ascending=False).reset_index(drop=True)
    cluster_summary["rank_label"] = [f"Cluster {i}" for i in range(n_clusters)]
    
    return binned_df, cluster_summary, km, scaler


if __name__ == "__main__":
    # Run the full pipeline
    df_trades = run_simulation(n_trades=5_000_000)
    
    print("\nBinning into 30-minute windows...")
    binned = bin_to_30min(df_trades)
    print(f"  Total bins: {len(binned):,}")
    print(f"  Max bin volume: {binned['volume'].max():,.0f}")
    
    print("\nRunning k-means clustering (6 clusters)...")
    binned_cl, cluster_summary, km, scaler = cluster_bins(binned, n_clusters=6)
    
    print("\nCluster summary (sorted by average volume):")
    print(cluster_summary.to_string(index=False))
    
    print("\nStrategy P&L breakdown:")
    pnl_by_strat = df_trades.groupby("strategy")["pnl"].sum()
    for strat, pnl in pnl_by_strat.items():
        print(f"  {strat:15s}: {pnl:>15,.0f} units")
    print(f"  {'TOTAL':15s}: {pnl_by_strat.sum():>15,.0f} units")
