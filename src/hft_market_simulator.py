
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# intraday_volume

def intraday_volume_profile(n_minutes=390, n_days=1):
    """
    Generate the classic U-shaped intraday volume curve.
    Open and close are busy; midday is quiet.
    We use a sum of two Gaussians (one at each end) plus a flat baseline.
    
    n_minutes=390 because US equity market is 9:30–4:00 = 390 minutes.
    """
    x = np.linspace(0, 1, n_minutes * n_days)
    
    #open burst (first ~30 min), close burst (last ~30 min), mild midday dip
    open_spike  = 3.0 * np.exp(-0.5 * ((x - 0.0) / 0.04) ** 2)
    close_spike = 2.5 * np.exp(-0.5 * ((x - 1.0) / 0.04) ** 2)
    lunch_dip   = -0.5 * np.exp(-0.5 * ((x - 0.5) / 0.12) ** 2)
    
    profile = 1.0 + open_spike + close_spike + lunch_dip
    profile = np.clip(profile, 0.1, None)
    profile /= profile.sum() 
    
    return profile


#trade sizes
def sample_trade_sizes(n, rng, market_type="mixed"):
    """
    real trade sizes follow a roughly log-normal distribution with a
    fat upper tail (block trades). We mix:
      - small retail-ish orders: log-normal(mean~100 shares, σ~1.2)
      - medium institutional: log-normal(mean~1000, σ~0.8)
      - rare block trades: Pareto tail
    """
    bucket = rng.choice(3, size=n, p=[0.70, 0.25, 0.05])
    sizes = np.zeros(n)
    
    # small orders
    mask0 = bucket == 0
    sizes[mask0] = np.exp(rng.normal(4.6, 1.2, mask0.sum()))   # ~100 shares median
    
    # institutional
    mask1 = bucket == 1
    sizes[mask1] = np.exp(rng.normal(6.9, 0.8, mask1.sum()))   # ~1000 shares median
    
    # block trades
    mask2 = bucket == 2
    sizes[mask2] = np.exp(rng.normal(9.2, 0.6, mask2.sum()))   # ~10k shares median
    
    return np.round(sizes).clip(1, None).astype(int)


#price dynamics
def price_path_with_impact(n_trades, rng, mid_price=100.0,
                            base_vol=0.0001, impact_coeff=2e-7):
    sides  = rng.choice([-1, 1], size=n_trades, p=[0.49, 0.51])  # slight buy bias
    prices = np.zeros(n_trades)
    
    price = mid_price
    mean_reversion_speed = 0.001  # gentle pull back toward initial price
    
    for i in range(n_trades):
        noise = rng.normal(0, base_vol * price)
        impact = sides[i] * impact_coeff * price
        reversion = -mean_reversion_speed * (price - mid_price)
        
        price = price + noise + impact + reversion
        price = max(price, 0.01)
        prices[i] = price
    
    return prices, sides

def assign_strategy(n_trades, rng):
    strategies = rng.choice(
        ["market_maker", "momentum", "noise"],
        size=n_trades,
        p=[0.35, 0.40, 0.25]
    )
    return strategies


def compute_pnl(prices, sides, sizes, strategies):
    n = len(prices)
    pnl = np.zeros(n)
    exit_prices = np.roll(prices, -1)
    exit_prices[-1] = prices[-1]
    raw_pnl = sides * (exit_prices - prices) * sizes
    mm_mask  = strategies == "market_maker"
    mom_mask = strategies == "momentum"
    noise_mask = strategies == "noise"
    pnl[mm_mask]   = np.abs(raw_pnl[mm_mask]) * 0.6 + prices[mm_mask] * sizes[mm_mask] * 0.00005
    pnl[mom_mask]  = raw_pnl[mom_mask] * 1.15
    pnl[noise_mask] = raw_pnl[noise_mask] * 0.92
    
    return pnl


#main sim

def run_simulation(n_trades=5_000_000, seed=42, chunk_size=500_000):
   
    print(f"Starting simulation: {n_trades:,} trades")
    rng = np.random.default_rng(seed)
    n_days = 45
    minutes_per_day = 390
    total_minutes = n_days * minutes_per_day  
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

def bin_to_30min(df):
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
   
    binned["ofi"] = (binned["buy_volume"] - binned["sell_volume"]) / (
        binned["buy_volume"] + binned["sell_volume"] + 1
    )
    binned["range_pct"] = (binned["price_high"] - binned["price_low"]) / binned["vwap"]
    binned["ret"] = (binned["price_close"] - binned["price_open"]) / binned["price_open"]
    
    return binned

def cluster_bins(binned_df, n_clusters=6, seed=42):
    features = ["trade_count", "volume", "range_pct", "ofi"]
    X = binned_df[features].fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
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
    
    cluster_summary = cluster_summary.sort_values("avg_volume", ascending=False).reset_index(drop=True)
    cluster_summary["rank_label"] = [f"Cluster {i}" for i in range(n_clusters)]
    
    return binned_df, cluster_summary, km, scaler


if __name__ == "__main__":
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
