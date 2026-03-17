"""
visualizations.py
-----------------
All the charts for the project in one place.
Clean, publication-ready figures using matplotlib/seaborn.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── consistent style across all figures ──────────────────────────────────────
PALETTE = {
    "primary":   "#1a73e8",
    "secondary": "#ea4335",
    "accent":    "#34a853",
    "warn":      "#fbbc04",
    "neutral":   "#5f6368",
    "bg":        "#0d1117",
    "surface":   "#161b22",
    "border":    "#30363d",
    "text":      "#e6edf3",
    "subtext":   "#8b949e",
}

def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["surface"],
        "axes.edgecolor":    PALETTE["border"],
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["subtext"],
        "ytick.color":       PALETTE["subtext"],
        "text.color":        PALETTE["text"],
        "grid.color":        PALETTE["border"],
        "grid.alpha":        0.5,
        "legend.facecolor":  PALETTE["surface"],
        "legend.edgecolor":  PALETTE["border"],
        "font.family":       "monospace",
        "font.size":         9,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.titlepad":     10,
    })

set_dark_style()


# ─────────────────────────────────────────────
# Figure 1: EGARCH fit + price series
# ─────────────────────────────────────────────

def plot_egarch_fit(df, egarch_model, save_path="outputs/fig1_egarch_fit.png"):
    fig = plt.figure(figsize=(14, 10), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(3, 1, hspace=0.4)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    dates = df["date"].values
    
    # ── Price ────────────────────────────────────────────────────────────────
    ax1.plot(dates, df["price"], color=PALETTE["primary"], lw=0.8, alpha=0.9)
    ax1.set_title("Synthetic Price Series")
    ax1.set_ylabel("Price ($)")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}"))
    
    # ── Returns ──────────────────────────────────────────────────────────────
    colors = [PALETTE["secondary"] if r < 0 else PALETTE["accent"] for r in df["log_ret"]]
    ax2.bar(dates, df["log_ret"] * 100, color=colors, alpha=0.6, width=0.8)
    ax2.axhline(0, color=PALETTE["neutral"], lw=0.5)
    ax2.set_title("Log Returns (%)")
    ax2.set_ylabel("Return (%)")
    ax2.grid(True, alpha=0.3)
    
    # ── EGARCH conditional volatility ────────────────────────────────────────
    ax3.fill_between(dates, 0, egarch_model.fitted_vol * 100,
                     color=PALETTE["warn"], alpha=0.4, label="EGARCH σ")
    ax3.plot(dates, egarch_model.fitted_vol * 100,
             color=PALETTE["warn"], lw=0.8, alpha=0.8)
    if "true_vol" in df.columns:
        ax3.plot(dates, df["true_vol"] * 100,
                 color=PALETTE["neutral"], lw=0.7, alpha=0.6,
                 linestyle="--", label="True σ (DGP)")
    
    p = egarch_model.params
    ax3.set_title(
        f"EGARCH(1,1) Conditional Volatility  "
        f"[ω={p['omega']:.3f}, α={p['alpha']:.3f}, γ={p['gamma']:.3f}, β={p['beta']:.3f}]"
    )
    ax3.set_ylabel("Daily Vol (%)")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle("Time Series Analysis: EGARCH Volatility Modeling",
                 fontsize=14, y=1.01, color=PALETTE["text"], fontweight="bold")
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# Figure 2: 30-day forecast fan chart
# ─────────────────────────────────────────────

def plot_forecast(df, forecast_result, save_path="outputs/fig2_forecast.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=PALETTE["bg"])
    
    # ── Left: price fan ──────────────────────────────────────────────────────
    ax = axes[0]
    
    # Historical tail (last 60 days)
    hist = df.tail(60)
    ax.plot(hist["date"], hist["price"],
            color=PALETTE["primary"], lw=1.2, label="Historical", zorder=3)
    
    # Forecast horizon
    last_date = df["date"].iloc[-1]
    fcast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                periods=30, freq="B")
    
    # Draw a subset of paths (semi-transparent)
    n_show = 80
    sample_paths = forecast_result["paths"][:n_show]
    for path in sample_paths:
        ax.plot(fcast_dates, path,
                color=PALETTE["primary"], alpha=0.04, lw=0.5)
    
    # Confidence bands
    ax.fill_between(fcast_dates,
                    forecast_result["lower5"], forecast_result["upper95"],
                    color=PALETTE["primary"], alpha=0.15, label="90% band")
    ax.fill_between(fcast_dates,
                    forecast_result["lower25"], forecast_result["upper75"],
                    color=PALETTE["primary"], alpha=0.25, label="50% band")
    ax.plot(fcast_dates, forecast_result["median"],
            color=PALETTE["accent"], lw=1.5, label="Median forecast", zorder=4)
    
    # Mark the split
    ax.axvline(last_date, color=PALETTE["warn"],
               lw=1.0, linestyle="--", alpha=0.8, label="Forecast start")
    
    ax.set_title("30-Day Price Forecast (Monte Carlo)")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ── Right: volatility forecast ────────────────────────────────────────
    ax2 = axes[1]
    
    vf = forecast_result["vol_forecast"] * 100
    days = np.arange(1, 31)
    
    ax2.fill_between(days, 0, vf, color=PALETTE["warn"], alpha=0.4)
    ax2.plot(days, vf, color=PALETTE["warn"], lw=1.5, marker="o", ms=3)
    ax2.axhline(np.mean(vf), color=PALETTE["neutral"],
                linestyle="--", lw=1, label=f"Mean σ = {np.mean(vf):.4f}%")
    
    ax2.set_title("EGARCH Volatility Forecast (30 Days)")
    ax2.set_xlabel("Forecast horizon (days)")
    ax2.set_ylabel("Conditional Vol (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    sigma_daily = np.mean(vf) / 100
    fig.suptitle(
        f"30-Day Forecast  |  Mean σ = {sigma_daily:.4f}  |  "
        f"Median exit: ${forecast_result['median'][-1]:.2f}",
        fontsize=12, color=PALETTE["text"], fontweight="bold"
    )
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# Figure 3: VaR / CVaR risk dashboard
# ─────────────────────────────────────────────

def plot_risk_metrics(returns, var, cvar, save_path="outputs/fig3_risk.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=PALETTE["bg"])
    
    # ── Left: return distribution with VaR/CVaR marked ───────────────────────
    ax = axes[0]
    
    n_bins = 80
    counts, bin_edges, patches = ax.hist(
        returns * 100, bins=n_bins,
        color=PALETTE["primary"], alpha=0.6, edgecolor="none"
    )
    
    # Color the tail red
    for patch, left in zip(patches, bin_edges[:-1]):
        if left < var * 100:
            patch.set_facecolor(PALETTE["secondary"])
            patch.set_alpha(0.8)
    
    # VaR line
    ax.axvline(var * 100, color=PALETTE["secondary"], lw=1.5,
               linestyle="--", label=f"VaR 95% = {var*100:.2f}%")
    # CVaR line
    ax.axvline(cvar * 100, color=PALETTE["warn"], lw=1.5,
               linestyle=":", label=f"CVaR 95% = {cvar*100:.2f}%")
    
    # Normal fit overlay
    from scipy.stats import norm as sp_norm
    x_fit = np.linspace(returns.min() * 100, returns.max() * 100, 300)
    y_fit = sp_norm.pdf(x_fit, returns.mean() * 100, returns.std() * 100)
    scale = counts.max() / y_fit.max()
    ax.plot(x_fit, y_fit * scale,
            color=PALETTE["neutral"], lw=1, linestyle="-", alpha=0.7,
            label="Normal fit (for comparison)")
    
    ax.set_title("Return Distribution with Tail Risk")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ── Right: rolling 21-day VaR ─────────────────────────────────────────────
    ax2 = axes[1]
    
    ret_series = pd.Series(returns)
    rolling_var  = ret_series.rolling(21).quantile(0.05) * 100
    rolling_cvar = ret_series.rolling(21).apply(
        lambda x: x[x <= np.percentile(x, 5)].mean(), raw=True
    ) * 100
    
    ax2.fill_between(range(len(rolling_var)),
                     rolling_var, 0,
                     color=PALETTE["secondary"], alpha=0.3, label="Rolling VaR (21d)")
    ax2.fill_between(range(len(rolling_cvar)),
                     rolling_cvar, rolling_var,
                     color=PALETTE["warn"], alpha=0.3, label="CVaR excess")
    ax2.plot(rolling_var.values,  color=PALETTE["secondary"], lw=0.8)
    ax2.plot(rolling_cvar.values, color=PALETTE["warn"],      lw=0.8)
    
    ax2.set_title("Rolling 21-Day VaR & CVaR")
    ax2.set_xlabel("Trading day")
    ax2.set_ylabel("Return (%)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(
        f"Risk Metrics  |  VaR (95%) = {var*100:.2f}%  |  CVaR (95%) = {cvar*100:.2f}%",
        fontsize=12, color=PALETTE["text"], fontweight="bold"
    )
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# Figure 4: HFT activity — binned heatmap & volume
# ─────────────────────────────────────────────

def plot_hft_activity(binned_df, save_path="outputs/fig4_hft_activity.png"):
    fig = plt.figure(figsize=(16, 12), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
    
    ax1 = fig.add_subplot(gs[0, :])  # full width top row
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    binned = binned_df.copy()
    
    # ── Volume over time ──────────────────────────────────────────────────────
    ax1.fill_between(binned.index, 0, binned["volume"] / 1e6,
                     color=PALETTE["primary"], alpha=0.5)
    ax1.plot(binned.index, binned["volume"] / 1e6,
             color=PALETTE["primary"], lw=0.4, alpha=0.8)
    
    top5 = binned["volume"].nlargest(5).index
    for idx in top5:
        ax1.axvline(idx, color=PALETTE["warn"], lw=0.7, alpha=0.6)
    
    ax1.set_title("30-Minute Binned Trade Volume")
    ax1.set_xlabel("Bin index")
    ax1.set_ylabel("Volume (millions of shares)")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}M"))
    
    # ── Trade count histogram by time-of-day ──────────────────────────────────
    binned["time_of_day"] = pd.to_datetime(binned["bin"]).dt.time.astype(str)
    avg_by_time = binned.groupby(pd.to_datetime(binned["bin"]).dt.time)["trade_count"].mean()
    
    bar_colors = [PALETTE["accent"] if i < 3 or i > len(avg_by_time) - 4
                  else PALETTE["primary"]
                  for i in range(len(avg_by_time))]
    ax2.bar(range(len(avg_by_time)), avg_by_time.values,
            color=bar_colors, alpha=0.8, edgecolor="none")
    ax2.set_title("Avg Trade Count by Time of Day")
    ax2.set_xlabel("Half-hour slot (0 = 9:30am)")
    ax2.set_ylabel("Average trades")
    ax2.grid(True, alpha=0.3, axis="y")
    
    # ── Order flow imbalance distribution ─────────────────────────────────────
    ofi_vals = binned["ofi"].dropna()
    ax3.hist(ofi_vals, bins=50, color=PALETTE["accent"],
             alpha=0.7, edgecolor="none", density=True)
    ax3.axvline(0, color=PALETTE["neutral"], lw=1, linestyle="--")
    ax3.axvline(ofi_vals.mean(), color=PALETTE["warn"],
                lw=1.5, linestyle="--",
                label=f"Mean OFI = {ofi_vals.mean():.3f}")
    ax3.set_title("Order Flow Imbalance Distribution")
    ax3.set_xlabel("OFI  (−1 = all sells, +1 = all buys)")
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle(
        f"HFT Market Activity  |  {len(binned):,} 30-Min Bins  |  "
        f"Peak volume: {binned['volume'].max()/1e6:.1f}M shares",
        fontsize=12, color=PALETTE["text"], fontweight="bold"
    )
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# Figure 5: K-means cluster analysis
# ─────────────────────────────────────────────

def plot_clusters(binned_df, cluster_summary, save_path="outputs/fig5_clusters.png"):
    fig = plt.figure(figsize=(16, 10), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(2, 3, hspace=0.5, wspace=0.4)
    
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    
    n_clusters = cluster_summary["cluster"].nunique() if "cluster" in cluster_summary.columns else len(cluster_summary)
    cmap = plt.cm.get_cmap("tab10", n_clusters)
    
    cluster_colors = {
        int(row["cluster"]): cmap(i)
        for i, row in cluster_summary.iterrows()
        if "cluster" in cluster_summary.columns and i < n_clusters
    }
    
    # fallback: map label to color by rank
    if "cluster" not in cluster_summary.columns:
        # use binned_df cluster column directly
        unique_clusters = sorted(binned_df["cluster"].unique())
        cluster_colors = {c: cmap(i) for i, c in enumerate(unique_clusters)}
    
    colors_series = binned_df["cluster"].map(cluster_colors)
    
    # ── Scatter: volume vs trade count ────────────────────────────────────────
    sample = binned_df.sample(min(3000, len(binned_df)), random_state=1)
    sc_colors = sample["cluster"].map(cluster_colors)
    
    axes[0].scatter(sample["trade_count"], sample["volume"] / 1e6,
                    c=sc_colors, s=6, alpha=0.5, edgecolors="none")
    axes[0].set_xlabel("Trade count")
    axes[0].set_ylabel("Volume (M)")
    axes[0].set_title("Trade Count vs Volume")
    axes[0].grid(True, alpha=0.3)
    
    # ── Scatter: OFI vs range ─────────────────────────────────────────────────
    axes[1].scatter(sample["ofi"], sample["range_pct"] * 100,
                    c=sc_colors, s=6, alpha=0.5, edgecolors="none")
    axes[1].set_xlabel("Order Flow Imbalance")
    axes[1].set_ylabel("Price Range (%)")
    axes[1].set_title("OFI vs Intraday Volatility")
    axes[1].grid(True, alpha=0.3)
    
    # ── Cluster volume bars ────────────────────────────────────────────────────
    cs = cluster_summary.head(n_clusters)
    bar_c = [cmap(i) for i in range(len(cs))]
    axes[2].bar(cs.index, cs["total_volume"] / 1e9,
                color=bar_c, alpha=0.85, edgecolor="none")
    axes[2].set_xticks(cs.index)
    axes[2].set_xticklabels(
        cs["rank_label"] if "rank_label" in cs.columns else [f"C{i}" for i in cs.index],
        rotation=30, ha="right", fontsize=7
    )
    axes[2].set_ylabel("Total Volume (B)")
    axes[2].set_title("Total Volume by Cluster")
    axes[2].grid(True, alpha=0.3, axis="y")
    
    # ── Cluster avg trade count ────────────────────────────────────────────────
    axes[3].barh(cs.index, cs["avg_trades"],
                 color=bar_c, alpha=0.85, edgecolor="none")
    axes[3].set_yticks(cs.index)
    axes[3].set_yticklabels(
        cs["rank_label"] if "rank_label" in cs.columns else [f"C{i}" for i in cs.index],
        fontsize=7
    )
    axes[3].set_xlabel("Avg trades per bin")
    axes[3].set_title("Activity Intensity by Cluster")
    axes[3].grid(True, alpha=0.3, axis="x")
    
    # ── Avg OFI per cluster ────────────────────────────────────────────────────
    ofi_colors = [PALETTE["accent"] if v > 0 else PALETTE["secondary"]
                  for v in cs["avg_ofi"]]
    axes[4].bar(cs.index, cs["avg_ofi"],
                color=ofi_colors, alpha=0.85, edgecolor="none")
    axes[4].axhline(0, color=PALETTE["neutral"], lw=0.8)
    axes[4].set_xticks(cs.index)
    axes[4].set_xticklabels(
        cs["rank_label"] if "rank_label" in cs.columns else [f"C{i}" for i in cs.index],
        rotation=30, ha="right", fontsize=7
    )
    axes[4].set_ylabel("Avg OFI")
    axes[4].set_title("Order Flow by Cluster")
    axes[4].grid(True, alpha=0.3, axis="y")
    
    # ── Cluster size (number of bins) ─────────────────────────────────────────
    axes[5].pie(cs["n_bins"], labels=cs["rank_label"] if "rank_label" in cs.columns else None,
                colors=bar_c, autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 7, "color": PALETTE["text"]},
                wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 1.5})
    axes[5].set_title("Bin Distribution by Cluster")
    
    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=cmap(i), label=row["rank_label"])
        for i, (_, row) in enumerate(cs.iterrows())
        if "rank_label" in row
    ]
    if legend_patches:
        fig.legend(handles=legend_patches, loc="lower center",
                   ncol=n_clusters, fontsize=8,
                   framealpha=0.3, bbox_to_anchor=(0.5, -0.02))
    
    top_vol_row = cs.iloc[0]
    top_vol_label = top_vol_row.get("rank_label", "Cluster 0")
    top_vol = top_vol_row["total_volume"]
    
    fig.suptitle(
        f"K-Means Cluster Analysis  |  {n_clusters} Regimes  |  "
        f"{top_vol_label}: {top_vol/1e9:.2f}B volume",
        fontsize=12, color=PALETTE["text"], fontweight="bold"
    )
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# Figure 6: P&L attribution by strategy
# ─────────────────────────────────────────────

def plot_pnl_attribution(trades_df, save_path="outputs/fig6_pnl.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=PALETTE["bg"])
    
    strat_colors = {
        "market_maker": PALETTE["accent"],
        "momentum":     PALETTE["primary"],
        "noise":        PALETTE["secondary"],
    }
    
    # ── Cumulative P&L per strategy ────────────────────────────────────────────
    ax = axes[0]
    for strat, color in strat_colors.items():
        subset = trades_df[trades_df["strategy"] == strat]["pnl"]
        # sample 5000 points for plotting (the full 5M line would be invisible)
        step = max(1, len(subset) // 5000)
        cumulative = subset.cumsum().iloc[::step].values
        ax.plot(cumulative / 1e6, color=color, lw=1.0, alpha=0.85, label=strat.replace("_", " ").title())
    
    ax.set_title("Cumulative P&L by Strategy")
    ax.set_xlabel("Trade sample index")
    ax.set_ylabel("Cumulative P&L (M units)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}M"))
    
    # ── Total P&L breakdown (bar) ─────────────────────────────────────────────
    ax2 = axes[1]
    pnl_totals = trades_df.groupby("strategy")["pnl"].sum() / 1e6
    bars = ax2.bar(
        [s.replace("_", "\n").title() for s in pnl_totals.index],
        pnl_totals.values,
        color=[strat_colors[s] for s in pnl_totals.index],
        alpha=0.85, edgecolor="none", width=0.5
    )
    
    for bar, val in zip(bars, pnl_totals.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}M", ha="center", va="bottom", fontsize=8,
                 color=PALETTE["text"])
    
    ax2.set_title("Total P&L by Strategy")
    ax2.set_ylabel("Total P&L (M units)")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}M"))
    
    # ── P&L per trade distribution ────────────────────────────────────────────
    ax3 = axes[2]
    for strat, color in strat_colors.items():
        subset = trades_df[trades_df["strategy"] == strat]["pnl"]
        # Clip extreme outliers for display
        clipped = subset.clip(subset.quantile(0.01), subset.quantile(0.99))
        ax3.hist(clipped, bins=60, color=color, alpha=0.5,
                 density=True, label=strat.replace("_", " ").title(), edgecolor="none")
    
    ax3.set_title("P&L per Trade Distribution")
    ax3.set_xlabel("P&L per trade (units)")
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axvline(0, color=PALETTE["neutral"], lw=1, linestyle="--")
    
    total_pnl = trades_df["pnl"].sum()
    fig.suptitle(
        f"P&L Attribution  |  Total: {total_pnl/1e6:.0f}M units  |  "
        f"{len(trades_df):,} trades",
        fontsize=12, color=PALETTE["text"], fontweight="bold"
    )
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close()
    print(f"  Saved: {save_path}")
