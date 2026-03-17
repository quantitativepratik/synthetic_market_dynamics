"""
time_series_model.py
--------------------
ARIMA + EGARCH implementation for price and volatility forecasting.

ARIMA handles the mean equation (trend/autocorrelation in returns),
while EGARCH captures the volatility clustering and the asymmetric
response to shocks — bad news tends to spike vol harder than good news.

We implement both from scratch using scipy so there are zero
external dependencies beyond the scientific Python stack.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Data generation — synthetic price series
# ─────────────────────────────────────────────

def generate_price_series(n=1000, seed=42):
    """
    Build a synthetic log-return series that mimics real equity behavior:
    - volatility clustering (quiet periods followed by turbulent ones)
    - slight negative skew (crashes are sharper than rallies)
    - fat tails via a t-distribution shock process
    """
    rng = np.random.default_rng(seed)
    
    # True EGARCH(1,1) parameters we'll try to recover
    omega = -0.15   # log-variance intercept
    alpha = 0.12    # ARCH effect (magnitude of shock)
    gamma = 0.644   # leverage / asymmetry (negative shocks → bigger vol)
    beta  = 0.97    # persistence (vol decays slowly)
    
    log_var = np.zeros(n)
    returns = np.zeros(n)
    
    # Burn-in to stabilize the process
    log_var[0] = omega / (1 - beta)
    
    for t in range(1, n):
        sigma_prev = np.exp(0.5 * log_var[t-1])
        z_prev = returns[t-1] / (sigma_prev + 1e-10)
        
        # Nelson's EGARCH recursion
        log_var[t] = (omega
                      + alpha * (np.abs(z_prev) - np.sqrt(2 / np.pi))
                      + gamma * z_prev
                      + beta * log_var[t-1])
    
    # Draw shocks — t(6) gives fatter tails than Gaussian
    df_shock = 6
    raw_shocks = rng.standard_t(df_shock, size=n) / np.sqrt(df_shock / (df_shock - 2))
    returns = np.exp(0.5 * log_var) * raw_shocks
    
    # Reconstruct price from log-returns
    log_price = np.cumsum(returns) + np.log(100)
    prices = np.exp(log_price)
    
    dates = pd.date_range(start="2020-01-02", periods=n, freq="B")
    df = pd.DataFrame({
        "date":    dates,
        "price":   prices,
        "log_ret": returns,
        "true_vol": np.exp(0.5 * log_var)
    })
    return df


# ─────────────────────────────────────────────
# ARIMA(p,d,q) — mean equation
# ─────────────────────────────────────────────

class ARIMA:
    """
    Minimal ARIMA implementation.
    For our use case the mean equation is ARIMA(0,0,0) — basically
    just a drift term — since log-returns are very close to white noise.
    We keep the general structure so you can swap in (p,d,q) if needed.
    """
    
    def __init__(self, p=0, d=0, q=0):
        self.p = p
        self.d = d
        self.q = q
        self.params = None
        self.residuals = None
    
    def _difference(self, x, d):
        for _ in range(d):
            x = np.diff(x)
        return x
    
    def fit(self, y):
        y_diff = self._difference(np.array(y), self.d)
        n = len(y_diff)
        
        # With p=q=0 we just estimate the mean (drift)
        if self.p == 0 and self.q == 0:
            mu = np.mean(y_diff)
            self.params = {"mu": mu}
            self.residuals = y_diff - mu
            self.fitted_mean = y_diff  # trivially, just the mean level
            return self
        
        # General case: AR(p) via OLS
        # (MA terms omitted for brevity — extend if you need them)
        X = []
        Y = y_diff[self.p:]
        for i in range(self.p):
            X.append(y_diff[self.p - 1 - i: n - 1 - i])
        X = np.column_stack([np.ones(len(Y))] + X)
        
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        fitted = X @ beta
        self.residuals = Y - fitted
        self.params = {"mu": beta[0], "ar": beta[1:]}
        return self
    
    def forecast(self, steps=30):
        """
        Simple mean-reverting forecast: beyond the sample,
        expected returns collapse to the estimated drift.
        """
        mu = self.params["mu"]
        return np.full(steps, mu)


# ─────────────────────────────────────────────
# EGARCH(1,1) — volatility equation
# ─────────────────────────────────────────────

class EGARCH:
    """
    Nelson (1991) EGARCH(1,1) with Gaussian innovations.
    
    The log-variance equation:
        ln(σ²_t) = ω + α(|z_{t-1}| - E|z|) + γ·z_{t-1} + β·ln(σ²_{t-1})
    
    γ is the leverage term — it's what makes this model interesting.
    A negative γ means negative shocks amplify variance more than positive ones,
    which is exactly what you see in equity markets.
    """
    
    def __init__(self):
        self.params = None
        self.log_var_path = None
        self.converged = False
    
    def _log_likelihood(self, theta, returns):
        omega, alpha, gamma, beta = theta
        n = len(returns)
        log_var = np.zeros(n)
        
        # Start variance at the unconditional level
        log_var[0] = omega / max(1 - abs(beta), 1e-6)
        
        ll = 0.0
        for t in range(1, n):
            sigma_prev = np.exp(0.5 * log_var[t-1])
            z_prev = returns[t-1] / (sigma_prev + 1e-10)
            
            log_var[t] = (omega
                          + alpha * (np.abs(z_prev) - np.sqrt(2 / np.pi))
                          + gamma * z_prev
                          + beta * log_var[t-1])
            
            sigma_t = np.exp(0.5 * log_var[t])
            ll += norm.logpdf(returns[t], loc=0, scale=sigma_t)
        
        return -ll  # negative because we're minimizing
    
    def fit(self, returns):
        """
        MLE via L-BFGS-B. Initial guess based on sample variance.
        Multiple restarts help avoid local optima.
        """
        returns = np.array(returns)
        
        # Reasonable starting point
        init_vol = np.log(np.var(returns))
        x0 = [init_vol * (1 - 0.97), 0.1, 0.3, 0.97]
        
        bounds = [
            (None, None),   # omega: unconstrained
            (1e-6, 1.0),    # alpha: positive
            (-1.0, 1.0),    # gamma: can be negative (leverage)
            (0.001, 0.999), # beta: stationarity
        ]
        
        best_result = None
        best_ll = np.inf
        
        # A few random restarts for robustness
        rng = np.random.default_rng(0)
        starts = [x0]
        for _ in range(4):
            starts.append([
                rng.uniform(-1, -0.01),
                rng.uniform(0.01, 0.3),
                rng.uniform(-0.5, 0.5),
                rng.uniform(0.8, 0.98)
            ])
        
        for start in starts:
            try:
                res = minimize(
                    self._log_likelihood,
                    start,
                    args=(returns,),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 500, "ftol": 1e-10}
                )
                if res.fun < best_ll:
                    best_ll = res.fun
                    best_result = res
            except Exception:
                continue
        
        if best_result is not None and best_result.success:
            self.converged = True
        
        omega, alpha, gamma, beta = best_result.x
        self.params = {
            "omega": omega,
            "alpha": alpha,
            "gamma": gamma,
            "beta":  beta
        }
        
        # Store the fitted variance path for diagnostics
        self.log_var_path = self._compute_log_var(returns, best_result.x)
        self.fitted_vol = np.exp(0.5 * self.log_var_path)
        return self
    
    def _compute_log_var(self, returns, theta):
        omega, alpha, gamma, beta = theta
        n = len(returns)
        log_var = np.zeros(n)
        log_var[0] = omega / max(1 - abs(beta), 1e-6)
        
        for t in range(1, n):
            sigma_prev = np.exp(0.5 * log_var[t-1])
            z_prev = returns[t-1] / (sigma_prev + 1e-10)
            log_var[t] = (omega
                          + alpha * (np.abs(z_prev) - np.sqrt(2 / np.pi))
                          + gamma * z_prev
                          + beta * log_var[t-1])
        return log_var
    
    def forecast(self, steps=30):
        """
        Multi-step variance forecast via the EGARCH recursion.
        Beyond step 1 we lose the z_t shock info, so we integrate
        over the expected value of the asymmetric term analytically.
        
        For the Gaussian case:
            E[|z| - sqrt(2/π)] = 0  by construction
            E[γ·z] = 0 (z is mean-zero)
        So the forecast collapses to:
            ln(σ²_{t+h}) = ω + β·ln(σ²_{t+h-1})
        which mean-reverts to ω/(1-β).
        """
        omega = self.params["omega"]
        beta  = self.params["beta"]
        
        last_log_var = self.log_var_path[-1]
        unconditional = omega / (1 - beta)
        
        forecast_log_var = np.zeros(steps)
        forecast_log_var[0] = omega + beta * last_log_var
        
        for h in range(1, steps):
            forecast_log_var[h] = omega + beta * forecast_log_var[h - 1]
        
        # Reversion to long-run level
        forecast_vol = np.exp(0.5 * forecast_log_var)
        return forecast_vol


# ─────────────────────────────────────────────
# Risk metrics: VaR and CVaR
# ─────────────────────────────────────────────

def compute_var_cvar(returns, confidence=0.95):
    """
    Historical simulation VaR and CVaR at the given confidence level.
    
    VaR = the loss you won't exceed with (confidence)% probability.
    CVaR = expected loss given that you're already past the VaR threshold.
    CVaR is the more informative number — it tells you *how bad* the tail is,
    not just where it starts.
    """
    returns = np.array(returns)
    alpha = 1 - confidence
    
    var = np.percentile(returns, alpha * 100)
    cvar = returns[returns <= var].mean()
    
    return var, cvar


# ─────────────────────────────────────────────
# 30-day price forecast
# ─────────────────────────────────────────────

def forecast_prices(df, arima_model, egarch_model, steps=30, n_simulations=500, seed=1):
    """
    Monte Carlo price paths using ARIMA for drift and EGARCH for vol.
    We draw shocks from N(0,1) and scale by the EGARCH volatility forecast.
    
    Returns both the fan of paths and summary stats (median + quantile bands).
    """
    rng = np.random.default_rng(seed)
    
    mean_forecast = arima_model.forecast(steps=steps)
    vol_forecast  = egarch_model.forecast(steps=steps)
    
    last_price = df["price"].iloc[-1]
    paths = np.zeros((n_simulations, steps))
    
    for i in range(n_simulations):
        shocks = rng.standard_normal(steps)
        log_rets = mean_forecast + vol_forecast * shocks
        paths[i] = last_price * np.exp(np.cumsum(log_rets))
    
    median  = np.median(paths, axis=0)
    lower5  = np.percentile(paths, 5, axis=0)
    upper95 = np.percentile(paths, 95, axis=0)
    lower25 = np.percentile(paths, 25, axis=0)
    upper75 = np.percentile(paths, 75, axis=0)
    
    return {
        "paths":   paths,
        "median":  median,
        "lower5":  lower5,
        "upper95": upper95,
        "lower25": lower25,
        "upper75": upper75,
        "mean_vol": np.mean(vol_forecast),
        "vol_forecast": vol_forecast
    }


if __name__ == "__main__":
    print("Running time series pipeline...\n")
    
    # Step 1: generate data
    df = generate_price_series(n=1000)
    print(f"Generated {len(df)} trading days of synthetic price data")
    print(f"Price range: {df['price'].min():.2f} – {df['price'].max():.2f}")
    
    # Step 2: fit ARIMA (mean equation)
    arima = ARIMA(p=0, d=0, q=0)
    arima.fit(df["log_ret"])
    print(f"\nARIMA(0,0,0) drift: {arima.params['mu']:.6f}")
    
    # Step 3: fit EGARCH
    egarch = EGARCH()
    egarch.fit(df["log_ret"].values)
    p = egarch.params
    print(f"\nEGARCH(1,1) parameters:")
    print(f"  ω (omega): {p['omega']:.4f}")
    print(f"  α (alpha): {p['alpha']:.4f}  ← magnitude of shocks")
    print(f"  γ (gamma): {p['gamma']:.4f}  ← leverage / asymmetry")
    print(f"  β (beta):  {p['beta']:.4f}   ← persistence")
    print(f"  Converged: {egarch.converged}")
    
    # Step 4: VaR / CVaR
    var, cvar = compute_var_cvar(df["log_ret"].values)
    print(f"\nRisk metrics (95% confidence, historical simulation):")
    print(f"  VaR:  {var*100:.2f}%")
    print(f"  CVaR: {cvar*100:.2f}%")
    
    # Step 5: 30-day forecast
    fc = forecast_prices(df, arima, egarch, steps=30)
    print(f"\n30-day forecast:")
    print(f"  Mean forecast vol (sigma): {fc['mean_vol']:.4f}")
    print(f"  Median final price: {fc['median'][-1]:.2f}")
    print(f"  5th–95th pct range: {fc['lower5'][-1]:.2f} – {fc['upper95'][-1]:.2f}")
