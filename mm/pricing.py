"""Black-Scholes pricing and Greeks.

Vectorised over strikes so the market maker can reprice a whole book each step.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr  # fast normal CDF (no scipy.stats per-call overhead)

SQRT_EPS = 1e-12
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _norm_pdf(x):
    """Standard normal PDF. Inlined; identical to scipy.stats.norm.pdf but faster."""
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _d1_d2(S, K, T, r, sigma):
    T = np.maximum(T, SQRT_EPS)
    sigma = np.maximum(sigma, SQRT_EPS)
    vol_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    return d1, d2


def price(S, K, T, r, sigma, kind="call"):
    """European option price. `kind` is 'call' or 'put'."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    disc = np.exp(-r * np.maximum(T, 0.0))
    if kind == "call":
        return S * ndtr(d1) - K * disc * ndtr(d2)
    return K * disc * ndtr(-d2) - S * ndtr(-d1)


def greeks(S, K, T, r, sigma, kind="call"):
    """Delta, gamma, vega, theta.

    vega is per 1.00 of vol (not per vol point); theta is per year.
    Callers scale to their own units.
    """
    T = np.maximum(T, SQRT_EPS)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    pdf_d1 = _norm_pdf(d1)
    disc = np.exp(-r * T)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T

    if kind == "call":
        delta = ndtr(d1)
        theta = -(S * pdf_d1 * sigma) / (2 * sqrt_T) - r * K * disc * ndtr(d2)
    else:
        delta = ndtr(d1) - 1.0
        theta = -(S * pdf_d1 * sigma) / (2 * sqrt_T) + r * K * disc * ndtr(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def gbm_path(S0, mu, sigma, T, n_steps, rng):
    """Geometric Brownian motion path, length n_steps + 1."""
    dt = T / n_steps
    z = rng.standard_normal(n_steps)
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_path = np.concatenate([[np.log(S0)], np.log(S0) + np.cumsum(log_increments)])
    return np.exp(log_path)


def stochastic_vol_path(sigma0, vol_of_vol, kappa, T, n_steps, rng):
    """Mean-reverting (log) vol path.

    The maker quotes a fixed vol; if realised vol wanders away from it the maker
    is systematically long or short gamma at the wrong price. That mismatch is
    the main risk in this model.
    """
    dt = T / n_steps
    log_sig = np.log(sigma0)
    a = 1.0 - kappa * dt
    shocks = vol_of_vol * np.sqrt(dt) * rng.standard_normal(n_steps)
    x = np.empty(n_steps + 1)
    x[0] = log_sig
    # AR(1) recursion; kept explicit because vectorising the cumulative form
    # loses accuracy when kappa*dt is not small.
    for i in range(1, n_steps + 1):
        x[i] = a * x[i - 1] + kappa * dt * log_sig + shocks[i - 1]
    return np.exp(x)


def gbm_path_stochastic_vol(S0, mu, sigma_path, T, n_steps, rng):
    """GBM whose instantaneous vol follows `sigma_path`."""
    dt = T / n_steps
    z = rng.standard_normal(n_steps)
    sig = sigma_path[:n_steps]
    log_increments = (mu - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * z
    log_path = np.concatenate([[np.log(S0)], np.log(S0) + np.cumsum(log_increments)])
    return np.exp(log_path)
