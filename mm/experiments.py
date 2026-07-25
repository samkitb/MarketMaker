"""Multi-seed sweeps.

A single simulated session tells you almost nothing: one path can look brilliant
by luck. Everything here runs N independent seeds and reports the *distribution*
of session P&L. The headline number is mean / std across seeds, which is the
honest risk-adjusted measure for this setup -- not an annualised Sharpe computed
from one path's step returns.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .simulator import Config, run


def run_seeds(cfg: Config, n_seeds: int = 40) -> dict:
    pnls, dds, gammas, thetas, edges, costs = [], [], [], [], [], []
    for s in range(n_seeds):
        r = run(replace(cfg, seed=1000 + s))
        pnls.append(r.total_pnl)
        dds.append(r.max_drawdown)
        gammas.append(r.attribution["gamma"])
        thetas.append(r.attribution["theta"])
        edges.append(r.attribution["edge_captured"])
        costs.append(r.attribution["hedge_costs"])
    pnls = np.array(pnls)
    return {
        "mean_pnl": pnls.mean(),
        "std_pnl": pnls.std(ddof=1),
        "ratio": pnls.mean() / pnls.std(ddof=1) if pnls.std(ddof=1) > 0 else np.nan,
        "p05": np.percentile(pnls, 5),
        "p95": np.percentile(pnls, 95),
        "loss_rate": float((pnls < 0).mean()),
        "mean_maxdd": float(np.mean(dds)),
        "mean_gamma": float(np.mean(gammas)),
        "mean_theta": float(np.mean(thetas)),
        "mean_edge": float(np.mean(edges)),
        "mean_hedge_cost": float(np.mean(costs)),
    }


def sweep(cfg: Config, param: str, values, n_seeds: int = 40) -> pd.DataFrame:
    rows = []
    for v in values:
        stats = run_seeds(replace(cfg, **{param: v}), n_seeds=n_seeds)
        stats[param] = v
        rows.append(stats)
    cols = [param] + [c for c in rows[0] if c != param]
    return pd.DataFrame(rows)[cols]


def hedge_frequency_study(cfg: Config = None, n_seeds: int = 40) -> pd.DataFrame:
    """The core tradeoff: hedging often kills variance but pays costs."""
    cfg = cfg or Config()
    return sweep(cfg, "hedge_every", [1, 2, 5, 10, 25, 50, 100, 250], n_seeds)


def spread_study(cfg: Config = None, n_seeds: int = 40) -> pd.DataFrame:
    """Wider quotes earn more per fill but trade less and hold more risk."""
    cfg = cfg or Config()
    return sweep(cfg, "half_spread_vol", [0.002, 0.005, 0.01, 0.02, 0.04], n_seeds)


def adverse_selection_study(cfg: Config = None, n_seeds: int = 40) -> pd.DataFrame:
    """How fast informed flow eats the spread."""
    cfg = cfg or Config()
    return sweep(cfg, "adverse_selection", [0.0, 0.2, 0.4, 0.6, 0.8], n_seeds)


def inventory_limit_study(cfg: Config = None, n_seeds: int = 40) -> pd.DataFrame:
    cfg = cfg or Config()
    return sweep(cfg, "max_inventory", [5, 10, 25, 50, 100], n_seeds)
