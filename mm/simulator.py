"""Options market-maker simulator.

The maker quotes a two-sided market on a strip of European options, gets hit by
Poisson customer flow, accumulates inventory, and delta-hedges on a fixed clock.

The point of the model is *risk*, not alpha: the maker earns the spread and pays
for it in gamma and hedging error. P&L is decomposed so you can see exactly where
the money came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .pricing import greeks, price, gbm_path, gbm_path_stochastic_vol, stochastic_vol_path


@dataclass
class Config:
    # underlying
    S0: float = 100.0
    mu: float = 0.0
    sigma_real: float = 0.20      # long-run realised vol of the path
    sigma_quote: float = 0.20     # fixed vol the maker prices with
    vol_of_vol: float = 0.9       # dispersion of realised vol (0 = constant vol)
    kappa: float = 4.0            # mean reversion speed of realised vol
    r: float = 0.0

    # book
    strikes: tuple = (90.0, 95.0, 100.0, 105.0, 110.0)
    T0: float = 0.25              # time to expiry at t=0, in years

    # session
    n_steps: int = 2_000

    # quoting
    half_spread_vol: float = 0.01   # half spread quoted in vol points
    arrival_rate: float = 0.35      # expected customer orders per step
    trade_size: float = 1.0
    max_inventory: float = 25.0     # per-option soft cap
    skew_strength: float = 0.6      # how hard quotes lean to shed inventory
    adverse_selection: float = 0.35 # 0 = uninformed flow, 1 = fully informed

    # hedging
    hedge_every: int = 10           # steps between delta hedges (1 = continuous-ish)
    hedge_cost_bps: float = 1.0     # cost per unit notional traded, in bps

    seed: int = 7


@dataclass
class Result:
    history: pd.DataFrame
    attribution: dict
    config: Config = field(repr=False, default=None)

    @property
    def total_pnl(self) -> float:
        return float(self.history["pnl"].iloc[-1])

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe of step P&L. Sessions are short, so treat as indicative."""
        d = self.history["pnl"].diff().dropna()
        if d.std() == 0:
            return 0.0
        steps_per_year = self.config.n_steps / self.config.T0
        return float(d.mean() / d.std() * np.sqrt(steps_per_year))

    @property
    def max_drawdown(self) -> float:
        p = self.history["pnl"].values
        return float((p - np.maximum.accumulate(p)).min())


def _fill_probability(half_spread_vol: float, arrival_rate: float) -> float:
    """Wider quotes get hit less often.

    Simple elastic model: tighter than the reference spread lifts fill odds,
    wider suppresses them.
    """
    reference = 0.01
    elasticity = np.exp(-(half_spread_vol - reference) / reference)
    return float(np.clip(arrival_rate * elasticity, 0.0, 1.0))


def run(cfg: Config) -> Result:
    rng = np.random.default_rng(cfg.seed)
    K = np.asarray(cfg.strikes, dtype=float)
    n_opts = len(K)

    if cfg.vol_of_vol > 0:
        vol_path = stochastic_vol_path(
            cfg.sigma_real, cfg.vol_of_vol, cfg.kappa, cfg.T0, cfg.n_steps, rng
        )
        S_path = gbm_path_stochastic_vol(
            cfg.S0, cfg.mu, vol_path, cfg.T0, cfg.n_steps, rng
        )
    else:
        vol_path = np.full(cfg.n_steps + 1, cfg.sigma_real)
        S_path = gbm_path(cfg.S0, cfg.mu, cfg.sigma_real, cfg.T0, cfg.n_steps, rng)
    dt = cfg.T0 / cfg.n_steps

    inventory = np.zeros(n_opts)      # signed option position, maker's book
    hedge_shares = 0.0                # underlying held against it
    cash = 0.0
    edge_captured = 0.0               # theoretical spread earned on fills
    hedge_costs = 0.0

    p_fill = _fill_probability(cfg.half_spread_vol, cfg.arrival_rate)

    rows = []
    attrib = {"theta": 0.0, "delta": 0.0, "gamma": 0.0, "hedge_error": 0.0}

    S_prev = S_path[0]
    T_prev = cfg.T0
    book_prev = price(S_prev, K, T_prev, cfg.r, cfg.sigma_quote)
    port_prev = float(inventory @ book_prev) + hedge_shares * S_prev + cash

    for t in range(1, cfg.n_steps + 1):
        S = S_path[t]
        T = max(cfg.T0 - t * dt, 1e-9)

        g_prev = greeks(S_prev, K, T_prev, cfg.r, cfg.sigma_quote)
        pos_delta_prev = float(inventory @ g_prev["delta"]) + hedge_shares
        pos_gamma_prev = float(inventory @ g_prev["gamma"])
        pos_theta_prev = float(inventory @ g_prev["theta"])

        dS = S - S_prev
        step_edge = 0.0

        # ---- customer flow -------------------------------------------------
        # Each option can trade at most once per step. Direction is random, but
        # the maker skews quotes to shed inventory, which biases which side fills.
        hits = rng.random(n_opts) < p_fill
        if hits.any():
            mid = price(S, K, T, cfg.r, cfg.sigma_quote)
            vega = greeks(S, K, T, cfg.r, cfg.sigma_quote)["vega"]
            edge_per_unit = vega * cfg.half_spread_vol   # spread in price terms

            # inventory skew: long book -> more likely to get lifted (we sell)
            util = np.clip(inventory / cfg.max_inventory, -1.0, 1.0)
            p_customer_buys = 0.5 + cfg.skew_strength * util

            # adverse selection: some of the flow knows where the underlying is
            # about to go. This is what makes market making hard -- without it the
            # maker just harvests spread and the Sharpe is fantasy.
            if cfg.adverse_selection > 0 and t < cfg.n_steps:
                informed_up = S_path[min(t + 1, cfg.n_steps)] > S
                tilt = cfg.adverse_selection * (0.45 if informed_up else -0.45)
                p_customer_buys = p_customer_buys + tilt

            p_customer_buys = np.clip(p_customer_buys, 0.05, 0.95)
            customer_buys = rng.random(n_opts) < p_customer_buys

            # maker takes the other side
            signed = np.where(customer_buys, -cfg.trade_size, cfg.trade_size)
            signed = np.where(hits, signed, 0.0)

            # respect the inventory cap
            would_be = inventory + signed
            blocked = np.abs(would_be) > cfg.max_inventory
            signed = np.where(blocked, 0.0, signed)

            traded = signed != 0
            inventory = inventory + signed
            # maker buys below mid / sells above mid -> always collects the edge
            step_edge = float(np.sum(edge_per_unit[traded] * cfg.trade_size))
            edge_captured += step_edge
            cash += step_edge
            # cash effect of the trade itself is carried in the mark-to-market
            cash -= float(np.sum(signed * mid))

        # ---- delta hedge ---------------------------------------------------
        if t % cfg.hedge_every == 0:
            g_now = greeks(S, K, T, cfg.r, cfg.sigma_quote)
            target = -float(inventory @ g_now["delta"])
            trade_shares = target - hedge_shares
            if trade_shares != 0.0:
                cost = abs(trade_shares) * S * cfg.hedge_cost_bps * 1e-4
                hedge_costs += cost
                cash -= trade_shares * S + cost
                hedge_shares = target

        # ---- mark to market -------------------------------------------------
        book = price(S, K, T, cfg.r, cfg.sigma_quote)
        port = float(inventory @ book) + hedge_shares * S + cash
        pnl_step = port - port_prev

        # ---- attribution ----------------------------------------------------
        theta_c = pos_theta_prev * dt
        delta_c = pos_delta_prev * dS
        gamma_c = 0.5 * pos_gamma_prev * dS**2
        explained = theta_c + delta_c + gamma_c
        attrib["theta"] += theta_c
        attrib["delta"] += delta_c
        attrib["gamma"] += gamma_c
        attrib["hedge_error"] += pnl_step - explained - step_edge

        g_end = greeks(S, K, T, cfg.r, cfg.sigma_quote)
        rows.append(
            {
                "t": t,
                "S": S,
                "pnl": port,
                "inventory_abs": float(np.abs(inventory).sum()),
                "net_delta": float(inventory @ g_end["delta"]) + hedge_shares,
                "hedge_shares": hedge_shares,
            }
        )

        S_prev, T_prev, port_prev = S, T, port

    attrib["edge_captured"] = edge_captured
    attrib["hedge_costs"] = -hedge_costs

    return Result(history=pd.DataFrame(rows), attribution=attrib, config=cfg)
