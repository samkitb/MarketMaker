"""Streamlit front end for the options market-maker simulator.

Run with:  streamlit run app.py
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from mm.simulator import Config, run
from mm.experiments import run_seeds, sweep

st.set_page_config(page_title="Options Market Maker", layout="wide")

st.title("Options market-maker simulator")
st.caption(
    "Quote a two-sided market on an option strip, get hit by informed and "
    "uninformed flow, hedge delta on a clock, and see where the P&L came from."
)

with st.sidebar:
    st.header("Market")
    sigma_real = st.slider("Realised vol (long-run)", 0.05, 0.60, 0.20, 0.01)
    sigma_quote = st.slider("Quoted vol", 0.05, 0.60, 0.20, 0.01)
    vol_of_vol = st.slider("Vol of vol", 0.0, 2.0, 0.9, 0.1)

    st.header("Quoting")
    half_spread = st.slider("Half spread (vol pts)", 0.001, 0.05, 0.01, 0.001)
    arrival = st.slider("Arrival rate", 0.05, 1.0, 0.35, 0.05)
    adverse = st.slider("Adverse selection", 0.0, 1.0, 0.35, 0.05)
    max_inv = st.slider("Max inventory", 5, 100, 25, 5)

    st.header("Hedging")
    hedge_every = st.select_slider(
        "Hedge every N steps", options=[1, 2, 5, 10, 25, 50, 100, 250], value=10
    )
    hedge_cost = st.slider("Hedge cost (bps)", 0.0, 10.0, 1.0, 0.5)

    st.header("Run")
    n_steps = st.select_slider("Steps", options=[300, 600, 1000, 2000], value=600)
    n_seeds = st.select_slider("Seeds (for distribution)", options=[10, 20, 40], value=20)

cfg = Config(
    sigma_real=sigma_real,
    sigma_quote=sigma_quote,
    vol_of_vol=vol_of_vol,
    half_spread_vol=half_spread,
    arrival_rate=arrival,
    adverse_selection=adverse,
    max_inventory=float(max_inv),
    hedge_every=int(hedge_every),
    hedge_cost_bps=hedge_cost,
    n_steps=int(n_steps),
)

res = run(cfg)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Session P&L", f"{res.total_pnl:,.1f}")
c2.metric("Max drawdown", f"{res.max_drawdown:,.1f}")
c3.metric("Edge captured", f"{res.attribution['edge_captured']:,.1f}")
c4.metric("Hedge costs", f"{res.attribution['hedge_costs']:,.1f}")

st.subheader("One session")
left, right = st.columns(2)
with left:
    st.line_chart(res.history.set_index("t")[["pnl"]], height=240)
    st.caption("Cumulative P&L")
with right:
    st.line_chart(res.history.set_index("t")[["S"]], height=240)
    st.caption("Underlying")

left, right = st.columns(2)
with left:
    st.line_chart(res.history.set_index("t")[["inventory_abs"]], height=220)
    st.caption("Gross option inventory")
with right:
    st.line_chart(res.history.set_index("t")[["net_delta"]], height=220)
    st.caption("Net delta after hedging (should hug zero)")

st.subheader("P&L attribution")
attrib = pd.DataFrame(
    {"component": list(res.attribution), "pnl": list(res.attribution.values())}
).set_index("component")
st.bar_chart(attrib, height=260)
st.caption(
    "Edge is what you charge for liquidity. Theta and gamma should roughly offset "
    "when quoted vol matches realised vol; they do not when it doesn't."
)

st.subheader("Across seeds")
st.caption(
    "A single path proves nothing. This runs the same configuration on independent "
    "seeds and reports the distribution."
)
if st.button("Run seed study"):
    with st.spinner(f"Running {n_seeds} seeds..."):
        stats = run_seeds(cfg, n_seeds=n_seeds)
    a, b, c, d = st.columns(4)
    a.metric("Mean P&L", f"{stats['mean_pnl']:,.1f}")
    b.metric("Std P&L", f"{stats['std_pnl']:,.1f}")
    c.metric("Mean / Std", f"{stats['ratio']:.2f}")
    d.metric("Losing sessions", f"{stats['loss_rate']:.0%}")
    st.write(
        f"5th percentile: {stats['p05']:,.1f} · 95th percentile: {stats['p95']:,.1f} "
        f"· mean max drawdown: {stats['mean_maxdd']:,.1f}"
    )

st.subheader("Hedge frequency tradeoff")
if st.button("Run hedge sweep"):
    with st.spinner("Sweeping hedge frequency..."):
        df = sweep(cfg, "hedge_every", [1, 5, 10, 25, 50, 100], n_seeds=n_seeds)
    st.dataframe(
        df[["hedge_every", "mean_pnl", "std_pnl", "ratio", "loss_rate", "mean_hedge_cost"]]
        .round(2),
        use_container_width=True,
    )
    st.line_chart(df.set_index("hedge_every")[["ratio"]], height=240)
    st.caption("Risk-adjusted return collapses as hedging gets sparser.")
