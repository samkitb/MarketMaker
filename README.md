# Options Market-Maker Simulator

A simulation of an options market maker: quote a two-sided market on a strip of
European options, get hit by a mix of informed and uninformed flow, carry the
resulting inventory, hedge delta on a fixed clock, and decompose the P&L.

The question this answers is not "can I predict prices" — it's **where does a
liquidity provider's money actually come from, and what takes it away.**

## Model

- **Underlying**: GBM with stochastic (mean-reverting log) volatility. The maker
  quotes a *fixed* vol, so realised vol drifting away from the quoted vol is a
  genuine source of risk rather than a free lunch.
- **Book**: 5 strikes, Black-Scholes priced, full Greeks recomputed each step.
- **Flow**: Poisson arrivals whose fill probability falls as the maker widens.
  Quotes are skewed by current inventory to shed risk.
- **Adverse selection**: a tunable fraction of the flow knows the next price
  move. This is the single most important parameter in the model.
- **Hedging**: delta hedged every *N* steps at a bps cost, so the discrete-hedging
  tradeoff is explicit.
- **Attribution**: session P&L split into edge captured, theta, gamma, hedging
  error, and hedge costs.

## Why the results are reported across seeds

A single simulated session is close to meaningless — one path can look excellent
by luck, and an annualised Sharpe computed from one path's step returns is
inflated by the steady accrual of spread. Every headline number here is the
distribution over independent seeds, summarised as mean / std of session P&L,
5th percentile, and the fraction of losing sessions.

## Findings

### 1. Discrete hedging: variance is bought, not free

20 seeds, 600 steps, adverse selection 0.35.

| hedge every | mean P&L | std | mean/std | 5th pct | losing sessions | hedge cost |
|---|---|---|---|---|---|---|
| 1 | 81.2 | 8.6 | **9.46** | 66.1 | 0% | −3.85 |
| 5 | 39.7 | 21.8 | 1.82 | 13.4 | 5% | −1.73 |
| 25 | 26.2 | 36.1 | 0.73 | −33.2 | 20% | −0.69 |
| 100 | 27.0 | 44.8 | 0.60 | −50.1 | 25% | −0.24 |
| 250 | 18.3 | 74.9 | **0.24** | −54.1 | 30% | −0.07 |

Hedging less often saves transaction costs — and it is still overwhelmingly the
wrong trade. Going from hedging every step to every 250 steps cuts hedge costs by
~98% but raises P&L dispersion **8.7×**, pushes mean max drawdown from −0.5 to
−60, and turns 0% losing sessions into 30%. Risk-adjusted return falls from 9.5
to 0.2.

### 2. Adverse selection is what actually kills the book

12 seeds, 400 steps. **Gross edge captured is identical (55.6) in every row** —
the maker charges exactly the same spread throughout.

| adverse selection | mean P&L | std | mean/std | losing sessions |
|---|---|---|---|---|
| 0.0 | 53.5 | 17.4 | 3.07 | 0% |
| 0.2 | 32.9 | 18.8 | 1.75 | 0% |
| 0.4 | 8.8 | 21.7 | 0.41 | 33% |
| 0.6 | −14.5 | 28.9 | −0.50 | 58% |
| 0.8 | −34.9 | 36.4 | −0.96 | **100%** |

The spread earned never changes; what changes is who is on the other side. Past
roughly 40% informed flow the business is structurally unprofitable no matter how
well it is hedged. Widening quotes is the only lever, and widening reduces fill
rates — which is the real tension in market making.

## Limitations (read before believing any of it)

- Single underlying, no smile: the maker quotes one vol across all strikes, so
  skew risk is absent.
- Flow model is synthetic. Arrival elasticity and the informed-trader tilt are
  assumptions, not calibrations, and results are sensitive to both.
- No queue position, latency, or partial fills — this is not a microstructure
  model of an actual order book.
- Costs are linear in size; real market impact is not.
- Adverse selection is implemented with one-step lookahead *for the simulated
  counterparty only*. The maker never sees the future.

## Layout

```
mm/pricing.py      Black-Scholes prices, Greeks, GBM and stochastic-vol paths
mm/simulator.py    quoting, inventory, hedging, P&L attribution
mm/experiments.py  multi-seed sweeps (hedge frequency, spread, adverse selection)
app.py             Streamlit UI
```

## Run

```bash
pip install numpy pandas scipy streamlit
streamlit run app.py
```

Or headless:

```python
from mm.simulator import Config, run
from mm.experiments import hedge_frequency_study

run(Config(hedge_every=10)).attribution
hedge_frequency_study(n_seeds=20)
```
