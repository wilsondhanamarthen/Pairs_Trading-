# Cointegration-Based Statistical Arbitrage Engine

A fully systematic pairs trading system that identifies cointegrated stock pairs and generates mean-reversion signals. Built with a strict walk-forward methodology — pairs are selected on training data only and evaluated entirely out-of-sample.

---

## Results (Out-of-Sample: 2021–2023)

| Pair | Total PnL | Sharpe | Win Rate | Trades |
|---|---|---|---|---|
| SBUX/SLB | $4,907 | 0.706 | 82.3% | 17 |
| AMD/WEN | $9,546 | 0.434 | 66.7% | 15 |
| MCD/WEN | $1,338 | 0.308 | 73.3% | 15 |
| DPZ/WEN | $1,498 | 0.195 | 52.6% | 19 |
| GOOGL/META | -$2,462 | -0.350 | 46.7% | 15 |
| **Portfolio** | **$14,827** | **0.524** | — | **81** |

- 496 pairs tested → 55 passed cointegration → top 5 traded
- Portfolio max drawdown: $11,469 (5.07%)

---

## Strategy Overview

**Core idea:** Two economically related stocks maintain a long-run equilibrium. When their spread temporarily diverges, bet on it reverting.

**Signal generation:**
- Compute spread: `spread = log(P_A) - β × log(P_B)`
- Standardise to z-score using a 60-day rolling window
- Enter when `|z| > 2.0`, exit when `|z| < 0.5`, stop-loss at `|z| > 3.5`

**Position sizing:**
- $10,000 notional per leg
- Dollar-neutral: long one stock, short the other
- 10 bps transaction cost per trade

---

## Project Structure

```
pairs_trading/
├── data.py            # Fetch and clean adjusted close prices (yfinance)
├── cointegration.py   # Engle-Granger cointegration test, hedge ratio, half-life
├── signals.py         # Rolling z-score signal generation, position state machine
├── backtest.py        # Trade simulation, PnL calculation, transaction costs
├── metrics.py         # Sharpe, Sortino, max drawdown, win rate, profit factor
└── main.py            # Walk-forward pipeline + 4-panel results visualisation
```

---

## Methodology

### Walk-Forward Validation
| Period | Dates | Purpose |
|---|---|---|
| Training | 2015–2020 | Pair selection + hedge ratio estimation |
| Testing | 2021–2023 | Out-of-sample backtest (model never sees this) |

This is critical. In-sample backtests are almost always overfit. A strategy that works out-of-sample is actually meaningful.

### Cointegration Testing (Engle-Granger)
**Step 1** — OLS regression to estimate hedge ratio β:
```
log(P_A) = α + β·log(P_B) + ε
```
**Step 2** — ADF test on residuals ε. If stationary (p < 0.05), the pair is cointegrated.

**Half-life filter:** Only pairs with mean-reversion half-life between 5–126 days are kept. Too fast = noise. Too slow = untradeable.

### Universe
32 liquid US equities across 4 sectors:
- **Consumer:** KO, PEP, MCD, SBUX, YUM, DPZ, CMG, WEN
- **Tech:** MSFT, AAPL, GOOGL, META, NVDA, AMD, INTC, QCOM
- **Energy:** XOM, CVX, COP, EOG, SLB, MPC, VLO, PSX
- **Financials:** JPM, BAC, WFC, GS, MS, C, USB, PNC

---

## Installation

```bash
pip install yfinance pandas numpy statsmodels matplotlib
```

## Usage

```bash
python main.py
```

Output:
- Terminal: cointegrated pairs, performance metrics per pair, portfolio summary
- `pairs_trading_results.png`: 4-panel results chart

To change the universe:
```python
# main.py
SECTOR = "consumer"   # single sector
SECTOR = None         # all 32 stocks (496 pairs tested)
```

---

## Key Metrics Explained

| Metric | What it means |
|---|---|
| Sharpe Ratio | Return per unit of risk. Above 0.5 is respectable for a simple strategy |
| Sortino Ratio | Like Sharpe but only penalises downside volatility |
| Max Drawdown | Largest peak-to-trough loss in dollar terms |
| Win Rate | Fraction of trades that were profitable |
| Profit Factor | Gross wins / gross losses. Above 1.0 = profitable |
| Half-Life | Days for spread to revert halfway to mean |

---

## Limitations

- No shorting constraints or borrowing costs modelled
- Transaction cost model (10 bps flat) is a simplification
- Cross-sector pairs (e.g. SBUX/SLB) may be spuriously cointegrated
- Past cointegration does not guarantee future cointegration

---

## Tech Stack

- **Python 3.10+**
- **yfinance** — market data
- **statsmodels** — OLS regression, ADF test
- **pandas / numpy** — data manipulation
- **matplotlib** — visualisation
