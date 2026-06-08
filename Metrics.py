import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict
from Backtest import BacktestResult

TRADING_DAYS = 252  

@dataclass
class PerformanceMetrics:
    pair: str
    total_pnl: float 
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    n_trades: int
    avg_holding_days: float

    def __str__(self):
        return (
            f"\n{'─'*50}\n"
            f"Pair:               {self.pair}\n"
            f"Total PnL:          ${self.total_pnl:,.2f}\n"
            f"Annualised Return:  {self.annualized_return:.2f}%\n"
            f"Sharpe Ratio:       {self.sharpe_ratio:.3f}\n"
            f"Sortino Ratio:      {self.sortino_ratio:.3f}\n"
            f"Max Drawdown:       ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2f}%)\n"
            f"Win Rate:           {self.win_rate:.1%}\n"
            f"Avg Win:            ${self.avg_win:,.2f}\n"
            f"Avg Loss:           ${self.avg_loss:,.2f}\n"
            f"Profit Factor:      {self.profit_factor:.2f}\n"
            f"N Trades:           {self.n_trades}\n"
            f"Avg Holding (days): {self.avg_holding_days:.1f}\n"
            f"{'─'*50}"
        )

def compute_sharpe(daily_pnl: pd.Series, risk_free: float = 0.0) -> float:
    excess = daily_pnl - risk_free / TRADING_DAYS
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)

def compute_sortino(daily_pnl: pd.Series, risk_free: float = 0.0) -> float:
    excess = daily_pnl - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    if downside.std() == 0:
        return 0.0
    return (excess.mean() / downside.std()) * np.sqrt(TRADING_DAYS)

def compute_max_drawdown(cumulative_pnl: pd.Series)-> tuple:

    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    max_dd = drawdown.min()

    peak = running_max[drawdown.idxmin()]
    max_dd_pct = -max_dd / peak if peak != 0 else 0.0

    return max_dd, max_dd_pct

def compute_trade_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0, "avg_holding_days": 0.0,
        }
 
    wins   = trades[trades["pnl"] > 0]["pnl"]
    losses = trades[trades["pnl"] <= 0]["pnl"]
 
    win_rate      = len(wins) / len(trades)
    avg_win       = wins.mean()   if len(wins)   > 0 else 0.0
    avg_loss      = losses.mean() if len(losses) > 0 else 0.0
    gross_wins    = wins.sum()    if len(wins)   > 0 else 0.0
    gross_losses  = abs(losses.sum()) if len(losses) > 0 else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else np.inf
 
    # Holding period
    if "entry_date" in trades.columns and "exit_date" in trades.columns:
        holding = (
            pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])
        ).dt.days.mean()
    else:
        holding = 0.0
 
    return {
        "win_rate":         win_rate,
        "avg_win":          avg_win,
        "avg_loss":         avg_loss,
        "profit_factor":    profit_factor,
        "avg_holding_days": holding,
    }
 
 
def evaluate(result: BacktestResult) -> PerformanceMetrics:
    """Full performance evaluation for a single BacktestResult."""
    pnl        = result.pnl
    cum_pnl    = result.cumulative_pnl
    max_dd, max_dd_pct = compute_max_drawdown(cum_pnl)
    trade_stats        = compute_trade_stats(result.trades)
 
    n_days            = len(pnl)
    annualised_return = (pnl.sum() / n_days) * TRADING_DAYS / 100  
 
    return PerformanceMetrics(
        pair               = result.pair,
        total_pnl          = round(pnl.sum(), 2),
        annualized_return  = round(annualised_return, 4),
        sharpe_ratio       = round(compute_sharpe(pnl), 4),
        sortino_ratio      = round(compute_sortino(pnl), 4),
        max_drawdown       = round(max_dd, 2),
        max_drawdown_pct   = round(max_dd_pct, 2),
        win_rate           = round(trade_stats["win_rate"], 4),
        avg_win            = round(trade_stats["avg_win"], 2),
        avg_loss           = round(trade_stats["avg_loss"], 2),
        profit_factor      = round(trade_stats["profit_factor"], 4),
        n_trades           = result.n_trades,
        avg_holding_days   = round(trade_stats["avg_holding_days"], 1),
    )
 
 
def evaluate_portfolio(backtest_results: Dict[str, BacktestResult]) -> pd.DataFrame:
    rows = []
    for key, result in backtest_results.items():
        m = evaluate(result)
        rows.append({
            "pair":              m.pair,
            "total_pnl":         m.total_pnl,
            "sharpe":            m.sharpe_ratio,
            "sortino":           m.sortino_ratio,
            "max_dd":            m.max_drawdown,
            "max_dd_pct":        m.max_drawdown_pct,
            "win_rate":          f"{m.win_rate:.1%}",
            "profit_factor":     m.profit_factor,
            "n_trades":          m.n_trades,
            "avg_hold_days":     m.avg_holding_days,
        })
 
    summary = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
 
    all_pnl = pd.concat(
        [r.pnl for r in backtest_results.values()], axis=1
    ).sum(axis=1)
 
    port_sharpe = compute_sharpe(all_pnl)
    port_dd, port_dd_pct = compute_max_drawdown(all_pnl.cumsum())
 
    print(f"\n{'═'*50}")
    print(f"PORTFOLIO SUMMARY ({len(backtest_results)} pairs)")
    print(f"{'═'*50}")
    print(f"Total PnL:        ${all_pnl.sum():,.2f}")
    print(f"Portfolio Sharpe: {port_sharpe:.3f}")
    print(f"Portfolio Max DD: ${port_dd:,.2f} ({port_dd_pct:.2f}%)")
    print(f"{'═'*50}\n")
 
    return summary
