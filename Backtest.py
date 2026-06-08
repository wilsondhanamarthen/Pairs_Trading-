import numpy as np 
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from Cointegration import CointegrationResult 
from Signals import SignalConfig, generate_signals

@dataclass
class BacktestConfig:
    notional: float = 10_000.0 
    cost_bps: float = 10.0
    signal_config: SignalConfig = None

    def __post_init__(self):
        if self.signal_config is None:
            self.signal_config = SignalConfig()

    @property
    def cost_rate(self) -> float:
        return self.cost_bps / 10_000.0
    
@dataclass
class BacktestResult:
    pair: str 
    pnl: pd.Series
    trades: pd.DataFrame
    signals_df: pd.DataFrame

    @property 
    def cumulative_pnl(self) -> pd.Series:
        return self.pnl.cumsum()
    
    @property 
    def n_trades(self) -> int:
        return len(self.trades)
    
def run_backtest(
    prices: pd.DataFrame,
    result: CointegrationResult,
    config: BacktestConfig = BacktestConfig(),
    train_end: Optional[str] = None
)-> BacktestResult: 
    signals_df = generate_signals(prices, result, config.signal_config)
    
    signals_df["position"] = signals_df["position"].shift(1).fillna(0)

    if train_end: 
        signals_df = signals_df.loc[train_end:]

    daily_pnl = _compute_daily_pnl(signals_df, result, config)
    trades = _extract_trades(signals_df, result, config, daily_pnl)

    return BacktestResult(
        pair = f"{result.ticker_a} - {result.ticker_b}",
        pnl = daily_pnl,
        trades = trades,
        signals_df = signals_df
    )

def _compute_daily_pnl(
    df: pd.DataFrame,
    result: CointegrationResult,
    config: BacktestConfig
) -> pd.Series:
    
    position  = df["position"]
    spread = df["spread"]

    spread_ret = spread.diff().fillna(0)
    gross_pnl = position.shift(1).fillna(0) * spread_ret * config.notional

    pos_change = position.diff().abs().fillna(0)
    costs = pos_change * config.notional * config.cost_rate * 2

    return (gross_pnl - costs).rename("pnl")

def _extract_trades(
    df: pd.DataFrame,
    result: CointegrationResult,
    config: BacktestConfig,
    daily_pnl: pd.Series
) -> pd.DataFrame:
    trade_entries = df[df["position"].diff().fillna(0) != 0].copy()
    records = []

    open_trade = None
    for date, row in trade_entries.iterrows():
        if open_trade is None and row["position"] != 0:
            open_trade = {"entry_date": date, "entry_zscore": row["zscore"], "side": row["position"]}
        elif open_trade is not None and row["position"] == 0:
            pnl_window = daily_pnl.loc[open_trade["entry_date"]:date].sum()
            records.append({
                "entry_date":   open_trade["entry_date"],
                "exit_date":    date,
                "side":         "LONG" if open_trade["side"] > 0 else "SHORT",
                "entry_zscore": round(open_trade["entry_zscore"], 3),
                "exit_zscore":  round(row["zscore"], 3),
                "pnl":          round(pnl_window, 2),
            })
            open_trade = None
        elif open_trade is not None and row["position"] != 0:
            pnl_window = daily_pnl.loc[open_trade["entry_date"]:date].sum()
            records.append({
                "entry_date":   open_trade["entry_date"],
                "exit_date":    date,
                "side":         "LONG" if open_trade["side"] > 0 else "SHORT",
                "entry_zscore": round(open_trade["entry_zscore"], 3),
                "exit_zscore":  round(row["zscore"], 3),
                "pnl":          round(pnl_window, 2),
            })
            open_trade = {"entry_date": date, "entry_zscore": row["zscore"], "side": row["position"]}
 
    return pd.DataFrame(records) 

def run_multi_pair_backtest(
    prices: pd.DataFrame,
    coint_result: list, 
    config: BacktestConfig = BacktestConfig(),
    train_end: Optional[str] = None, 
) -> dict: 
    results = {}
    for coint in coint_result: 
        key = f"{coint.ticker_a}/{coint.ticker_b}"
        try:  
            results[key] = run_backtest(prices, coint, config, train_end)
        except Exception as e:
            import logging 
            logging.getLogger(__name__).warning(f"Backtest failed for {key}: {e}")
    return results
    

    
