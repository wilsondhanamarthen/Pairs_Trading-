import numpy as np 
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple
from Cointegration import CointegrationResult 

@dataclass
class SignalConfig:
    lookback: int = 60
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    stop_threshold: float = 3.5

def compute_zscore(spread: pd.Series, lookback: int) -> pd.Series:
    mu    = spread.rolling(window=lookback, min_periods=lookback // 2).mean()
    sigma = spread.rolling(window=lookback, min_periods=lookback // 2).std()
    return (spread - mu) / sigma 

def generate_signals(
    prices: pd.DataFrame,
    result: CointegrationResult,
    config: SignalConfig
) -> pd.DataFrame: 
    
    price_a = prices[result.ticker_a]
    price_b = prices[result.ticker_b]

    log_a = np.log(price_a)
    log_b = np.log(price_b)
    spread = log_a - result.hedge_ratio * log_b - result.intercept
    zscore = compute_zscore(spread, config.lookback)

    position = _compute_positions(zscore, config)
    signal = position.diff().fillna(0)

    df = pd.DataFrame({
        "prices_a": price_a,
        "prices_b": price_b,
        "spread": spread,
        "zscore": zscore,
        "position": position,
        "signal": signal
    })

    return df.dropna(subset=["zscore"])

def _compute_positions(zscore: pd.Series, config: SignalConfig) -> pd.Series:
    position = pd.Series(0.0, index=zscore.index)
    pos = 0.0 

    for i, z in enumerate(zscore):
        if np.isnan(z): 
            position.iloc[i] = 0.0
            continue

        if abs(z) > config.stop_threshold:
            pos = 0.0
        elif abs(z) < config.exit_threshold:
            pos = 0.0
        elif z < -config.entry_threshold:
            pos = 1.0
        elif z > config.entry_threshold:
            pos = -1.0

        position.iloc[i] = pos
    return position

def get_trade_log(signals_df: pd.DataFrame, ticker_a: str, ticker_b: str)-> pd.DataFrame:

    events = signals_df[signals_df["signal"] != 0].copy()
    events["sides"] = events["position"].map({1.0: "LONG spread", -1.0: "SHORT spread", 0.0: "CLOSE"})
    events["pair"] = f"{ticker_a} / {ticker_b}"
    return events[["pair", "sides", "zscore", "spread", "prices_a", "prices_b"]]
