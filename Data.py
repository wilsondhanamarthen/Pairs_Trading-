import yfinance as yf
import numpy as np
import pandas as pd
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def fetch_prices(
    tickers: List[str],
    start: str,
    end: str,
    missing_threshold: float = 0.05, 
) -> pd.DataFrame:
    logger.info(f"Fetching {len(tickers)} tickers from {start} to {end}")

    raw = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices = _clean(prices, missing_threshold)
    logger.info(f"Returned {prices.shape[1]} tickers, {prices.shape[0]} trading days") 
    return prices 

def _clean(df: pd.DataFrame, missing_threshold: float) -> pd.DataFrame: 
    n = len(df)
    missing_frac = df.isna().sum()/n 
    bad = missing_frac[missing_frac> missing_threshold].index.tolist() 
    if bad:
        logger.warning(f"Dropping tickers with too many NaNs: {bad}")
        df = df.drop(columns=bad)
    return df.ffill().bfill().dropna(axis=1)

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()

def get_sector_universe(sector: Optional[str] = None) -> List[str]:
    universe = {
        "energy":     ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX"],
        "tech":       ["MSFT", "AAPL", "GOOGL", "META", "NVDA", "AMD", "INTC", "QCOM"],
        "consumer":   ["KO", "PEP", "MCD", "SBUX", "YUM", "DPZ", "CMG", "WEN"],
        "financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC"],
    }
    if sector is None:
        return [t for tickers in universe.values() for t in tickers]
    if sector not in universe:
        raise ValueError(f"Unknown sector '{sector}'. Choose from: {list(universe.keys())}")
    return universe[sector]

