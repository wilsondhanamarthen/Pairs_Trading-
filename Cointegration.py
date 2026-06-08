import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller
from dataclasses import dataclass, field 
from typing import List, Tuple
import logging  

logger = logging.getLogger(__name__)

@dataclass
class CointegrationResult:
    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    intercept: float
    adf_stat: float
    p_value: float
    half_life: float
    spread: pd.Series = field(repr=False)

    @property
    def is_valid(self) -> bool:
        return self.p_value < 0.05 and 5 <= self.half_life <=126

    def __str__(self) -> str:
        return (
             f"{self.ticker_a}/ {self.ticker_b}: | "
             f"β={self.hedge_ratio:.4f} | "
             f"p={self.p_value:.4f} | "
             f"half_life={self.half_life:.1f}d | "
             f"valid={self.is_valid}"
        )

def compute_spread(
    price_a: pd.Series,
    price_b: pd.Series, 
) -> Tuple[float, float, pd.Series]:
    log_a = np.log(price_a)
    log_b = np.log(price_b)

    X = add_constant(log_b)
    model = OLS(log_a, X).fit()

    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]
    spread = log_a - beta * log_b - alpha
    return alpha, beta, spread

def compute_half_life(spread: pd.Series) -> float:
    
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()

    X = add_constant(lagged)
    model = OLS(delta, X).fit()

    lam = model.params.iloc[1]
    if lam >= 0:
        return np.inf
    
    half_life = -np.log(2) / np.log(1 + lam)
    return max(half_life, 0)

def test_cointegration(
    price_a: pd.Series,
    price_b: pd.Series,
    ticker_a: str,
    ticker_b: str
) -> CointegrationResult:
    alpha, beta, spread = compute_spread(price_a, price_b)
    adf_result = adfuller(spread, autolag='AIC')
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    half_life = compute_half_life(spread)


    return CointegrationResult(
        ticker_a = ticker_a,
        ticker_b = ticker_b,
        hedge_ratio = beta,
        intercept = alpha,
        adf_stat = adf_stat,
        p_value = p_value,
        half_life = half_life,
        spread = spread
)

def find_cointegrated_pairs(
    prices: pd.DataFrame, 
    p_threshold: float = 0.05, 
    min_half_life: float = 5, 
    max_half_life: float = 126, 
    top_n: int = 10, 
) -> List[CointegrationResult]:
    tickers = prices.columns.tolist()
    pairs   = list(combinations(tickers, 2))
    results = []
 
    logger.info(f"Testing {len(pairs)} pairs for cointegration...")
 
    for ticker_a, ticker_b in pairs:
        try:
            result = test_cointegration(
            prices[ticker_a],
            prices[ticker_b],
                ticker_a,
                ticker_b,
            )
            if (
                result.p_value < p_threshold
                and min_half_life <= result.half_life <= max_half_life
            ):
                results.append(result)
        except Exception as e:
            logger.debug(f"Skipping {ticker_a}/{ticker_b}: {e}")
 
    results.sort(key=lambda r: r.p_value)
    logger.info(f"Found {len(results)} cointegrated pairs")
 
    return results[:top_n]
 

def summarize_pairs(results: List[CointegrationResult]) -> pd.DataFrame:
    
    return pd.DataFrame([
        {
            "pair":        f"{r.ticker_a}/{r.ticker_b}",
            "hedge_ratio": round(r.hedge_ratio, 4),
            "p_value":     round(r.p_value, 4),
            "adf_stat":    round(r.adf_stat, 4),
            "half_life_d": round(r.half_life, 1),
        }
        for r in results
    ])  

