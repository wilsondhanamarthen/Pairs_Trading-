import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import warnings
from Data import fetch_prices, get_sector_universe
from Cointegration import find_cointegrated_pairs, summarize_pairs
from Backtest import BacktestConfig, run_multi_pair_backtest
from Metrics import evaluate_portfolio, evaluate 
from Signals import SignalConfig 

SECTOR = None
TRAIN_START = "2015-01-01"
TRAIN_END = "2020-12-01"
TEST_START = "2021-01-01"
TEST_END = "2023-12-31"

BACKTEST_CFG = BacktestConfig(
    notional = 10_000, 
    cost_bps = 10, 
    signal_config = SignalConfig(
        lookback = 60, 
        entry_threshold = 2.0, 
        stop_threshold = 3.5,
    )
)
    
     

def main():
    print("=" * 60)
    print("PAIRS TRADING PIPELINE")
    print("=" * 60)

    tickers = get_sector_universe(SECTOR)
    prices = fetch_prices(tickers, start=TRAIN_START, end=TEST_END)

    train_prices = prices.loc[:TRAIN_END]
    print(f"\n[1] Finding cointegrated pairs in {SECTOR} sector ({TRAIN_START}-{TRAIN_END})")
    coint_results = find_cointegrated_pairs(train_prices, top_n=5)

    print("\nTop cointegrated pairs found:")
    print(summarize_pairs(coint_results).to_string(index=False))

    if not coint_results:
        print("No cointegrated pairs found. Try a different sector or data range")
        return
    
    print(f"\n[2] Running out-of-sample backtest ({TEST_START}-{TEST_END})")
    bt_results = run_multi_pair_backtest(
        prices = prices,
        coint_result = coint_results,
        config = BACKTEST_CFG,
        train_end = TEST_START,
    )

    print("\n[3] Peforming metrices")
    summary = evaluate_portfolio(bt_results)
    print(summary.to_string(index=False))

    for result in bt_results.values():
        print(evaluate(result))

    print("\n[4] Generating plots...")
    plot_results(bt_results, prices, coint_results[0])

def plot_results(bt_results, prices, best_pair):
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2,2, figure=fig, hspace=0.4,wspace=0.35)

    ax1 = fig.add_subplot(gs[0,0])
    portfolio_pnl = None

    for key, result in bt_results.items():
        cum = result.cumulative_pnl
        ax1.plot(cum.index, cum.values, linewidth=1, alpha=0.6, label=key)
        portfolio_pnl = cum if portfolio_pnl is None else portfolio_pnl.add(cum, fill_value=0)

    if portfolio_pnl is not None:
        ax1.plot(portfolio_pnl.index, portfolio_pnl.values, linewidth=2.0, color="black", label="Portfolio", zorder=5)

    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_title("Cumulative PnL (out-of-sample)", fontsize=14)
    ax1.set_ylabel("PnL ($)")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax2 = fig.add_subplot(gs[0,1])
    key = f"{best_pair.ticker_a}/{best_pair.ticker_b}"
    sig = bt_results[key].signals_df if key in bt_results else None

    if sig is not None:
        ax2.plot(sig.index, sig["zscore"], linewidth=1, color="#2563eb", label="z-score")
        ax2.axhline( 2.0, color="red",   linestyle="--", linewidth=0.8, label="±entry (2σ)")
        ax2.axhline(-2.0, color="red",   linestyle="--", linewidth=0.8)
        ax2.axhline( 0.5, color="green", linestyle=":",  linewidth=0.8, label="±exit (0.5σ)")
        ax2.axhline(-0.5, color="green", linestyle=":",  linewidth=0.8)
        ax2.axhline( 3.5, color="black", linestyle="-.", linewidth=0.8, label="±stop (3.5σ)")
        ax2.axhline(-3.5, color="black", linestyle="-.", linewidth=0.8)
        ax2.fill_between(sig.index, 2.0, 3.5,  alpha=0.06, color="red")
        ax2.fill_between(sig.index, -3.5, -2.0, alpha=0.06, color="red")

    ax2.set_title(f"Spread Z-score: {key}", fontsize=11)
    ax2.set_ylabel("z-score (σ)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax3 = fig.add_subplot(gs[1,0])
    for ticker in [best_pair.ticker_a, best_pair.ticker_b]:
        p = prices.loc[TEST_START:][ticker] 
        ax3.plot(p.index, p / p.iloc[0], linewidth=1.2, label=ticker)

    ax3.set_title(f"Normalized Price: {key}", fontsize=11)
    ax3.set_ylabel("Normalized Price (base=1)")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax4 = fig.add_subplot(gs[1,1])
    trade_list = [r.trades for r in bt_results.values() if not r.trades.empty]
    if trade_list:
        all_trades = pd.concat(trade_list, ignore_index=True)
        colors = ["#16a34a" if p > 0 else "#dc2626" for p in all_trades["pnl"]]
        ax4.bar(range(1, len(all_trades) + 1), all_trades["pnl"].values, color=colors, alpha=0.7)
        ax4.axhline(0, color="black", linewidth=0.8)
        ax4.set_title("Individual Trade PnL", fontsize=11)
        ax4.set_xlabel("Trade #")
        ax4.set_ylabel("PnL ($)")
        ax4.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Pairs Trading Strategy — Out-of-Sample Results", fontsize=13, fontweight="bold")
    plt.savefig("pairs_trading_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved to pairs_trading_results.png")
 
 
if __name__ == "__main__":
    main()
