"""
Run the live-sync backtest against an exported MT5 CSV file.

This keeps MT5 out of the loop and points the backtest at a chosen backup model
folder. It is intentionally small so we can run long CSV ranges from PowerShell.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _load_mt5_csv(path: Path, start: datetime | None, end: datetime) -> pl.DataFrame:
    raw = pl.read_csv(path, separator="\t")
    df = raw.with_columns(
        [
            pl.concat_str([pl.col("<DATE>"), pl.lit(" "), pl.col("<TIME>")])
            .str.strptime(pl.Datetime, format="%Y.%m.%d %H:%M:%S")
            .alias("time"),
            pl.col("<OPEN>").cast(pl.Float64).alias("open"),
            pl.col("<HIGH>").cast(pl.Float64).alias("high"),
            pl.col("<LOW>").cast(pl.Float64).alias("low"),
            pl.col("<CLOSE>").cast(pl.Float64).alias("close"),
            pl.col("<TICKVOL>").cast(pl.Int64).alias("volume"),
            pl.col("<SPREAD>").cast(pl.Int64).alias("spread"),
            pl.col("<VOL>").cast(pl.Int64).alias("real_volume"),
        ]
    ).select(["time", "open", "high", "low", "close", "volume", "spread", "real_volume"])

    df = df.sort("time")
    if start:
        # Keep 500 M15 candles before the requested range for indicator/regime warmup.
        warmup_cutoff = start - timedelta(minutes=15 * 500)
        df = df.filter((pl.col("time") >= warmup_cutoff) & (pl.col("time") <= end))
    else:
        df = df.filter(pl.col("time") <= end)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline CSV live-sync backtest")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--start", default="2020-01-01T00:00:00")
    parser.add_argument("--end", default="2026-05-06T23:59:59")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--cooldown", type=int, default=10)
    parser.add_argument("--trend-mult", type=float, default=0.6)
    parser.add_argument("--golden-only", action="store_true")
    parser.add_argument("--no-sell-filter", action="store_true")
    parser.add_argument("--risk-percent", type=float, default=None)
    parser.add_argument("--min-lot", type=float, default=0.01)
    parser.add_argument("--max-lot", type=float, default=100.0)
    parser.add_argument("--lot-step", type=float, default=0.01)
    args = parser.parse_args()

    # Offline run only: satisfy config validation without connecting to MT5.
    os.environ.setdefault("MT5_LOGIN", "1")
    os.environ.setdefault("MT5_PASSWORD", "offline")
    os.environ.setdefault("MT5_SERVER", "offline")

    from backtests.backtest_live_sync import LiveSyncBacktest, TradeResult
    from src.feature_eng import FeatureEngineer
    from src.ml_model import TradingModel
    from src.regime_detector import MarketRegimeDetector
    from src.smc_polars import SMCAnalyzer

    requested_start = _parse_dt(args.start)
    requested_end = _parse_dt(args.end)

    print("=" * 70, flush=True)
    print("FULL CSV LIVE-SYNC BACKTEST", flush=True)
    print("=" * 70, flush=True)
    print(f"CSV: {args.csv}", flush=True)
    print(f"Models: {args.models}", flush=True)
    print(f"Requested range: {requested_start} to {requested_end}", flush=True)

    df = _load_mt5_csv(args.csv, requested_start, requested_end)
    data_start = df["time"].min()
    data_end = df["time"].max()
    print(f"Rows loaded: {len(df)}", flush=True)
    print(f"Data subset range: {data_start} to {data_end}", flush=True)

    print("Calculating indicators...", flush=True)
    df = FeatureEngineer().calculate_all(df, include_ml_features=True)
    df = SMCAnalyzer().calculate_all(df)

    regime_detector = MarketRegimeDetector(model_path=str(args.models / "hmm_regime.pkl")).load()
    try:
        df = regime_detector.predict(df)
    except Exception as exc:
        print(f"WARNING: regime predict skipped: {exc}", flush=True)
    print(f"Indicator columns: {len(df.columns)}", flush=True)

    backtest = LiveSyncBacktest(
        ml_threshold=args.threshold,
        signal_confirmation=2,
        pullback_filter=True,
        golden_time_only=args.golden_only,
        trade_cooldown_bars=args.cooldown,
        trend_reversal_mult=args.trend_mult,
        sell_filter_strict=not args.no_sell_filter,
        use_precomputed_regime=True,
        use_precomputed_ml=True,
        risk_percent_per_trade=args.risk_percent,
        min_lot_size=args.min_lot,
        max_lot_size=args.max_lot,
        lot_step=args.lot_step,
    )
    backtest.regime_detector = MarketRegimeDetector(
        model_path=str(args.models / "hmm_regime.pkl")
    ).load()
    backtest.ml_model = TradingModel(model_path=str(args.models / "xgboost_model.pkl")).load()

    feature_cols = [name for name in backtest.ml_model.feature_names if name in df.columns]
    print("Precomputing ML probabilities...", flush=True)
    df = backtest.ml_model.predict_proba(df, feature_cols)

    effective_start = max(requested_start, data_start)
    times = df["time"].to_list()
    if effective_start <= data_start and len(times) > 100:
        effective_start = times[100]
    effective_end = min(requested_end, data_end)
    print(f"Effective backtest range: {effective_start} to {effective_end}", flush=True)
    if args.risk_percent is not None:
        print(
            f"Equity risk sizing: {args.risk_percent:.2f}% per trade "
            f"(min_lot={args.min_lot}, max_lot={args.max_lot}, step={args.lot_step})",
            flush=True,
        )

    stats = backtest.run(df, start_date=effective_start, end_date=effective_end)
    net_pnl = stats.total_profit - stats.total_loss

    print("\n" + "=" * 70, flush=True)
    print("BACKTEST RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"Effective range: {effective_start} to {effective_end}", flush=True)
    print(f"Total Trades: {stats.total_trades}", flush=True)
    print(f"Wins: {stats.wins}", flush=True)
    print(f"Losses: {stats.losses}", flush=True)
    print(f"Win Rate: {stats.win_rate:.1f}%", flush=True)
    print(f"Total Profit: ${stats.total_profit:.2f}", flush=True)
    print(f"Total Loss: ${stats.total_loss:.2f}", flush=True)
    print(f"Net P/L: ${net_pnl:.2f}", flush=True)
    print(f"Profit Factor: {stats.profit_factor:.2f}", flush=True)
    print(f"Max Drawdown: {stats.max_drawdown:.1f}% (${stats.max_drawdown_usd:.2f})", flush=True)
    print(f"Avg Win: ${stats.avg_win:.2f}", flush=True)
    print(f"Avg Loss: ${stats.avg_loss:.2f}", flush=True)
    print(f"Expectancy: ${stats.expectancy:.2f}", flush=True)
    print(f"Sharpe Ratio: {stats.sharpe_ratio:.2f}", flush=True)

    exit_counts: dict[str, int] = {}
    for trade in stats.trades:
        exit_counts[trade.exit_reason.value] = exit_counts.get(trade.exit_reason.value, 0) + 1
    print("Exit Reasons:", flush=True)
    for reason, count in sorted(exit_counts.items(), key=lambda item: -item[1]):
        pct = count / stats.total_trades * 100 if stats.total_trades else 0
        print(f"  {reason}: {count} ({pct:.1f}%)", flush=True)

    session_stats: dict[str, dict[str, float]] = {}
    for trade in stats.trades:
        session_stats.setdefault(trade.session, {"wins": 0, "losses": 0, "profit": 0.0})
        if trade.result == TradeResult.WIN:
            session_stats[trade.session]["wins"] += 1
        else:
            session_stats[trade.session]["losses"] += 1
        session_stats[trade.session]["profit"] += trade.profit_usd
    print("Session Performance:", flush=True)
    for session, row in session_stats.items():
        total = row["wins"] + row["losses"]
        win_rate = row["wins"] / total * 100 if total else 0
        print(f"  {session}: {total:.0f} trades, {win_rate:.1f}% WR, ${row['profit']:.2f}", flush=True)

    output_dir = Path("backtests/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_path = output_dir / f"csv_backtest_full_{stamp}.csv"
    summary_path = output_dir / f"csv_backtest_full_{stamp}_summary.txt"
    backtest.save_results(stats, str(trades_path))
    summary_path.write_text(
        "\n".join(
            [
                f"CSV: {args.csv}",
                f"Models: {args.models}",
                f"Data subset: {data_start} to {data_end}",
                f"Effective range: {effective_start} to {effective_end}",
                f"Risk percent per trade: {args.risk_percent if args.risk_percent is not None else 'fixed lot'}",
                f"Lot limits: min={args.min_lot}, max={args.max_lot}, step={args.lot_step}",
                f"Total trades: {stats.total_trades}",
                f"Wins: {stats.wins}",
                f"Losses: {stats.losses}",
                f"Win rate: {stats.win_rate:.1f}%",
                f"Total profit: ${stats.total_profit:.2f}",
                f"Total loss: ${stats.total_loss:.2f}",
                f"Net P/L: ${net_pnl:.2f}",
                f"Profit factor: {stats.profit_factor:.2f}",
                f"Max drawdown: {stats.max_drawdown:.1f}% (${stats.max_drawdown_usd:.2f})",
                f"Avg win: ${stats.avg_win:.2f}",
                f"Avg loss: ${stats.avg_loss:.2f}",
                f"Expectancy: ${stats.expectancy:.2f}",
                f"Sharpe ratio: {stats.sharpe_ratio:.2f}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved trades: {trades_path}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
