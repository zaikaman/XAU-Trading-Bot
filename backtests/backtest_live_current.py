"""
Live-current backtest for main_live.py settings.

This runner intentionally models the current live entry/exit stack instead of the
older LiveSyncBacktest:
- ML V2 loader with legacy V1 fallback, threshold as stored by the model
- SMC-only entry logic from main_live._combine_signals
- Equity-risk lot sizing from current equity with no hardcoded max lot cap
- WIB session, night spread guard, max 2 positions, and pyramiding
- SmartRiskManager exits evaluated on historical bar timestamps
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtests.ml_v2.ml_v2_feature_eng import MLV2FeatureEngineer
from backtests.ml_v2.ml_v2_model import TradingModelV2
from src.dynamic_confidence import MarketQuality, create_dynamic_confidence
from src.feature_eng import FeatureEngineer
from src.regime_detector import MarketRegime, MarketRegimeDetector
from src.smc_polars import SMCAnalyzer, SMCSignal
from src.position_manager import SmartPositionManager
from src.smart_risk_manager import ExitReason as SmartExitReason
from src.smart_risk_manager import create_smart_risk_manager
import src.smart_risk_manager as smart_risk_module


logger.remove()
logger.add(sys.stderr, level="WARNING")

WIB = ZoneInfo("Asia/Jakarta")
UTC = ZoneInfo("UTC")


class HistoricalDateTime(datetime):
    """datetime shim so SmartRiskManager sees historical time progression."""

    current: datetime = datetime(2026, 1, 1, tzinfo=WIB)

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls.current.replace(tzinfo=None)
        return cls.current.astimezone(tz)


class BacktestRegimeValue:
    def __init__(self, value: str):
        self.value = value


class BacktestRegimeState:
    def __init__(self, value: str):
        self.regime = BacktestRegimeValue(value)


@dataclass
class OpenPosition:
    ticket: int
    entry_idx: int
    entry_time: datetime
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    ml_confidence: float
    smc_confidence: float
    regime: str
    session: str
    signal_reason: str
    parent_ticket: Optional[int] = None


@dataclass
class ClosedTrade:
    ticket: int
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    profit_usd: float
    result: str
    exit_reason: str
    ml_confidence: float
    smc_confidence: float
    regime: str
    session: str
    signal_reason: str
    parent_ticket: Optional[int]


def load_mt5_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", try_parse_dates=False).rename(
        {
            "<OPEN>": "open",
            "<HIGH>": "high",
            "<LOW>": "low",
            "<CLOSE>": "close",
            "<TICKVOL>": "volume",
            "<VOL>": "real_volume",
            "<SPREAD>": "spread",
        }
    )
    return (
        df.with_columns(
            [
                (pl.col("<DATE>") + pl.lit(" ") + pl.col("<TIME>"))
                .str.strptime(pl.Datetime, "%Y.%m.%d %H:%M:%S")
                .alias("time"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
                pl.col("real_volume").cast(pl.Float64),
                pl.col("spread").cast(pl.Float64),
            ]
        )
        .select(["time", "open", "high", "low", "close", "volume", "real_volume", "spread"])
        .sort("time")
    )


def build_h1(df_m15: pl.DataFrame) -> pl.DataFrame:
    return (
        df_m15.group_by_dynamic("time", every="1h", closed="left", label="left")
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("real_volume").sum().alias("real_volume"),
                pl.col("spread").mean().alias("spread"),
            ]
        )
        .drop_nulls(["open", "high", "low", "close"])
    )


def as_wib(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(WIB)


def session_for(ts: datetime) -> tuple[bool, str, float, str]:
    """Approximate create_wib_session_filter(aggressive=True) for historical bars."""
    w = as_wib(ts)
    mins = w.hour * 60 + w.minute
    if w.weekday() == 5 or w.weekday() == 6:
        return False, "Weekend", 0.0, "low"
    if w.weekday() == 5 and w.hour == 4 and w.minute >= 30:
        return False, "Friday Close", 0.0, "low"
    if 4 * 60 <= mins < 6 * 60:
        return False, "Rollover", 0.0, "low"
    if 0 <= mins < 4 * 60:
        return False, "Dead Zone", 0.0, "low"
    if 20 * 60 <= mins <= 23 * 60 + 59:
        return True, "London-NY Overlap (GOLDEN)", 1.2, "extreme"
    if 15 * 60 <= mins <= 16 * 60:
        return True, "Tokyo-London Overlap", 0.7, "high"
    if 15 * 60 <= mins <= 23 * 60 + 59:
        return True, "London", 1.0, "high"
    if 7 * 60 <= mins <= 16 * 60:
        return True, "Tokyo", 0.7, "medium"
    if 6 * 60 <= mins <= 13 * 60:
        return True, "Sydney", 0.5, "low"
    return False, "Off Hours", 0.0, "low"


def profit_for(position: OpenPosition, price: float) -> float:
    if position.direction == "BUY":
        pips = (price - position.entry_price) / 0.1
    else:
        pips = (position.entry_price - price) / 0.1
    return pips * 10.0 * position.lot_size


class LiveCurrentBacktest:
    def __init__(self, initial_capital: float = 1000.0):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        self.max_drawdown_usd = 0.0
        self.ticket_counter = 2_000_000
        self.trades: List[ClosedTrade] = []
        self.positions: Dict[int, OpenPosition] = {}
        self.last_trade_time: Optional[datetime] = None
        self.pyramid_done_tickets: set[int] = set()
        self.last_pyramid_time: Optional[datetime] = None
        self.current_trade_date = None

        self.smc = SMCAnalyzer()
        self.dynamic_confidence = create_dynamic_confidence()
        self.smart_risk = create_smart_risk_manager(capital=initial_capital)
        # Backtests must not mutate live runtime risk-state files.
        self.smart_risk._save_daily_state = lambda: None
        self.smart_risk._total_loss = 0.0
        self.smart_risk._state = smart_risk_module.RiskState()
        self.position_manager = SmartPositionManager(
            breakeven_pips=20.0,
            trail_start_pips=35.0,
            trail_step_pips=20.0,
            atr_be_mult=2.0,
            atr_trail_start_mult=3.0,
            atr_trail_step_mult=2.0,
            min_profit_to_protect=0.0,
            max_drawdown_from_peak=40.0,
            enable_market_close_handler=True,
            min_profit_before_close=0.0,
            max_loss_to_hold=0.0,
        )

    def _set_historical_now(self, ts: datetime) -> None:
        wib_now = as_wib(ts)
        HistoricalDateTime.current = wib_now
        smart_risk_module.datetime = HistoricalDateTime
        smart_risk_module.time.time = lambda: wib_now.timestamp()
        if self.current_trade_date != wib_now.date():
            self.current_trade_date = wib_now.date()
            self.smart_risk._state = smart_risk_module.RiskState()
            self.smart_risk._daily_pnl = []
            self.smart_risk._current_date = wib_now.date()

    def _close_position(self, pos: OpenPosition, idx: int, ts: datetime, price: float, reason: str) -> None:
        profit = profit_for(pos, price)
        self.equity += profit
        self.smart_risk.record_trade_result(profit)
        self.smart_risk.unregister_position(pos.ticket)
        self.positions.pop(pos.ticket, None)

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd_usd = self.peak_equity - self.equity
        dd_pct = dd_usd / self.peak_equity * 100 if self.peak_equity > 0 else 0.0
        if dd_pct > self.max_drawdown:
            self.max_drawdown = dd_pct
            self.max_drawdown_usd = dd_usd

        self.trades.append(
            ClosedTrade(
                ticket=pos.ticket,
                entry_time=pos.entry_time,
                exit_time=ts,
                direction=pos.direction,
                entry_price=pos.entry_price,
                exit_price=price,
                stop_loss=pos.stop_loss,
                take_profit=pos.take_profit,
                lot_size=pos.lot_size,
                profit_usd=profit,
                result="WIN" if profit > 0 else "LOSS" if profit < 0 else "BREAKEVEN",
                exit_reason=reason,
                ml_confidence=pos.ml_confidence,
                smc_confidence=pos.smc_confidence,
                regime=pos.regime,
                session=pos.session,
                signal_reason=pos.signal_reason,
                parent_ticket=pos.parent_ticket,
            )
        )

    def _evaluate_open_positions(self, df: pl.DataFrame, i: int, ml_pred) -> None:
        row = df.row(i, named=True)
        ts = row["time"]
        self._set_historical_now(ts)
        atr = float(row.get("atr") or 0.0)
        baseline_atr = float(df.slice(max(0, i - 96), min(i, 96))["atr"].drop_nulls().mean() or atr or 0.0)
        regime = str(row.get("regime_name") or "normal")

        for ticket, pos in list(self.positions.items()):
            # Broker-level TP/SL approximation inside the M15 bar.
            if pos.direction == "BUY":
                if row["low"] <= pos.stop_loss:
                    self._close_position(pos, i, ts, pos.stop_loss, "broker_sl")
                    continue
                if row["high"] >= pos.take_profit:
                    self._close_position(pos, i, ts, pos.take_profit, "take_profit")
                    continue
                mark = row["close"]
            else:
                if row["high"] >= pos.stop_loss:
                    self._close_position(pos, i, ts, pos.stop_loss, "broker_sl")
                    continue
                if row["low"] <= pos.take_profit:
                    self._close_position(pos, i, ts, pos.take_profit, "take_profit")
                    continue
                mark = row["close"]

            current_profit = profit_for(pos, mark)
            position_df = pl.DataFrame(
                [
                    {
                        "ticket": pos.ticket,
                        "type": pos.direction,
                        "price_open": pos.entry_price,
                        "sl": pos.stop_loss,
                        "tp": pos.take_profit,
                        "profit": current_profit,
                        "volume": pos.lot_size,
                    }
                ]
            )
            pm_actions = self.position_manager.analyze_positions(
                positions=position_df,
                df_market=df.head(i + 1),
                regime_state=BacktestRegimeState(regime),
                ml_prediction=ml_pred,
                current_price=mark,
            )
            position_closed = False
            for action in pm_actions:
                if action.ticket != ticket:
                    continue
                if action.action == "CLOSE":
                    self._close_position(pos, i, ts, mark, f"position_manager:{action.reason}")
                    position_closed = True
                    break
                if action.action == "TRAIL_SL" and action.new_sl:
                    pos.stop_loss = action.new_sl
            if position_closed:
                continue

            market_context = {
                "rsi": row.get("rsi"),
                "macd_hist": row.get("macd_histogram"),
                "session_name": pos.session,
                "is_golden": "GOLDEN" in pos.session.upper(),
            }
            should_close, reason, _message = self.smart_risk.evaluate_position(
                ticket=ticket,
                current_price=mark,
                current_profit=current_profit,
                ml_signal=ml_pred.signal,
                ml_confidence=ml_pred.confidence,
                regime=regime,
                current_atr=atr,
                baseline_atr=baseline_atr,
                market_context=market_context,
            )
            if should_close:
                reason_value = reason.value if isinstance(reason, SmartExitReason) else str(reason)
                self._close_position(pos, i, ts, mark, reason_value)

    def _combine_signal(self, smc_signal, ml_pred, regime: str, session: str, volatility: str, df_slice: pl.DataFrame):
        if smc_signal is None:
            return None
        analysis = self.dynamic_confidence.analyze_market(
            session=session,
            regime=regime,
            volatility=volatility,
            trend_direction=regime,
            has_smc_signal=True,
            ml_signal=ml_pred.signal,
            ml_confidence=ml_pred.confidence,
        )
        if analysis.quality == MarketQuality.AVOID or regime == MarketRegime.CRISIS.value:
            return None
        if smc_signal.confidence < 0.55:
            return None

        ml_agrees = (
            (smc_signal.signal_type == "BUY" and ml_pred.signal == "BUY")
            or (smc_signal.signal_type == "SELL" and ml_pred.signal == "SELL")
        )
        combined = (smc_signal.confidence + ml_pred.confidence) / 2 if ml_agrees else smc_signal.confidence

        atr_ratio = 1.0
        if "atr" in df_slice.columns:
            atrs = df_slice["atr"].drop_nulls()
            if len(atrs) >= 96:
                cur = atrs.tail(1).item()
                base = atrs.tail(96).mean()
                atr_ratio = cur / base if base else 1.0
        if session == "London" and atr_ratio < 1.2:
            combined *= 0.90
        if regime == MarketRegime.HIGH_VOLATILITY.value:
            combined *= 0.90

        return SMCSignal(
            signal_type=smc_signal.signal_type,
            entry_price=smc_signal.entry_price,
            stop_loss=smc_signal.stop_loss,
            take_profit=smc_signal.take_profit,
            confidence=combined,
            reason=f"SMC-ONLY: {smc_signal.reason} | ML: {ml_pred.signal} ({ml_pred.confidence:.0%})",
        )

    def _open_position(self, signal: SMCSignal, i: int, ts: datetime, ml_pred, regime: str, session: str, lot: float, parent: Optional[int] = None):
        self.ticket_counter += 1
        pos = OpenPosition(
            ticket=self.ticket_counter,
            entry_idx=i,
            entry_time=ts,
            direction=signal.signal_type,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            lot_size=lot,
            ml_confidence=ml_pred.confidence,
            smc_confidence=signal.confidence,
            regime=regime,
            session=session,
            signal_reason=signal.reason,
            parent_ticket=parent,
        )
        self.positions[pos.ticket] = pos
        sl_distance = abs(pos.entry_price - pos.stop_loss)
        risk_amount = pos.lot_size * sl_distance * 100
        self.smart_risk.update_capital(self.equity)
        guard = self.smart_risk.register_position(
            pos.ticket,
            pos.entry_price,
            pos.lot_size,
            pos.direction,
            max_loss_usd=risk_amount,
        )
        guard.entry_time = as_wib(ts)
        self.last_trade_time = ts
        return pos

    def _try_pyramid(self, df: pl.DataFrame, i: int, signal: Optional[SMCSignal], ml_pred) -> None:
        if signal is None or not self.positions:
            return
        ts = df["time"][i]
        if self.last_pyramid_time and (ts - self.last_pyramid_time).total_seconds() < 30:
            return
        can_open, _ = self.smart_risk.can_open_position()
        if not can_open:
            return
        row = df.row(i, named=True)
        atr = float(row.get("atr") or 0.0)
        baseline_atr = float(df.slice(max(0, i - 96), min(i, 96))["atr"].drop_nulls().mean() or atr or 0.0)
        sm = max(0.3, min(atr / baseline_atr, 1.5)) if baseline_atr > 0 else 1.0

        for pos in list(self.positions.values()):
            if pos.ticket in self.pyramid_done_tickets:
                continue
            if pos.session not in ("London", "New York", "London-NY Overlap", "London-NY Overlap (GOLDEN)"):
                continue
            if signal.signal_type != pos.direction or signal.confidence < 0.75 or ml_pred.signal != pos.direction:
                continue
            profit = profit_for(pos, row["close"])
            atr_unit = atr * pos.lot_size * 100 if atr > 0 else 10 * sm
            if profit < 0.5 * atr_unit:
                continue
            pyramid_signal = SMCSignal(
                signal_type=pos.direction,
                entry_price=row["close"],
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                confidence=signal.confidence,
                reason=f"PYRAMID: Add to winner #{pos.ticket}",
            )
            sl_distance = abs(pyramid_signal.entry_price - pyramid_signal.stop_loss)
            if sl_distance <= 0:
                continue
            pyramid_lot = round(((self.equity * 0.02) / (sl_distance * 100)) / 0.01) * 0.01
            pyramid_lot = max(0.01, pyramid_lot)
            self._open_position(pyramid_signal, i, ts, ml_pred, pos.regime, pos.session, pyramid_lot, parent=pos.ticket)
            self.pyramid_done_tickets.add(pos.ticket)
            self.last_pyramid_time = ts
            break

    def run(self, df: pl.DataFrame, model: TradingModelV2, start: datetime, end: datetime) -> None:
        times = df["time"].to_list()
        start_idx = next((idx for idx, t in enumerate(times) if t >= start), 200)
        end_idx = next((idx for idx, t in enumerate(times) if t >= end), len(df))
        features = [f for f in model.feature_names if f in df.columns]

        print(f"Running live-current backtest: {times[start_idx]} to {times[end_idx - 1]}")
        print(f"Bars: {end_idx - start_idx} | ML features: {len(features)}")

        for i in range(start_idx, end_idx):
            ts = times[i]
            self._set_historical_now(ts)
            self.smart_risk.update_capital(self.equity)
            df_slice = df.head(i + 1)
            row = df.row(i, named=True)

            try:
                ml_pred = model.predict(df_slice, features)
            except Exception:
                continue

            self._evaluate_open_positions(df, i, ml_pred)

            can_session, session, _session_mult, volatility = session_for(ts)
            if not can_session:
                continue

            # Live cooldown is 150 seconds. With M15 bars this only blocks same-bar
            # synthetic entries; pyramids still have their own 30s guard.
            if self.last_trade_time and (ts - self.last_trade_time).total_seconds() < 150:
                continue

            if not self.smart_risk.get_trading_recommendation()["can_trade"]:
                continue

            try:
                raw_smc = self.smc.generate_signal(df_slice)
            except Exception:
                continue

            regime = str(row.get("regime_name") or "normal")
            final_signal = self._combine_signal(raw_smc, ml_pred, regime, session, volatility, df_slice)
            self._try_pyramid(df, i, final_signal, ml_pred)
            if final_signal is None:
                continue

            wib = as_wib(ts)
            if wib.hour >= 22 or wib.hour <= 5:
                spread_limit = 80 if "GOLDEN" in session.upper() else 50
                if float(row.get("spread") or 0.0) > spread_limit:
                    continue

            can_open, _ = self.smart_risk.can_open_position()
            if not can_open:
                continue

            sl_distance = abs(final_signal.entry_price - final_signal.stop_loss)
            if sl_distance <= 0:
                continue
            raw_lot = (self.equity * 0.02) / (sl_distance * 100)
            lot = round(raw_lot / 0.01) * 0.01
            lot = max(0.01, lot)
            if lot <= 0:
                continue

            self._open_position(final_signal, i, ts, ml_pred, regime, session, lot)

            if len(self.trades) and len(self.trades) % 100 == 0:
                print(f"  {len(self.trades)} closed trades...")

        final_ts = times[end_idx - 1]
        final_close = df["close"][end_idx - 1]
        for pos in list(self.positions.values()):
            self._close_position(pos, end_idx - 1, final_ts, final_close, "end_of_test")


def prepare_features(df: pl.DataFrame) -> pl.DataFrame:
    fe = FeatureEngineer()
    smc = SMCAnalyzer()
    reg = MarketRegimeDetector(model_path="models/hmm_regime.pkl")
    v2 = MLV2FeatureEngineer()

    h1 = build_h1(df)
    h1 = fe.calculate_all(h1, include_ml_features=True)
    h1 = smc.calculate_all(h1)

    df = fe.calculate_all(df, include_ml_features=True)
    df = smc.calculate_all(df)
    try:
        reg.load()
        df = reg.predict(df)
    except Exception as exc:
        print(f"Regime precompute skipped: {exc}")
    if "regime" not in df.columns:
        df = df.with_columns(pl.lit(1).alias("regime"))
    if "regime_confidence" not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("regime_confidence"))
    return v2.add_all_v2_features(df, h1)


def load_live_model() -> TradingModelV2:
    model = TradingModelV2(confidence_threshold=0.60, model_path="models/xgboost_model.pkl")
    model.load("models/xgboost_model.pkl")
    if not model.fitted or not model.feature_names:
        model = TradingModelV2(confidence_threshold=0.60, model_path="models/xgboost_model.pkl")
        model.load_legacy_v1("models/xgboost_model.pkl")
    return model


def write_outputs(bt: LiveCurrentBacktest, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [trade.__dict__ for trade in bt.trades]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    wins = sum(1 for t in bt.trades if t.profit_usd > 0)
    losses = sum(1 for t in bt.trades if t.profit_usd < 0)
    total_profit = sum(t.profit_usd for t in bt.trades if t.profit_usd > 0)
    total_loss = abs(sum(t.profit_usd for t in bt.trades if t.profit_usd < 0))
    net = total_profit - total_loss
    summary = {
        "total_trades": len(bt.trades),
        "wins": wins,
        "losses": losses,
        "win_rate": f"{wins / len(bt.trades) * 100:.1f}%" if bt.trades else "0.0%",
        "total_profit": f"${total_profit:.2f}",
        "total_loss": f"${total_loss:.2f}",
        "net_pnl": f"${net:.2f}",
        "ending_equity": f"${bt.initial_capital + net:.2f}",
        "return_pct": f"{net / bt.initial_capital * 100:.2f}%",
        "profit_factor": f"{total_profit / total_loss:.2f}" if total_loss else "inf",
        "avg_win": f"${total_profit / wins:.2f}" if wins else "$0.00",
        "avg_loss": f"${total_loss / losses:.2f}" if losses else "$0.00",
        "avg_trade": f"${net / len(bt.trades):.2f}" if bt.trades else "$0.00",
        "max_drawdown_pct": f"{bt.max_drawdown:.1f}%",
        "max_drawdown_usd": f"${bt.max_drawdown_usd:.2f}",
        "sharpe_ratio": f"{(np.mean([t.profit_usd for t in bt.trades]) / np.std([t.profit_usd for t in bt.trades]) * np.sqrt(252)):.2f}"
        if len(bt.trades) > 1 and np.std([t.profit_usd for t in bt.trades]) > 0
        else "0.00",
    }
    summary_path = output.with_name(output.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(summary.items())

    print("\n" + "=" * 72)
    print("LIVE-CURRENT BACKTEST RESULTS")
    print("=" * 72)
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nExit Reasons:")
    counts: Dict[str, int] = {}
    for t in bt.trades:
        counts[t.exit_reason] = counts.get(t.exit_reason, 0) + 1
    for reason, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = count / len(bt.trades) * 100 if bt.trades else 0
        print(f"  {reason}: {count} ({pct:.1f}%)")

    print("\nSession Performance:")
    sessions: Dict[str, Dict[str, float]] = {}
    for t in bt.trades:
        s = sessions.setdefault(t.session, {"trades": 0, "wins": 0, "profit": 0.0})
        s["trades"] += 1
        s["wins"] += 1 if t.profit_usd > 0 else 0
        s["profit"] += t.profit_usd
    for session, data in sessions.items():
        wr = data["wins"] / data["trades"] * 100 if data["trades"] else 0
        print(f"  {session}: {int(data['trades'])} trades, {wr:.1f}% WR, ${data['profit']:.2f}")

    print(f"\nSaved trades: {output}")
    print(f"Saved summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/XAUUSD_M15_202202102100_202605070830.csv")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-05-01")
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--output", default="backtests/results/live_current_20260101_20260430.csv")
    args = parser.parse_args()

    os.environ.setdefault("MT5_LOGIN", "1")
    os.environ.setdefault("MT5_PASSWORD", "dummy")
    os.environ.setdefault("MT5_SERVER", "dummy")
    os.environ.setdefault("RISK_PER_TRADE", "2")

    df = load_mt5_csv(Path(args.csv))
    print(f"Loaded {len(df)} bars: {df['time'][0]} to {df['time'][-1]}")
    df = prepare_features(df)
    model = load_live_model()
    print(f"Loaded model: fitted={model.fitted}, threshold={model.confidence_threshold:.0%}, features={len(model.feature_names)}")

    bt = LiveCurrentBacktest(initial_capital=args.capital)
    bt.run(df, model, datetime.fromisoformat(args.start), datetime.fromisoformat(args.end))
    write_outputs(bt, Path(args.output))


if __name__ == "__main__":
    main()
