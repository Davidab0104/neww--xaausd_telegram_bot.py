#!/usr/bin/env python3
"""
live_forex_bot_with_rf.py

Features:
- /start, /stop, /symbol SYMBOL, /train, /backtest Ndays, /setbalance, /setrisk
- EMA(9,21,50), RSI(14), Bollinger Bands(20)
- Candlestick patterns (TA-Lib preferred, fallback)
- Scalping ensemble signals + confidence
- RandomForestRegressor price-predictor trained on features; saved/loaded via joblib
- Backtest endpoint to compute winrate & expectancy
- Uses Twelve Data (1-min)
"""

import os
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load

# optional TA-Lib
try:
    import talib  # type: ignore
    TALIB_AVAILABLE = True
except Exception:
    TALIB_AVAILABLE = False

# ---------- config ----------
load_dotenv()
BOT_TOKEN = "8353106506:AAFLJBOKmGrsXLF7hP8-LbnggUXzD0nkdTc" or ""
TWELVE_API_KEY = "227f52dcf840492aad25ed204b2df1f1" or ""

if not BOT_TOKEN or not TWELVE_API_KEY:
    raise RuntimeError("Please set TELEGRAM_BOT_TOKEN and TWELVEDATA_API_KEY in a .env file")

MODEL_FILE = "rf_model.pkl"
POLL_SECONDS = 60
TD_INTERVAL = "1min"
TD_OUTPUTSIZE = 5000  # fetch up to this many points (TwelveData limits may apply)

# scalping / sizing defaults (modifiable via commands)
ACCOUNT_BALANCE = 1000.0
MAX_RISK_PCT = 1.0
SIGNAL_CONFIDENCE_THRESHOLD = 60.0  # percent
PREDICT_WINDOW = 20  # used for moving-window features (and fallback predict)

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("live_forex_bot_rf")

# per-chat state
chat_state: Dict[int, Dict[str, Any]] = {}
DEFAULT_SYMBOL = "XAU/USD"

# ---------- Utilities ----------
def normalize_symbol(user_symbol: str) -> str:
    s = user_symbol.strip().upper().replace(" ", "").replace("\\", "/")
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2 and parts[0] and parts[1]:
            return f"{parts[0]}/{parts[1]}"
    if len(s) == 6:
        return f"{s[0:3]}/{s[3:6]}"
    return s

def fetch_ohlcv_sync(symbol: str, interval: str = TD_INTERVAL, outputsize: int = 500) -> Optional[pd.DataFrame]:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": str(outputsize),
        "apikey": TWELVE_API_KEY,
        "format": "JSON"
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        data = r.json()
    except Exception as e:
        logger.exception("HTTP error while fetching data: %s", e)
        return None

    if not data or "values" not in data:
        logger.warning("TwelveData returned no values (response: %s)", data)
        return None

    df = pd.DataFrame(data["values"])
    df = df.rename(columns={c: c.lower() for c in df.columns})
    required = ["datetime", "open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            logger.warning("Missing required column '%s' in data", col)
            return None
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

# ---------- Indicators ----------
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    try:
        if TALIB_AVAILABLE:
            df["EMA9"] = talib.EMA(close.values, timeperiod=9)
            df["EMA21"] = talib.EMA(close.values, timeperiod=21)
            df["EMA50"] = talib.EMA(close.values, timeperiod=50)
            df["RSI"] = talib.RSI(close.values, timeperiod=14)
            upper, middle, lower = talib.BBANDS(close.values, timeperiod=20)
            df["BB_upper"], df["BB_middle"], df["BB_lower"] = upper, middle, lower
        else:
            df["EMA9"] = close.ewm(span=9, adjust=False).mean()
            df["EMA21"] = close.ewm(span=21, adjust=False).mean()
            df["EMA50"] = close.ewm(span=50, adjust=False).mean()
            delta = close.diff()
            up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            down = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
            rs = up / down
            df["RSI"] = 100 - (100 / (1 + rs))
            mid = close.rolling(20).mean()
            std = close.rolling(20).std()
            df["BB_middle"] = mid
            df["BB_upper"] = mid + 2 * std
            df["BB_lower"] = mid - 2 * std
    except Exception:
        logger.exception("Indicator error; using pandas fallback")
        df["EMA9"] = close.ewm(span=9, adjust=False).mean()
        df["EMA21"] = close.ewm(span=21, adjust=False).mean()
        df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    return df

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ---------- Candlestick detection ----------
def detect_patterns(df: pd.DataFrame) -> List[str]:
    patterns = []
    if len(df) < 3:
        return patterns
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    idx = -2
    try:
        if TALIB_AVAILABLE:
            val = talib.CDLENGULFING(o, h, l, c)[idx]
            if val > 0:
                patterns.append("Bullish Engulfing")
            elif val < 0:
                patterns.append("Bearish Engulfing")
            if talib.CDLHAMMER(o, h, l, c)[idx] > 0:
                patterns.append("Hammer")
            if talib.CDLINVERTEDHAMMER(o, h, l, c)[idx] > 0:
                patterns.append("Inverted Hammer")
            if talib.CDLSHOOTINGSTAR(o, h, l, c)[idx] > 0:
                patterns.append("Shooting Star")
            if talib.CDLMORNINGSTAR(o, h, l, c, penetration=0.3)[idx] > 0:
                patterns.append("Morning Star")
    except Exception:
        logger.exception("TA-Lib pattern check failed; using fallback")
    if not patterns:
        # fallback simple checks (permissive)
        prev_o, prev_c = df["open"].iloc[-3], df["close"].iloc[-3]
        cur_o, cur_c, cur_h, cur_l = df["open"].iloc[-2], df["close"].iloc[-2], df["high"].iloc[-2], df["low"].iloc[-2]
        body = abs(cur_c - cur_o)
        upper_wick = cur_h - max(cur_c, cur_o)
        lower_wick = min(cur_c, cur_o) - cur_l
        if (prev_c < prev_o) and (cur_c > cur_o) and (cur_c >= prev_o) and (cur_o <= prev_c):
            patterns.append("Bullish Engulfing")
        if (prev_c > prev_o) and (cur_c < cur_o) and (cur_c <= prev_o) and (cur_o >= prev_c):
            patterns.append("Bearish Engulfing")
        if body > 0 and (lower_wick >= 2 * body) and (upper_wick <= 0.5 * body):
            patterns.append("Hammer")
        if body > 0 and (upper_wick >= 2 * body) and (lower_wick <= 0.5 * body):
            if cur_c < cur_o:
                patterns.append("Shooting Star")
            else:
                patterns.append("Inverted Hammer")
        if len(df) >= 4:
            c1_o, c1_c = df["open"].iloc[-4], df["close"].iloc[-4]
            c2_o, c2_c = df["open"].iloc[-3], df["close"].iloc[-3]
            c3_o, c3_c = df["open"].iloc[-2], df["close"].iloc[-2]
            if (c1_c < c1_o) and (abs(c2_c - c2_o) < 0.35 * abs(c1_o - c1_c)) and (c3_c > c3_o) and (c3_c > ((c1_o + c1_c)/2)):
                patterns.append("Morning Star")
    return patterns

# ---------- Scalping ensemble & RF features ----------
def generate_signals(df: pd.DataFrame) -> Dict[str, Any]:
    res = {"votes": {}, "direction": None, "confidence": 0.0, "explain": ""}
    if len(df) < 40:
        res["explain"] = "insufficient data"
        return res

    df = calculate_indicators(df)
    close = df["close"].values
    idx = -2
    c = float(close[idx])
    ema9_prev, ema21_prev = float(df["EMA9"].iloc[-3]), float(df["EMA21"].iloc[-3])
    ema9_curr, ema21_curr, ema50_curr = float(df["EMA9"].iloc[idx]), float(df["EMA21"].iloc[idx]), float(df["EMA50"].iloc[idx])
    rsi_prev, rsi_curr = float(df["RSI"].iloc[-3]), float(df["RSI"].iloc[idx])
    bb_u, bb_m, bb_l = float(df["BB_upper"].iloc[idx]), float(df["BB_middle"].iloc[idx]), float(df["BB_lower"].iloc[idx])
    atr_s = float(atr(df).iloc[idx])

    votes_buy = votes_sell = 0.0
    weights = {"ema": 1.0, "rsi": 0.9, "bbreak": 1.0, "meanrev": 0.8}

    # EMA crossover
    ema_signal = 0
    if (ema9_prev <= ema21_prev) and (ema9_curr > ema21_curr) and (c > ema50_curr):
        votes_buy += weights["ema"]; ema_signal = 1
    if (ema9_prev >= ema21_prev) and (ema9_curr < ema21_curr) and (c < ema50_curr):
        votes_sell += weights["ema"]; ema_signal = -1
    res["votes"]["ema"] = ema_signal

    # RSI bounce
    rsi_signal = 0
    RSI_OVERSOLD, RSI_OVERBOUGHT = 30.0, 70.0
    if (rsi_prev < RSI_OVERSOLD) and (rsi_curr > RSI_OVERSOLD):
        votes_buy += weights["rsi"]; rsi_signal = 1
    if (rsi_prev > RSI_OVERBOUGHT) and (rsi_curr < RSI_OVERBOUGHT):
        votes_sell += weights["rsi"]; rsi_signal = -1
    res["votes"]["rsi"] = rsi_signal

    # Bollinger breakout
    bb_signal = 0
    atr_min = (0.0005 * c) if c > 1 else 0.01
    if (atr_s > atr_min) and (c > bb_u):
        votes_buy += weights["bbreak"]; bb_signal = 1
    if (atr_s > atr_min) and (c < bb_l):
        votes_sell += weights["bbreak"]; bb_signal = -1
    res["votes"]["bbreak"] = bb_signal

    # Mean reversion (z-score relative to BB middle)
    meanrev_signal = 0
    denom = (bb_u - bb_m) if (bb_u - bb_m) != 0 else 1.0
    z = (c - bb_m) / denom
    if (z > 2.0) and (rsi_curr > 65):
        votes_sell += weights["meanrev"]; meanrev_signal = -1
    if (z < -2.0) and (rsi_curr < 35):
        votes_buy += weights["meanrev"]; meanrev_signal = 1
    res["votes"]["meanrev"] = meanrev_signal

    total_possible = sum(weights.values())
    if (votes_buy + votes_sell) == 0:
        res["confidence"] = 0.0
        res["direction"] = None
    else:
        if votes_buy > votes_sell:
            res["direction"] = "BUY"
            res["confidence"] = min(100.0, 100.0 * (votes_buy / total_possible))
        else:
            res["direction"] = "SELL"
            res["confidence"] = min(100.0, 100.0 * (votes_sell / total_possible))
    res["explain"] = f"buy={votes_buy:.2f}, sell={votes_sell:.2f}, z={z:.2f}, atr={atr_s:.6f}"
    return res

def prepare_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds features for ML predictor: EMA9, EMA21, EMA50, RSI, BB width, past returns.
    Target: next period return (close_next / close - 1)
    """
    df = calculate_indicators(df.copy())
    df["ret1"] = df["close"].pct_change().fillna(0)
    df["bb_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"].replace(0, np.nan)
    df["bb_width"] = df["bb_width"].fillna(0)
    # rolling returns
    for w in (1, 3, 5):
        df[f"r_{w}"] = df["close"].pct_change(periods=w).fillna(0)
    # target
    df["target"] = df["close"].shift(-1) / df["close"] - 1.0
    df = df.dropna().reset_index(drop=True)
    feature_cols = ["EMA9", "EMA21", "EMA50", "RSI", "bb_width", "r_1", "r_3", "r_5"]
    return df[feature_cols + ["target"]]

def train_rf_model(df: pd.DataFrame, save_path: str = MODEL_FILE) -> Dict[str, Any]:
    """
    Train RandomForestRegressor on prepared features and save model.
    Returns training metrics.
    """
    data = prepare_ml_features(df)
    if data.shape[0] < 50:
        return {"status": "error", "reason": "not enough data to train"}
    X = data.drop(columns=["target"]).values
    y = data["target"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # simple metrics
    mse = float(np.mean((y_test - y_pred) ** 2))
    dump(model, save_path)
    return {"status": "ok", "mse": mse, "trained_samples": len(y_train)}

def load_rf_model(path: str = MODEL_FILE) -> Optional[RandomForestRegressor]:
    try:
        model = load(path)
        return model
    except Exception:
        logger.info("No saved model found at %s", path)
        return None

def predict_with_model(df: pd.DataFrame, model: RandomForestRegressor) -> Dict[str, float]:
    """
    Predict next-period return using model. Returns {'pred': float, 'pct': float}
    """
    data = prepare_ml_features(df)
    if data.shape[0] < 1:
        return {"pred": 0.0, "pct": 0.0}
    X = data.drop(columns=["target"]).values
    last_X = X[-1].reshape(1, -1)
    pred = model.predict(last_X)[0]
    last_close = float(df["close"].iloc[-2])
    pred_price = last_close * (1.0 + pred)
    pred_pct = pred * 100.0
    return {"pred": float(pred_price), "pct": float(pred_pct)}

# ---------- Backtest ----------
def backtest_ensemble(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Walk through df (older->newer), generate signals at each point, and simulate one-period exit (next close).
    Returns winrate and expectancy (average return per trade).
    """
    df = df.copy().reset_index(drop=True)
    df = calculate_indicators(df)
    trades = []
    # iterate from index 30 to len-2 (since we use second-last candle and next close)
    for i in range(30, len(df) - 1):
        window = df.iloc[: i + 1].reset_index(drop=True)
        sig = generate_signals(window)
        if not sig["direction"]:
            continue
        entry = float(window["close"].iloc[-1])
        next_close = float(df["close"].iloc[i + 1])  # exit next close
        ret = (next_close - entry) / entry if sig["direction"] == "BUY" else (entry - next_close) / entry
        trades.append({"direction": sig["direction"], "ret": ret, "confidence": sig["confidence"]})
    if not trades:
        return {"trades": 0, "winrate": 0.0, "expectancy": 0.0}
    wins = sum(1 for t in trades if t["ret"] > 0)
    winrate = wins / len(trades) * 100.0
    expectancy = np.mean([t["ret"] for t in trades]) * 100.0  # percent
    return {"trades": len(trades), "winrate": round(winrate, 2), "expectancy_pct": round(expectancy, 4)}

# ---------- Position sizing ----------
def position_size_suggestion(account_balance: float, risk_pct: float, stop_distance_pct: float, price: float) -> Dict[str, float]:
    risk_amount = account_balance * (risk_pct / 100.0)
    if stop_distance_pct <= 0:
        return {"size_units": 0.0, "risk_amount": round(risk_amount, 2)}
    dollars_per_unit = price * (stop_distance_pct / 100.0)
    if dollars_per_unit <= 0:
        return {"size_units": 0.0, "risk_amount": round(risk_amount, 2)}
    units = risk_amount / dollars_per_unit
    return {"size_units": float(units), "risk_amount": round(risk_amount, 2)}

# ---------- Formatting & Live message ----------
def format_report_and_alert(df: pd.DataFrame, symbol: str, model: Optional[RandomForestRegressor]) -> str:
    if len(df) < 10:
        return f"Insufficient data for {symbol}"
    df = calculate_indicators(df)
    last = df.iloc[-2]
    patterns = detect_patterns(df)
    sig = generate_signals(df)
    model_pred = None
    if model:
        try:
            model_pred = predict_with_model(df, model)
        except Exception:
            logger.exception("Model prediction failed")
            model_pred = None

    time_str = pd.to_datetime(last["datetime"]).strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"📊 *{symbol}* — {time_str}")
    lines.append(f"💰 Price: `{last['close']:.5f}`")
    lines.append(f"📈 EMA9: `{last['EMA9']:.5f}` | EMA21: `{last['EMA21']:.5f}` | EMA50: `{last['EMA50']:.5f}`")
    lines.append(f"📊 RSI(14): `{last['RSI']:.2f}`")
    lines.append(f"🎯 BB Upper: `{last['BB_upper']:.5f}` | Mid: `{last['BB_middle']:.5f}` | Lower: `{last['BB_lower']:.5f}`")
    lines.append(f"🕯 Patterns: `{', '.join(patterns) if patterns else 'None'}`")
    # include ensemble info
    if sig["direction"]:
        lines.append(f"🔔 Ensemble direction: *{sig['direction']}* ({sig['confidence']}% confidence)")
        lines.append(f"ℹ️ {sig['explain']}")
    else:
        lines.append("ℹ️ Ensemble: No strong direction")
    # model
    if model_pred:
        lines.append(f"🔮 Model predicted next price: `{model_pred['pred']:.5f}` ({model_pred['pct']:.2f}%)")
    return "\n".join(lines)

# ---------- Async live task ----------
async def live_task(chat_id: int):
    logger.info("Live task started for chat %s", chat_id)
    # load model once per task
    model = await asyncio.to_thread(load_rf_model)
    while True:
        state = chat_state.get(chat_id)
        if not state or not state.get("running"):
            logger.info("Stopping live task for chat %s", chat_id)
            break
        symbol = state.get("symbol", DEFAULT_SYMBOL)
        df = await asyncio.to_thread(fetch_ohlcv_sync, symbol, TD_INTERVAL, TD_OUTPUTSIZE)
        if df is None:
            try:
                app = state.get("app")
                if app:
                    await app.bot.send_message(chat_id=chat_id, text=f"❌ Failed to fetch {symbol}. Retrying in {POLL_SECONDS}s.")
            except Exception:
                logger.exception("Failed to send fetch-failure")
            await asyncio.sleep(POLL_SECONDS)
            continue
        text = format_report_and_alert(df, symbol, model)
        try:
            app = state.get("app")
            if app:
                await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
        except Exception:
            logger.exception("Failed to send message")
        # if ensemble produced actionable signal with confidence, also send an explicit BUY/SELL alert with suggested size
        sig = generate_signals(df)
        if sig["direction"] and sig["confidence"] >= SIGNAL_CONFIDENCE_THRESHOLD:
            last_close = float(df["close"].iloc[-2])
            bb_mid = float(df["BB_middle"].iloc[-2])
            stop_distance_pct = abs((bb_mid - last_close) / last_close) * 100.0
            if stop_distance_pct < 0.1:
                stop_distance_pct = 0.2
            size = position_size_suggestion(ACCOUNT_BALANCE, MAX_RISK_PCT, stop_distance_pct, last_close)
            alert = (
                f"🚨 *{sig['direction']}* signal — confidence {sig['confidence']}%\n"
                f"Price: `{last_close:.5f}` Predicted: `{(predict_with_model(df, model)['pred'] if model else last_close):.5f}`\n"
                f"Suggested size: {size['size_units']:.2f} units (risk ${size['risk_amount']})\n"
                f"{sig['explain']}\n"
                f"⚠️ This is a signal — not financial advice."
            )
            try:
                app = state.get("app")
                if app:
                    await app.bot.send_message(chat_id=chat_id, text=alert, parse_mode="Markdown")
            except Exception:
                logger.exception("Failed to send alert")
        await asyncio.sleep(POLL_SECONDS)

# ---------- Commands ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    symbol = normalize_symbol(DEFAULT_SYMBOL)
    if chat_id not in chat_state:
        chat_state[chat_id] = {"symbol": symbol, "running": False, "task": None, "app": context.application}
    state = chat_state[chat_id]
    state["app"] = context.application
    if state.get("running"):
        await update.message.reply_text("✅ Live updates already running. Use /stop to stop.")
        return
    state["running"] = True
    state["symbol"] = symbol
    state["task"] = asyncio.create_task(live_task(chat_id))
    await update.message.reply_text(f"▶️ Started live updates for *{symbol}* every {POLL_SECONDS}s.", parse_mode="Markdown")

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    state = chat_state.get(chat_id)
    if not state or not state.get("running"):
        await update.message.reply_text("ℹ️ No live updates running.")
        return
    state["running"] = False
    t = state.get("task")
    if t:
        t.cancel()
    state["task"] = None
    await update.message.reply_text("⏹ Stopped live updates.")

async def cmd_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Usage: /symbol SYMBOL (e.g. /symbol XAUUSD or /symbol XAU/USD)")
        return
    raw = " ".join(context.args)
    sym = normalize_symbol(raw)
    df = await asyncio.to_thread(fetch_ohlcv_sync, sym, TD_INTERVAL, 10)
    if df is None:
        await update.message.reply_text(f"❌ Could not validate symbol `{sym}`. Try XAU/USD, EUR/USD.", parse_mode="Markdown")
        return
    if chat_id not in chat_state:
        chat_state[chat_id] = {"symbol": sym, "running": False, "task": None, "app": context.application}
    else:
        chat_state[chat_id]["symbol"] = sym
        chat_state[chat_id]["app"] = context.application
    await update.message.reply_text(f"🔁 Symbol set to *{sym}* for this chat.", parse_mode="Markdown")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    s = chat_state.get(chat_id)
    if not s:
        await update.message.reply_text("No state for this chat. Use /start.")
        return
    await update.message.reply_text(f"Symbol: {s.get('symbol')}\nRunning: {s.get('running')}")

async def cmd_train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    # retrain model using recent data for this chat's symbol
    state = chat_state.get(chat_id)
    if not state:
        await update.message.reply_text("Start the bot first in this chat (use /start) to set symbol then run /train.")
        return
    symbol = state.get("symbol", DEFAULT_SYMBOL)
    await update.message.reply_text(f"🔁 Training RF model using recent {symbol} data... This may take a moment.")
    # fetch large history
    df = await asyncio.to_thread(fetch_ohlcv_sync, symbol, TD_INTERVAL, 2000)
    if df is None:
        await update.message.reply_text("❌ Failed to fetch data for training.")
        return
    # run training in thread
    res = await asyncio.to_thread(train_rf_model, df, MODEL_FILE)
    if res.get("status") == "ok":
        await update.message.reply_text(f"✅ Trained RF model. MSE: {res['mse']:.6f}. Trained samples: {res['trained_samples']}")
    else:
        await update.message.reply_text(f"❌ Training failed: {res.get('reason', 'unknown')}")

async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    days = 2
    if context.args:
        try:
            days = int(context.args[0])
        except Exception:
            await update.message.reply_text("Usage: /backtest N  (N = days)")
            return
    # estimate points: days * 24 * 60 (1-min)
    points = min(20000, max(200, days * 24 * 60))
    state = chat_state.get(chat_id)
    symbol = state.get("symbol", DEFAULT_SYMBOL) if state else DEFAULT_SYMBOL
    await update.message.reply_text(f"🔍 Running backtest for {symbol} over last {days} days (~{points} points). This may take a moment.")
    df = await asyncio.to_thread(fetch_ohlcv_sync, symbol, TD_INTERVAL, points)
    if df is None:
        await update.message.reply_text("❌ Failed to fetch historical data for backtest.")
        return
    res = await asyncio.to_thread(backtest_ensemble, df)
    await update.message.reply_text(f"📈 Backtest results for {symbol}:\nTrades: {res['trades']}\nWinrate: {res['winrate']}%\nExpectancy (avg % per trade): {res['expectancy_pct']}%")

async def cmd_setbalance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ACCOUNT_BALANCE
    if not context.args:
        await update.message.reply_text(f"Usage: /setbalance 1000\nCurrent: {ACCOUNT_BALANCE}")
        return
    try:
        ACCOUNT_BALANCE = float(context.args[0])
        await update.message.reply_text(f"Account balance set to {ACCOUNT_BALANCE:.2f}")
    except Exception:
        await update.message.reply_text("Invalid number")

async def cmd_setrisk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global MAX_RISK_PCT
    if not context.args:
        await update.message.reply_text(f"Usage: /setrisk 1.0\nCurrent: {MAX_RISK_PCT}%")
        return
    try:
        MAX_RISK_PCT = float(context.args[0])
        await update.message.reply_text(f"Max risk per trade set to {MAX_RISK_PCT:.2f}%")
    except Exception:
        await update.message.reply_text("Invalid number")

# ---------- Model helpers for load/save accessible from live task ----------
def load_rf_model(path: str = MODEL_FILE) -> Optional[RandomForestRegressor]:
    try:
        m = load(path)
        logger.info("Loaded RF model from %s", path)
        return m
    except Exception:
        logger.info("No RF model at %s; run /train to create one", path)
        return None

# ---------- Main ----------
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("symbol", cmd_symbol))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("train", cmd_train))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(CommandHandler("setbalance", cmd_setbalance))
    app.add_handler(CommandHandler("setrisk", cmd_setrisk))

    logger.info("Starting bot (polling)...")
    app.run_polling()

if __name__ == "__main__":
    main()
