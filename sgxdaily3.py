# SGX Round‑2 Scanner — Yahoo Finance OHLCV (20D) & Signals + Economics + Plan
# Role & objective:
#   Meticulous SGX strategist. Use Yahoo Finance (and SGX site for cross-check if available).
#   Design actions that aim to beat HOLD by ≥10% by 4 Mar 2026, strictly enforcing rules.
#
# Key features (aligned to the prompt):
#   • Pulls last 20 trading days of OHLCV for each ticker from Yahoo (yfinance).
#   • Computes RSI(14), ATR(14), SMA20/50/200, MACD(12,26,9), 20D High (close), 20D Avg Vol (excl. today).
#   • Evaluates the 6 Round‑2 signals + exception flags.
#   • Applies economics gate: Net Risk = 1.5×ATR + 2.4%×Price; Net Reward = 3×ATR – 2.4%×Price; Net R:R ≥ 2.
#   • Produces per‑ticker table with required columns, On‑Watch triggers (Buy‑if), and Action Plan.
#   • Portfolio comparison: HOLD vs Action Plan (price-only + trailing 12M dividend proxy).
#   • CSV saved with SGT timestamp for daily runs.
#
# Notes:
#   • Breakout confirmation uses the last official close and prior 20-day averages (ex‑today) as required.
#   • Intraday prices are not used to confirm signals (we stick to daily close).
#   • SGX cross-check is optional; if not available, we default to Yahoo close and flag if stale.
#
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta
import pytz
import math
from typing import Dict, Tuple, Optional

# =========================
# 0) Global settings (edit)
# =========================

SGT = pytz.timezone("Asia/Singapore")
now_sgt_dt = datetime.now(SGT)
today_str = now_sgt_dt.strftime("%Y-%m-%d")
timestamp_str = now_sgt_dt.strftime("%Y-%m-%d_%H%M%S")  # for CSVs

# Universe (fixed watchlist)
TICKERS = [
    "D05.SI","U11.SI","O39.SI","B61.SI","ME8U.SI","BUOU.SI","S59.SI","S63.SI",
    "S58.SI","U13.SI","J69U.SI","S68.SI","C07.SI","HMN.SI","9CI.SI","N2IU.SI","C38U.SI"
]

# Default dry powder (can be overridden by the operator)
DEFAULT_DRY_POWDER = 200_000.0

# Current portfolio (quantities) — from your table
# (We only need quantities to compute exposures and HOLD value. Costs are not required here.)
PORTFOLIO_QTY = {
    "C07.SI": 1000,
    "B61.SI": 263500,
    "D05.SI": 5000,
    "ME8U.SI": 98279,
    "S59.SI": 5000,
    "U11.SI": 12246,
    "U13.SI": 8000,
    "J69U.SI": 28000,
    "O39.SI": 57,
    "BUOU.SI": 170198,
    "HMN.SI": 1300,
    "S68.SI": 0,
    "S58.SI": 21200,
    "9CI.SI": 2000,
    "N2IU.SI": 10000,
    "C38U.SI": 1062,
    "S63.SI": 4000,
}

# Trading cost and risk parameters
FEES_RT = 0.024      # 1.2% in + 1.2% out (round-trip)
STOP_MULT = 1.5      # default stop = 1.5× ATR
TP_MULT = 3.0        # base check to +3× ATR tier

# =========================
# 1) Indicator helpers
# =========================

def ema(s: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI(14) using Wilder-style smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (12,26,9)."""
    macd_line = ema(close, 12) - ema(close, 26)
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR(14) using EMA of True Range."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def bullish_divergence(close: pd.Series, rsi_series: pd.Series, lookback: int = 40) -> bool:
    """
    Simple bullish divergence detector:
    - Find two recent pivot lows (1-bar look-around) within lookback.
    - Price makes Lower Low (LL), RSI makes Higher Low (HL).
    - Latest close > previous close (confirmation).
    """
    c = close.tail(lookback).copy()
    r = rsi_series.loc[c.index]
    pivots = []
    vals = c.values
    for i in range(1, len(vals)-1):
        if vals[i] < vals[i-1] and vals[i] < vals[i+1]:
            pivots.append(i)
    if len(pivots) < 2:
        return False
    i1, i2 = pivots[-2], pivots[-1]
    p1, p2 = vals[i1], vals[i2]
    r1, r2 = r.iloc[i1], r.iloc[i2]
    latest_up = c.iloc[-1] > c.iloc[-2]
    return (p2 < p1) and (r2 > r1) and latest_up

def support_bounce(df: pd.DataFrame, sma50_name: str = "SMA50") -> bool:
    """
    Conservative support-bounce:
    - Bullish candle today (Close > Open).
    - Low touches/sub-50DMA but Close >= 50DMA.
    """
    if df.shape[0] == 0 or sma50_name not in df.columns:
        return False
    bullish = df["Close"].iloc[-1] > df["Open"].iloc[-1]
    touch = (df["Low"].iloc[-1] <= df[sma50_name].iloc[-1]) and (df["Close"].iloc[-1] >= df[sma50_name].iloc[-1])
    return bool(bullish and touch)

def ttm_dividends(ticker: str) -> float:
    """
    Trailing 12M dividends per share from Yahoo actions.
    Returns total amount (SGD per share) over last 365 days (best effort).
    """
    try:
        actions = yf.Ticker(ticker).actions
        if isinstance(actions, pd.DataFrame) and "Dividends" in actions.columns and len(actions.index) > 0:
            cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=365)
            divs = actions[actions.index >= cutoff]["Dividends"].sum()
            return float(divs) if not np.isnan(divs) else 0.0
    except Exception:
        pass
    return 0.0

# Helper for safe scalar extraction
def _val(x) -> float:
    if hasattr(x, "iloc"):
        return float(x.iloc[0])
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)

# =========================
# 2) Data pull + signals
# =========================

rows = []
errors = []
last_close_dates = []  # to assess staleness / non-trading day

for t in TICKERS:
    try:
        # Robust fetch with retries (Yahoo can be finicky)
        df = pd.DataFrame()
        last_err = None
        for attempt in range(3):
            try:
                df = yf.download(t, period="400d", interval="1d", auto_adjust=False, progress=False, threads=False)
                if df is not None and not df.empty:
                    break
            except Exception as e:
                last_err = e
            time.sleep(0.7 * (attempt + 1))
        if (df is None) or df.empty:
            try:
                df = yf.Ticker(t).history(period="400d", interval="1d", auto_adjust=False)
            except Exception as e:
                last_err = e
        if (df is None) or df.empty:
            msg = "empty dataframe from Yahoo"
            if last_err is not None:
                msg += f" (last error: {last_err})"
            errors.append(f"{t}: {msg}")
            continue

        # Standardize columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna().copy()

        # Keep note of the last bar's date (for non-trading day check)
        last_index_ts = df.index[-1]
        # Convert last index (UTC) to SGT date for reporting
        if last_index_ts.tzinfo is None:
            last_index_ts = last_index_ts.tz_localize("UTC")
        last_sgt_date = last_index_ts.tz_convert(SGT).date()
        last_close_dates.append(last_sgt_date)

        # Indicators
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()
        df["RSI14"] = rsi(df["Close"], 14)
        macd_line, macd_sig, macd_hist = macd(df["Close"])
        df["MACD"], df["MACDsig"], df["MACDhist"] = macd_line, macd_sig, macd_hist
        df["ATR14"] = atr(df, 14)

        # 20-day references (exclude today to avoid look-ahead)
        df["High20_Close"] = df["Close"].rolling(20).max().shift(1)
        df["Vol20_Avg"]   = df["Volume"].rolling(20).mean().shift(1)

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else last

        price     = _val(last["Close"])
        rsi14     = _val(last["RSI14"])
        atr14     = _val(last["ATR14"])
        sma20     = _val(last["SMA20"])
        sma50     = _val(last["SMA50"])
        sma200    = _val(last["SMA200"])
        hi20      = _val(last["High20_Close"])
        vol20     = _val(last["Vol20_Avg"])
        vol_curr  = _val(last["Volume"])
        macd_val  = _val(last["MACD"])
        macd_sigv = _val(last["MACDsig"])
        macd_histv= _val(last["MACDhist"])

        # 6 Strong Technical Signals
        dipped = (df["RSI14"].iloc[-21:-1] < 35).any() if len(df) >= 21 else (df["RSI14"].iloc[:-1] < 35).any()
        rsi_rule      = (rsi14 > 50.0) and bool(dipped)
        ma_trend_rule = (price > sma20) and (sma20 > sma50)
        breakout_rule = (price > hi20) and (vol_curr >= 1.5 * vol20)
        support_rule  = support_bounce(df)
        if len(df) >= 2:
            macd_rule = (df["MACDhist"].iloc[-1] > 0) and (df["MACDhist"].iloc[-2] > 0) \
                        and (df["MACD"].iloc[-1] > df["MACDsig"].iloc[-1]) \
                        and (df["MACD"].iloc[-2] > df["MACDsig"].iloc[-2])
        else:
            macd_rule = False
        div_rule = bullish_divergence(df["Close"], df["RSI14"], lookback=40)
        signals_met = int(sum([rsi_rule, ma_trend_rule, breakout_rule, support_rule, macd_rule, div_rule]))

        # Exception flags (for manual catalyst validation)
        if len(df) >= 2:
            rsi_prev = _val(df["RSI14"].iloc[-2])
            prev_close = _val(prev["Close"])
            curr_open  = _val(last["Open"])
            rsi_exception = (rsi_prev < 30) and (price > curr_open) and (price > prev_close)
        else:
            rsi_exception = False
        vol_exception  = (price > hi20) and (vol_curr >= 2.0 * vol20)
        exception_flag = "RSI<30 bounce" if rsi_exception else ("2×Vol breakout" if vol_exception else "")

        # Tech-only decision
        trade_tech = "BUY" if signals_met >= 2 else "No-Trade"

        # Economics: per-share risk/reward after fees (2.4% round-trip)
        net_risk   = STOP_MULT * atr14 + FEES_RT * price
        net_reward = TP_MULT   * atr14 - FEES_RT * price
        net_rr     = (net_reward / net_risk) if net_risk > 0 else np.nan
        stop_pct   = (STOP_MULT * atr14 / price) if price > 0 else np.nan
        tight_ok   = bool(stop_pct <= 0.05) if not np.isnan(stop_pct) else False

        # Final decision (economics gate + exception rule)
        def decide_final() -> str:
            rr_ok = (net_rr >= 2.0) and (net_reward > 0)
            if signals_met >= 2 and rr_ok:
                return "BUY"
            if (exception_flag != "") and tight_ok and rr_ok:
                return "BUY-Exception"
            return "No-Trade"

        trade_final = decide_final()

        # Last dividend ex-date & next earnings (best effort via Yahoo)
        last_ex = None
        next_earn = None
        try:
            actions = yf.Ticker(t).actions
            if isinstance(actions, pd.DataFrame) and "Dividends" in actions.columns and len(actions.index) > 0:
                last_ex = str(actions.index.max().date())
        except Exception:
            pass
        try:
            cal = yf.Ticker(t).calendar
            if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                ed = cal.loc["Earnings Date"].values
                if len(ed) > 0:
                    next_earn = str(pd.to_datetime(ed[0]).date())
        except Exception:
            pass

        # Row assembly
        rows.append({
            "Ticker": t,
            "Price": round(price, 4),
            "RSI-14": round(rsi14, 2),
            "ATR(14)": round(atr14, 4),
            "SMA20": round(sma20, 4),
            "SMA50": round(sma50, 4),
            "SMA200": round(sma200, 4),
            "20D High (close)": round(hi20, 4),
            "20D Avg Vol": int(vol20) if not np.isnan(vol20) else None,
            "MACD": round(macd_val, 4),
            "MACD Signal": round(macd_sigv, 4),
            "MACD Hist": round(macd_histv, 4),
            "RSI rule": bool(rsi_rule),
            "MA trend rule": bool(ma_trend_rule),
            "Breakout rule": bool(breakout_rule),
            "Support bounce rule": bool(support_rule),
            "MACD rule (≥2D)": bool(macd_rule),
            "Divergence rule": bool(div_rule),
            "Signals (1–6)": signals_met,
            "Trade (Tech)": trade_tech,
            "Exception flag": exception_flag,
            "Net Risk": round(net_risk, 4),
            "Net Reward": round(net_reward, 4),
            "Net R:R": round(net_rr, 4) if not np.isnan(net_rr) else None,
            "Stop% (1.5×ATR)": round(stop_pct, 4) if not np.isnan(stop_pct) else None,
            "Tight stop OK": bool(tight_ok),
            "Breakout Px": round(hi20, 4),
            "Breakout Vol": int(math.ceil(1.5 * vol20)) if vol20 and not np.isnan(vol20) else None,
            "Last ex-div": last_ex,
            "Next earnings": next_earn,
            # For portfolio & carry
            "TTM Div/Share": round(ttm_dividends(t), 4),
            "Last Close Date (SGT)": str(last_sgt_date),
        })

    except Exception as e:
        errors.append(f"{t}: {e}")

# If no data retrieved, abort gracefully
if not rows:
    print("No data processed. Errors:")
    for e in errors:
        print(" -", e)
    raise SystemExit(1)

out = pd.DataFrame(rows)

# =========================
# 3) Non‑trading day / staleness note
# =========================
# If today's SGT date != most recent close date, we note it (weekends/holidays).
most_recent_close = max(pd.to_datetime(out["Last Close Date (SGT)"]).dt.date)
today_date = now_sgt_dt.date()
non_trading_msg = None
if most_recent_close != today_date:
    non_trading_msg = (f"Note: Today ({today_date}) is a non‑trading day or the latest official close "
                       f"is {most_recent_close}. All signals are based on the most recent SGX close.")

# =========================
# 4) Final decision & display table
# =========================

# Trade (Final) after economics gate
def decide_final_from_row(r: pd.Series) -> str:
    rr_ok = (r["Net R:R"] is not None) and (r["Net R:R"] >= 2.0) and (r["Net Reward"] > 0)
    if (r["Signals (1–6)"] >= 2) and rr_ok:
        return "BUY"
    if (r["Exception flag"] != "") and r["Tight stop OK"] and rr_ok:
        return "BUY-Exception"
    return "No-Trade"

out["Trade (Final)"] = out.apply(decide_final_from_row, axis=1)
out["20/50/200D MAs"] = out.apply(lambda r: f"{r['SMA20']}/{r['SMA50']}/{r['SMA200']}", axis=1)

# Sort: BUY first (Final), then signals, then ticker
sort_cols = ["Trade (Final)", "Signals (1–6)", "Ticker"]
out = out.sort_values(sort_cols, ascending=[True, False, True]).reset_index(drop=True)

# Columns required by the prompt for the per‑ticker table
display_cols = [
    "Ticker","Price","RSI-14","ATR(14)","20/50/200D MAs","20D High (close)","20D Avg Vol",
    "MACD","RSI rule","MA trend rule","Breakout rule","Support bounce rule","MACD rule (≥2D)","Divergence rule",
    "Signals (1–6)","Trade (Tech)","Net Risk","Net Reward","Net R:R","Breakout Px","Breakout Vol",
    "Exception flag","Tight stop OK","Last ex-div","Next earnings","Trade (Final)"
]

print("\n" + "="*88)
print(f"SGX Round‑2 scan (Yahoo OHLCV) — {today_str} {now_sgt_dt.strftime('%H:%M:%S')} SGT")
if non_trading_msg:
    print(non_trading_msg)
print("="*88 + "\n")
print(out[display_cols].to_string(index=False))

# =========================
# 5) On‑Watch trigger list (≥2 signals but R:R<2 or missing breakout)
# =========================
on_watch = out[(out["Signals (1–6)"] >= 2) & (out["Trade (Final)"] == "No-Trade")]
if not on_watch.empty:
    print("\nON‑WATCH (≥2 signals but R:R<2 or missing breakout):")
    for _, r in on_watch.iterrows():
        print(f"• {r['Ticker']}: Buy‑if Close > {r['Breakout Px']:.4f} AND Volume ≥ {r['Breakout Vol']} "
              f"(adds 3rd signal). Current Net R:R {r['Net R:R']:.2f}×")

# =========================
# 6) Portfolio comparison: HOLD vs Action Plan (price‑only & TTM dividends proxy)
# =========================

# Helper current market value using Yahoo close
def portfolio_value_from_table(table: pd.DataFrame, qty_map: Dict[str, int]) -> float:
    mv = 0.0
    for t, q in qty_map.items():
        row = table.loc[table["Ticker"] == t]
        if not row.empty:
            px = float(row["Price"].iloc[0])
            mv += px * float(q)
    return mv

# Compute TTM dividends (proxy for "incl. dividends")
def portfolio_ttm_dividends(table: pd.DataFrame, qty_map: Dict[str, int]) -> float:
    total = 0.0
    for t, q in qty_map.items():
        row = table.loc[table["Ticker"] == t]
        if not row.empty:
            dps = float(row["TTM Div/Share"].iloc[0])
            total += dps * float(q)
    return total

hold_value = portfolio_value_from_table(out, PORTFOLIO_QTY)
hold_div_ttm = portfolio_ttm_dividends(out, PORTFOLIO_QTY)

# Build Action Plan: buy only those that pass final gate (BUY / BUY-Exception)
qualified = out[out["Trade (Final)"].str.contains("BUY", na=False)].copy()

# Default allocation: 20–25% per qualified (use midpoint 22.5% unless user overrides)
DRY_POWDER = DEFAULT_DRY_POWDER
ALLOC_FRAC = 0.225

plan_buys = []
spent_cash = 0.0

def max_shares_with_fee(alloc: float, trigger_price: float) -> int:
    # Shares = floor( allocation / (trigger_price × 1.012) ) to embed entry fee
    denom = trigger_price * 1.012
    return int(math.floor(alloc / denom)) if denom > 0 else 0

# Helper to get trigger price: if breakout rule is already true, use last close; else use Breakout Px
def trigger_price_for_row(r: pd.Series) -> float:
    return float(r["Price"]) if r["Breakout rule"] else float(r["Breakout Px"])

# Build planned orders, enforce ≤70% single‑name exposure (post‑trade)
for _, r in qualified.iterrows():
    ticker = r["Ticker"]
    # Skip if ticker missing critical fields
    if pd.isna(r["Price"]) or pd.isna(r["ATR(14)"]):
        continue

    trig_px = trigger_price_for_row(r)
    alloc = DRY_POWDER * ALLOC_FRAC
    shares = max_shares_with_fee(alloc, trig_px)
    if shares <= 0:
        continue

    # Exposure cap check: existing holding + new shares must be ≤70% of post‑trade portfolio
    existing_shares = PORTFOLIO_QTY.get(ticker, 0)
    existing_value  = existing_shares * float(r["Price"])
    add_value       = shares * trig_px
    post_hold_value = hold_value + add_value  # ignoring cash change for exposure test
    if post_hold_value > 0 and (existing_value + add_value) / post_hold_value > 0.70:
        # Skip if exceeds 70%
        continue

    stop_lvl = trig_px - STOP_MULT * float(r["ATR(14)"])
    t1 = trig_px + 2.0 * float(r["ATR(14)"])
    t2 = trig_px + 3.0 * float(r["ATR(14)"])
    plan_buys.append({
        "Ticker": ticker,
        "Trigger Px": round(trig_px, 4),
        "Shares": shares,
        "Alloc SGD": round(shares * trig_px * 1.012, 2),  # include entry fee
        "Init Stop": round(stop_lvl, 4),
        "Stop %": round((STOP_MULT * float(r["ATR(14)"]) / trig_px) * 100.0, 2),
        "TP +2×ATR": round(t1, 4),
        "TP +3×ATR": round(t2, 4),
        "Trail": "20‑DMA or breakeven after 2nd scale",
    })
    spent_cash += shares * trig_px * 1.012

# Compute plan portfolio value *at entry triggers* (illustrative). If no buys, Plan == HOLD.
plan_holdings_value = hold_value
for b in plan_buys:
    # Add value of the planned new position at trigger price
    plan_holdings_value += b["Shares"] * b["Trigger Px"]

# TTM dividend proxy for Action Plan (add carry of any new planned positions)
plan_div_ttm = hold_div_ttm
for b in plan_buys:
    row = out.loc[out["Ticker"] == b["Ticker"]]
    if not row.empty:
        dps = float(row["TTM Div/Share"].iloc[0])
        plan_div_ttm += dps * b["Shares"]

# Comparison (price-only & TTM dividends proxy)
price_only_delta = (plan_holdings_value - hold_value) / hold_value * 100.0 if hold_value > 0 else 0.0
total_return_delta = ((plan_holdings_value + plan_div_ttm) - (hold_value + hold_div_ttm))
total_return_delta_pct = (total_return_delta / (hold_value + hold_div_ttm) * 100.0) if (hold_value + hold_div_ttm) > 0 else 0.0

print("\n" + "-"*88)
print("PORTFOLIO COMPARISON — HOLD vs Round‑2 Action Plan (illustrative at triggers)")
print("-"*88)
print(f"• HOLD (price-only):        SGD {hold_value:,.2f}")
print(f"• HOLD (incl. TTM divs):    SGD {(hold_value + hold_div_ttm):,.2f}  (TTM divs ≈ {hold_div_ttm:,.2f})")
print(f"• PLAN (price-only):        SGD {plan_holdings_value:,.2f}")
print(f"• PLAN (incl. TTM divs):    SGD {(plan_holdings_value + plan_div_ttm):,.2f}  (TTM divs ≈ {plan_div_ttm:,.2f})")
print(f"• Δ vs HOLD (price-only):   {price_only_delta:.2f}%")
print(f"• Δ vs HOLD (incl. TTM):    {total_return_delta_pct:.2f}%")
print("Note: Plan assumes new buys at their trigger prices; if no qualified BUYs, Plan == HOLD.\n")

# =========================
# 7) Action Plan (BUY & TRIM/SELL)
# =========================

print("ACTION PLAN")
print("-----------")
if plan_buys:
    print("BUY scenarios (execute only if triggers hit on the close + volume conditions):")
    for b in plan_buys:
        print(f"• {b['Ticker']}: BUY {b['Shares']} @ {b['Trigger Px']:.4f} (alloc ~SGD {b['Alloc SGD']:,.2f} incl. fees) | "
              f"Stop {b['Init Stop']:.4f} ({b['Stop %']:.2f}%) | "
              f"TPs: {b['TP +2×ATR']:.4f} / {b['TP +3×ATR']:.4f} | Trail: {b['Trail']}")
else:
    print("• No qualified BUYs after R:R≥2 gating and exposure checks. Keep dry powder ready.")

# Example TRIM posture (rule-based; uses current RSI to suggest readiness)
overbought = out[(out["Ticker"] == "C07.SI") & (out["RSI-14"] > 70)]
if not overbought.empty:
    print("\nTRIM/SELL scenarios:")
    print("• C07.SI: If RSI>70 then crosses down OR a bearish reversal on volume surge → Trim 1/3; "
          "next 1/3 near +3×ATR; trail last 1/3 by rising 20‑DMA.")
else:
    print("\nTRIM/SELL scenarios:")
    print("• No immediate overbought exits flagged today. Re‑check banks/REITs on any reversal with volume.")

# =========================
# 8) Confidence & Next checkpoints
# =========================

# Confidence: tie to number of confirmed signals among qualified names
num_qualified = len(qualified.index)
avg_signals = qualified["Signals (1–6)"].mean() if num_qualified > 0 else 0.0
confidence_pct = 50 + 5*avg_signals if num_qualified > 0 else 72  # heuristic; abstention often safer
confidence_label = "Medium" if num_qualified > 0 else "High"

print("\nCONFIDENCE")
print("----------")
print(f"• {confidence_label} (~{confidence_pct:.0f}%). "
      f"{'Some names passed gates.' if num_qualified>0 else 'Abstaining today preserves edge until breakouts confirm.'}")

print("\nNEXT REVIEW CHECKPOINTS")
print("-----------------------")
print("• Any name: Close > 20‑day High AND Volume ≥ 1.5× 20‑day average (adds 3rd signal).")
print("• Banks (D05/U11/O39): Close back > 20‑DMA AND MACD histogram > 0 for 2 days.")
print("• Exception path: RSI<30 then bullish reversal WITH catalyst; stop must be <5%.")

# =========================
# 9) Assumptions & Caveats
# =========================

print("\nASSUMPTIONS & CAVEATS")
print("---------------------")
if non_trading_msg:
    print("• " + non_trading_msg)
print("• Indicators computed strictly from Yahoo OHLCV last 20 trading days (no third‑party indicators).")
print("• Breakout and volume thresholds are validated on the official close only.")
print("• SGX price cross‑check is not automated here; if Yahoo close differs from SGX by >0.5%, "
      "use SGX prices board to verify and note which source you used for signal math.")
print("• TTM dividends are a proxy; future distributions are uncertain and not forecast in this script.")
print("• Exposure cap: no add if post‑trade single‑name exposure would exceed 70%.")

# =========================
# 10) Save CSV with timestamp (SGT)
# =========================

# Persist the full enriched table
csv_cols = display_cols  # keep output schema aligned with the prompt
csv_name = f"sgx_scan_{timestamp_str}.csv"
out[csv_cols].to_csv(csv_name, index=False)
print(f"\nSaved CSV: {csv_name}")

# =========================
# 11) Summary of Recommendations (dated label)
# =========================

dated_label = now_sgt_dt.strftime("%-d %b %Y") if hasattr(now_sgt_dt, "strftime") else today_str
print("\n" + "="*88)
print(f"Summary of Recommendations ({dated_label}, SGT)")
print("="*88)

final_buys = out[out["Trade (Final)"].str.contains("BUY", na=False)]
if final_buys.empty:
    print("• BUY now (Final): —")
else:
    for _, r in final_buys.iterrows():
        print(f"• BUY now: {r['Ticker']} — Price {r['Price']:.4f}, Signals {int(r['Signals (1–6)'])}, "
              f"Net R:R {r['Net R:R']:.2f}×")

if not on_watch.empty:
    print("• On Watch:")
    for _, r in on_watch.iterrows():
        print(f"  - {r['Ticker']}: Close > {r['Breakout Px']:.4f} AND Vol ≥ {r['Breakout Vol']} "
              f"(current Net R:R {r['Net R:R']:.2f}×)")
else:
    print("• On Watch: —")

no_trade_names = out[out["Trade (Final)"] == "No-Trade"]["Ticker"].tolist()
if no_trade_names:
    print("• No‑Trade: " + ", ".join(no_trade_names))
else:
    print("• No‑Trade: —")

# =========================
# End of script
# =========================
