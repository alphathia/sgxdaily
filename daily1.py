# SGX Round‑2 Scanner — Yahoo Finance OHLCV (20D) & Signals
# Author: Your SGX equity strategist
# Date: auto (today, local SGT)
# Data source: Yahoo Finance "Historical Data" via yfinance (exact OHLCV)
# Universe: D05.SI, U11.SI, O39.SI, B61.SI, ME8U.SI, BUOU.SI, S59.SI, S63.SI,
#           S58.SI, U13.SI, J69U.SI, S68.SI, C07.SI, HMN.SI, 9CI.SI, N2IU.SI, C38U.SI

import pandas as pd, numpy as np, yfinance as yf
from datetime import datetime
import pytz

# Set up Singapore timezone for accurate local timestamps
SGT = pytz.timezone("Asia/Singapore")
today_sgt = datetime.now(SGT).strftime("%Y-%m-%d")

# SGX stock universe - 17 major Singapore Exchange listed stocks
# .SI suffix indicates Singapore Exchange listing
TICKERS = [
    "D05.SI","U11.SI","O39.SI","B61.SI","ME8U.SI","BUOU.SI","S59.SI","S63.SI",
    "S58.SI","U13.SI","J69U.SI","S68.SI","C07.SI","HMN.SI","9CI.SI","N2IU.SI","C38U.SI"
]

# --------- Technical Indicator Helper Functions ---------

def ema(s, span):
    """
    Calculate Exponential Moving Average
    Args:
        s: Price series (pandas Series)
        span: Number of periods for EMA calculation
    Returns:
        EMA series
    """
    return s.ewm(span=span, adjust=False).mean()

def rsi(close, period=14):
    """
    Calculate Relative Strength Index (RSI)
    RSI measures momentum - values above 70 suggest overbought, below 30 suggest oversold
    Args:
        close: Closing price series
        period: RSI calculation period (default 14 days)
    Returns:
        RSI series (0-100 scale)
    """
    delta = close.diff()  # Price changes day-to-day
    gain = delta.clip(lower=0.0)  # Only positive changes (gains)
    loss = -delta.clip(upper=0.0)  # Only negative changes (losses, made positive)
    
    # Calculate average gains and losses using exponential smoothing
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # RSI formula: 100 - (100 / (1 + RS)), where RS = avg_gain / avg_loss
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(close):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    MACD helps identify trend changes and momentum
    Args:
        close: Closing price series
    Returns:
        macd_line: MACD line (12-day EMA - 26-day EMA)
        signal: Signal line (9-day EMA of MACD line)
        hist: Histogram (MACD line - Signal line)
    """
    macd_line = ema(close, 12) - ema(close, 26)  # Fast EMA - Slow EMA
    signal = ema(macd_line, 9)  # Signal line smooths MACD
    hist = macd_line - signal  # Histogram shows MACD vs Signal relationship
    return macd_line, signal, hist

def atr(df, period=14):
    """
    Calculate Average True Range (ATR)
    ATR measures volatility - higher values indicate more volatile stocks
    Args:
        df: DataFrame with High, Low, Close columns
        period: ATR calculation period (default 14 days)
    Returns:
        ATR series
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)  # Previous day's closing price
    
    # True Range is the maximum of:
    # 1. Current High - Current Low
    # 2. Current High - Previous Close (absolute value)
    # 3. Current Low - Previous Close (absolute value)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    # ATR is the exponential moving average of True Range
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def bullish_divergence(close, rsi_series, lookback=40):
    """
    Detect bullish divergence pattern
    Bullish divergence occurs when price makes lower lows but RSI makes higher lows,
    suggesting potential upward reversal despite falling prices
    Args:
        close: Closing price series
        rsi_series: RSI series
        lookback: Number of bars to look back for pivot points (default 40)
    Returns:
        Boolean indicating if bullish divergence pattern is detected
    """
    # Get recent price and RSI data
    c = close.tail(lookback).copy()
    r = rsi_series.loc[c.index]
    
    # Find pivot lows (local minima) - points lower than both neighbors
    pivots = []
    vals = c.values
    idxs = list(c.index)
    for i in range(1, len(vals)-1):
        if vals[i] < vals[i-1] and vals[i] < vals[i+1]:  # Lower than both neighbors
            pivots.append(i)
    
    # Need at least 2 pivot lows to compare
    if len(pivots) < 2:
        return False
    
    # Compare the two most recent pivot lows
    i1, i2 = pivots[-2], pivots[-1]  # Second-last and last pivot indices
    p1, p2 = vals[i1], vals[i2]      # Price values at these pivots
    r1, r2 = r.iloc[i1], r.iloc[i2]  # RSI values at these pivots
    
    # Bullish divergence criteria:
    # 1. Price makes Lower Low (p2 < p1)
    # 2. RSI makes Higher Low (r2 > r1) 
    # 3. Latest close is higher than previous close (confirms momentum shift)
    latest_up = close.iloc[-1] > close.iloc[-2]
    return (p2 < p1) and (r2 > r1) and latest_up

def support_bounce(df, sma50_name="SMA50"):
    """
    Detect support bounce pattern at 50-day moving average
    This identifies potential reversal when price bounces off the 50-day SMA support
    Args:
        df: DataFrame with OHLC data and SMA50 column
        sma50_name: Column name for 50-day moving average (default "SMA50")
    Returns:
        Boolean indicating if support bounce pattern is detected
    """
    # Check if latest candle is bullish (close > open)
    bullish = df["Close"].iloc[-1] > df["Open"].iloc[-1]
    
    # Check if price touched/dipped below SMA50 but closed above it
    # This indicates a bounce off support level
    touch = (df["Low"].iloc[-1] <= df[sma50_name].iloc[-1]) and (df["Close"].iloc[-1] >= df[sma50_name].iloc[-1])
    
    return bool(bullish and touch)

# --------- Main Analysis Loop ---------

# Storage for results and error messages
rows = []
errors = []

# Process each ticker in the SGX stock universe
for t in TICKERS:
    try:
        # Download historical data from Yahoo Finance
        # Pull 400 trading days to ensure all long-term moving averages are stable
        # auto_adjust=False preserves original OHLCV data without adjustments
        df = yf.download(t, period="400d", interval="1d", auto_adjust=False, progress=False)
        
        # Skip ticker if no data is available
        if df.empty:
            errors.append(f"{t}: empty dataframe from Yahoo")
            continue

        # Flatten MultiIndex columns if they exist (happens when downloading single ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Clean data by removing any rows with missing values
        df = df.dropna().copy()

        # --------- Calculate Technical Indicators ---------
        
        # Moving Averages - trend identification
        df["SMA20"] = df["Close"].rolling(20).mean()    # Short-term trend (20 days)
        df["SMA50"] = df["Close"].rolling(50).mean()    # Medium-term trend (50 days)
        df["SMA200"] = df["Close"].rolling(200).mean()  # Long-term trend (200 days)

        # Momentum and volatility indicators
        df["RSI14"] = rsi(df["Close"], 14)  # 14-day RSI for momentum
        macd_line, macd_signal, macd_hist = macd(df["Close"])  # MACD components
        df["MACD"], df["MACDsig"], df["MACDhist"] = macd_line, macd_signal, macd_hist
        df["ATR14"] = atr(df, 14)  # 14-day ATR for volatility

        # Volume and breakout reference levels
        # Shift by 1 day to avoid look-ahead bias in breakout detection
        df["High20_Close"] = df["Close"].rolling(20).max().shift(1)  # 20-day high of closes
        df["Vol20_Avg"] = df["Volume"].rolling(20).mean().shift(1)   # 20-day average volume

        # --------- Extract Current Values ---------
        
        # Get most recent complete trading day data
        last = df.iloc[-1]  # Latest day
        prev = df.iloc[-2] if len(df) >= 2 else last  # Previous day for comparison

        # Extract key metrics for analysis - handle both Series and scalar values
        def extract_value(series_or_scalar):
            if hasattr(series_or_scalar, 'iloc'):
                return float(series_or_scalar.iloc[0])
            elif hasattr(series_or_scalar, 'item'):
                return float(series_or_scalar.item())
            else:
                return float(series_or_scalar)
        
        price = extract_value(last["Close"])
        rsi14 = extract_value(last["RSI14"])
        atr14 = extract_value(last["ATR14"])
        sma20 = extract_value(last["SMA20"])
        sma50 = extract_value(last["SMA50"])
        sma200 = extract_value(last["SMA200"])
        hi20 = extract_value(last["High20_Close"])
        vol20 = extract_value(last["Vol20_Avg"])
        curr_volume = extract_value(last["Volume"])
        macd_val = extract_value(last["MACD"])
        macd_sig = extract_value(last["MACDsig"])
        macd_hist_val = extract_value(last["MACDhist"])

        # --------- Trading Signal Rules (6 Main Signals) ---------
        
        # Signal 1: RSI Recovery Rule
        # Look for RSI > 50 after it dipped below 35 in the past 20 days
        # This indicates momentum recovery after oversold conditions
        if len(df) >= 21:  # Ensure we have enough data points
            dipped = (df["RSI14"].iloc[-21:-1] < 35).any()  # Check past 20 days for RSI < 35
        else:
            dipped = (df["RSI14"].iloc[:-1] < 35).any()  # Use available data
        rsi_rule = (rsi14 > 50.0) and bool(dipped)  # Current RSI > 50 AND recent dip occurred

        # Signal 2: Moving Average Trend Rule
        # Bullish trend: Price above short-term MA, and short-term MA above medium-term MA
        # This ensures we're trading with the trend
        ma_trend_rule = (price > sma20) and (sma20 > sma50)

        # Signal 3: Volume Breakout Rule
        # Price breaks above 20-day high with increased volume (≥1.5x average)
        # High volume confirms breakout strength and reduces false signals
        breakout_rule = (price > hi20) and (curr_volume >= 1.5 * vol20)

        # Signal 4: Support Bounce Rule
        # Bullish reversal pattern at the 50-day moving average support level
        # Uses the support_bounce function defined earlier
        support_rule = support_bounce(df)

        # Signal 5: MACD Bullish Momentum Rule
        # MACD above signal line with positive histogram for at least 2 days
        # This confirms sustained bullish momentum
        if len(df) >= 2:  # Ensure we have at least 2 data points
            macd_rule = (
                (df["MACDhist"].iloc[-1] > 0) and (df["MACDhist"].iloc[-2] > 0)  # Positive histogram ≥2 days
                and (df["MACD"].iloc[-1] > df["MACDsig"].iloc[-1])  # MACD above signal line today
                and (df["MACD"].iloc[-2] > df["MACDsig"].iloc[-2])  # MACD above signal line yesterday
            )
        else:
            macd_rule = False  # Not enough data for this rule

        # Signal 6: Bullish Divergence Rule
        # Price makes lower lows while RSI makes higher lows (potential reversal)
        # Uses the bullish_divergence function defined earlier
        div_rule = bullish_divergence(df["Close"], df["RSI14"], lookback=40)

        # Count total signals triggered (0-6 possible)
        signals_met = sum([rsi_rule, ma_trend_rule, breakout_rule, support_rule, macd_rule, div_rule])

        # --------- Exception Cases (Manual Review Required) ---------
        
        # Exception A: Extreme Oversold Bounce
        # RSI was < 30 yesterday, today shows bullish candle with higher close
        # These require manual review as they can be risky counter-trend plays
        if len(df) >= 2:
            prev_rsi = extract_value(df["RSI14"].iloc[-2])
            prev_close = extract_value(prev["Close"])
            curr_open = extract_value(last["Open"])
            rsi_exception = (prev_rsi < 30) and (price > curr_open) and (price > prev_close)
        else:
            rsi_exception = False
        
        # Exception B: High Volume Catalyst Breakout
        # Price breakout with volume ≥2x average suggests major news/catalyst
        # Requires manual review to understand the catalyst
        vol_exception = (price > hi20) and (curr_volume >= 2.0 * vol20)

        # Flag the type of exception for manual review
        exception_flag = "RSI<30 bounce" if rsi_exception else ("2×Vol breakout" if vol_exception else "")

        # --------- Final Trade Decision ---------
        # Conservative approach: Require at least 2 signals for BUY recommendation
        trade_decision = "BUY" if signals_met >= 2 else "No-Trade"

        # --------- Fetch Upcoming Catalysts (Earnings & Dividends) ---------
        
        # Try to get earnings calendar and dividend/split history from Yahoo Finance
        # This is best-effort as Yahoo Finance API can be unreliable for this data
        cal = {}
        try:
            cal = yf.Ticker(t).calendar  # Earnings calendar
        except Exception:
            pass  # Ignore if calendar data unavailable
            
        try:
            actions = yf.Ticker(t).actions  # Dividend and stock split history
        except Exception:
            actions = pd.DataFrame()  # Empty DataFrame if data unavailable

        # Extract last dividend ex-date
        last_div_ex = None
        if isinstance(actions, pd.DataFrame) and "Dividends" in actions.columns and len(actions.index) > 0:
            last_div_ex = str(actions.index.max().date())  # Most recent dividend date

        # Extract next earnings date
        next_earnings = None
        if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            # Yahoo sometimes returns start/end date range; take the first date
            ed = cal.loc["Earnings Date"].values
            if len(ed) > 0:
                next_earnings = str(pd.to_datetime(ed[0]).date())

        # --------- Compile Results for This Ticker ---------
        
        # Create comprehensive result row with all calculated metrics and signals
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
            "MACD Signal": round(macd_sig, 4),
            "MACD Hist": round(macd_hist_val, 4),
            # Individual signal results
            "RSI rule": bool(rsi_rule),
            "MA trend rule": bool(ma_trend_rule),
            "Breakout rule": bool(breakout_rule),
            "Support bounce rule": bool(support_rule),
            "MACD rule (≥2D)": bool(macd_rule),
            "Divergence rule": bool(div_rule),
            # Summary metrics
            "Signals Met (1–6)": int(signals_met),
            "Trade Decision": trade_decision,
            "Exception flag": exception_flag,
            # Catalyst information
            "Last ex-div (Yahoo)": last_div_ex,
            "Next earnings (Yahoo)": next_earnings,
        })
        
    except Exception as e:
        # Log any errors encountered while processing this ticker
        errors.append(f"{t}: {e}")

# --------- Generate Final Report ---------

# Create DataFrame from all ticker results
if not rows:
    print("No data was successfully processed for any tickers.")
    print("Errors encountered:")
    for error in errors:
        print(f" - {error}")
    exit(1)

out = pd.DataFrame(rows)

# Check if we have the expected columns before sorting
if "Trade Decision" in out.columns and "Signals Met (1–6)" in out.columns:
    # Sort by: 1) BUY signals first, 2) Most signals met, 3) Alphabetical by ticker
    out = out.sort_values(["Trade Decision","Signals Met (1–6)","Ticker"], ascending=[True, False, True])
else:
    print("Warning: Expected columns not found. Available columns:")
    print(out.columns.tolist())
    # Just sort by ticker if standard columns not available
    out = out.sort_values("Ticker")

# Display results to console
print(f"\nSGX Round‑2 scan (Yahoo OHLCV) — {today_sgt} SGT\n")
print(out.to_string(index=False))  # Print without DataFrame index numbers

# Save results to CSV file with timestamp
csv_name = f"sgx_scan_{today_sgt}.csv"
out.to_csv(csv_name, index=False)
print(f"\nSaved: {csv_name}")

# Display any errors or warnings encountered during processing
if errors:
    print("\nWarnings:")
    for m in errors:
        print(" -", m)
