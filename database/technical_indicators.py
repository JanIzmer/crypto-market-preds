import sys
import os
import math
import argparse
from typing import Optional

import pandas as pd
import numpy as np
from sqlalchemy import text

# Import database configuration and engine function from db_config.py
from db_config import DATABASE_URL, get_db_engine 

CSV_PATH = "notebooks/data/BTCUSDT-1h.csv" # Default CSV path

# -------------------------
# Helper indicator functions
# -------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df to be sorted by candle_time in ascending order for a single ticker.
    Adds columns required for calculating flags and atr14.
    """
    # Ensure all values are numeric
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # EMA12, EMA26
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()

    # MACD line and signal (signal = ema9 of macd)
    df["macd_line"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # SMA50
    df["sma50"] = df["close"].rolling(window=50, min_periods=1).mean()

    # RSI14 (Wilder smoothing)
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # Wilder's smoothing: SMMA ≈ EMA with alpha = 1/14
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down
    df["rsi14"] = 100 - (100 / (1 + rs))

    # True Range (TR)
    prev_close = df["close"].shift(1)
    df["tr"] = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    # ATR14 (Wilder's smoothing)
    df["atr14"] = df["tr"].ewm(alpha=1/14, adjust=False).mean()

    # ROC9 (rate of change over 9 periods)
    df["roc9"] = df["close"].pct_change(periods=9)

    # Volume pct change 1
    df["volume_pct_change_1"] = df["volume"].pct_change(periods=1)

    # Bollinger Bands (20, 2)
    df["sma20"] = df["close"].rolling(window=20, min_periods=1).mean()
    df["std20"] = df["close"].rolling(window=20, min_periods=1).std(ddof=0)
    df["bb_upper"] = df["sma20"] + 2 * df["std20"]
    df["bb_lower"] = df["sma20"] - 2 * df["std20"]

    # VWAP per candle (typical price weighted by volume) - here computed per candle:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    # VWAP is usually cumulative over a session; for single-candle representation we use typical_price
    df["vwap"] = (typical_price * df["volume"]) / df["volume"].replace({0: np.nan})

    # Fill small NaN where appropriate
    df.fillna(value=np.nan, inplace=True)

    return df


def detect_cross(series_a: pd.Series, series_b: pd.Series):
    """
    Returns two boolean series: (cross_up, cross_down)
    """
    prev_a = series_a.shift(1)
    prev_b = series_b.shift(1)

    cross_up = (series_a > series_b) & (prev_a <= prev_b)
    cross_down = (series_a < series_b) & (prev_a >= prev_b)
    return cross_up.fillna(False), cross_down.fillna(False)


def build_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses computed indicator columns to create flag columns for database insertion.
    """

    # EMA crosses
    ema_up, ema_down = detect_cross(df["ema12"], df["ema26"])
    df["ema12_cross_ema26_up"] = ema_up.astype(int)
    df["ema12_cross_ema26_down"] = ema_down.astype(int)

    # SMA50 crosses on close
    sma50_up, sma50_down = detect_cross(df["close"], df["sma50"])
    df["close_cross_sma50_up"] = sma50_up.astype(int)
    df["close_cross_sma50_down"] = sma50_down.astype(int)

    # MACD cross with signal
    macd_up, macd_down = detect_cross(df["macd_line"], df["macd_signal"])
    df["macd_cross_signal_up"] = macd_up.astype(int)
    df["macd_cross_signal_down"] = macd_down.astype(int)

    # Bollinger band crosses on close
    bb_up, bb_down = detect_cross(df["close"], df["bb_upper"])
    df["close_cross_upper_bb"] = bb_up.astype(int)
    bb_lower_up, bb_lower_down = detect_cross(df["close"], df["bb_lower"])
    df["close_cross_lower_bb"] = bb_lower_down.astype(int)

    # RSI thresholds
    df["rsi_overbought"] = (df["rsi14"] > 70).astype(int)
    df["rsi_oversold"] = (df["rsi14"] < 30).astype(int)

    # Strong trend heuristic
    df["strong_trend"] = (
        ((df["macd_hist"] > 0) & (df["rsi14"] > 50) & (df["close"] > df["sma50"])) |
        ((df["macd_hist"] < 0) & (df["rsi14"] < 50) & (df["close"] < df["sma50"]))
    ).astype(int)

    # Include necessary columns for the output DF
    df["atr14"] = df["atr14"]

    # Select the exact set of columns we will upsert (11 flags + PKs + ATR14)
    out_cols = [
        "ticker", "candle_time", "atr14",
        "close_cross_lower_bb", "close_cross_sma50_down", "close_cross_sma50_up",
        "close_cross_upper_bb", "ema12_cross_ema26_down", "ema12_cross_ema26_up",
        "macd_cross_signal_down", "macd_cross_signal_up",
        "rsi_overbought", "rsi_oversold", "strong_trend",
    ]

    # Ensure all columns exist (fill with 0/NaN where missing)
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[out_cols].copy()


# -------------------------
# DB upsert functions
# -------------------------

def upsert_kline_data(engine, df_kline: pd.DataFrame):
    """
    Upserts OHLCV data into the PARENT table: kline_data.
    This must be executed BEFORE upsert_technical_indicators.
    """
    df = df_kline.copy()
    
    # CRITICAL FIX: Convert Pandas datetime64[ns] to standard Python datetime 
    # for reliable insertion into MySQL using pymysql driver.
    df['candle_time'] = df['candle_time'].dt.to_pydatetime()
    
    # Columns for kline_data table
    insert_cols = ["ticker", "candle_time", "open", "high", "low", "close", "volume"]

    # Build parameterized insert statement
    cols_sql = ", ".join(f"`{c}`" for c in insert_cols)
    vals_sql = ", ".join(f":{c}" for c in insert_cols)
    update_sql = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in insert_cols[2:]) # update OHLCV only

    sql = text(f"""
        INSERT INTO kline_data ({cols_sql})
        VALUES ({vals_sql})
        ON DUPLICATE KEY UPDATE {update_sql};
    """)

    conn = engine.connect()
    trans = conn.begin()
    try:
        # records will contain Python datetime objects which PyMySQL handles correctly
        records = df[insert_cols].to_dict(orient="records")
        conn.execute(sql, records)
        trans.commit()
        print(f"Upserted {len(records)} rows into kline_data (Parent Table).")
    except Exception as e:
        trans.rollback()
        raise
    finally:
        conn.close()


def upsert_technical_indicators(engine, df_flags: pd.DataFrame):
    """
    Upserts indicator data into the CHILD table: technical_indicators.
    """
    df = df_flags.copy()
    # Conversion is necessary here because the data came as strings from main() concatenation
    df["candle_time"] = pd.to_datetime(df["candle_time"]) 

    # CRITICAL FIX: Convert Pandas datetime64[ns] to standard Python datetime 
    # for reliable insertion into MySQL using pymysql driver.
    df['candle_time'] = df['candle_time'].dt.to_pydatetime()

    # List of columns to insert (11 flags + PKs + ATR14)
    insert_cols = [
        "ticker", "candle_time", "atr14",
        "close_cross_lower_bb", "close_cross_sma50_down", "close_cross_sma50_up",
        "close_cross_upper_bb", "ema12_cross_ema26_down", "ema12_cross_ema26_up",
        "macd_cross_signal_down", "macd_cross_signal_up",
        "rsi_overbought", "rsi_oversold", "strong_trend"
    ]

    # Build parameterized insert statement
    cols_sql = ", ".join(f"`{c}`" for c in insert_cols)
    vals_sql = ", ".join(f":{c}" for c in insert_cols)
    update_sql = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in insert_cols[2:]) # update all but PKs

    sql = text(f"""
        INSERT INTO technical_indicators ({cols_sql})
        VALUES ({vals_sql})
        ON DUPLICATE KEY UPDATE {update_sql};
    """)

    conn = engine.connect()
    trans = conn.begin()
    try:
        # records will contain Python datetime objects which PyMySQL handles correctly
        records = df[insert_cols].to_dict(orient="records")
        conn.execute(sql, records)
        trans.commit()
        print(f"Upserted {len(records)} rows into technical_indicators (Child Table).")
    except Exception as e:
        trans.rollback()
        raise
    finally:
        conn.close()


# -------------------------
# Main
# -------------------------
def main(csv_path: str, db_url: Optional[str] = None):
    # Use the imported DATABASE_URL as default if not explicitly provided
    current_db_url = db_url or DATABASE_URL

    current_csv_path = csv_path

    if not os.path.exists(current_csv_path):
        print(f"CSV file not found: {current_csv_path}", file=sys.stderr)
        return

    print(f"Reading CSV from {current_csv_path}...")
    df = pd.read_csv(current_csv_path)

    # --- Data Preprocessing: Handling common column differences ---

    # 1. Rename 'timestamp' to 'candle_time' if present.
    if 'timestamp' in df.columns and 'candle_time' not in df.columns:
        print("Renaming 'timestamp' column to 'candle_time'...")
        df.rename(columns={'timestamp': 'candle_time'}, inplace=True)

    # 2. Add 'ticker' column if missing.
    if 'ticker' not in df.columns:
        # Infer default ticker from the filename or use a reliable default.
        filename = os.path.basename(current_csv_path)
        default_ticker = filename.split('-')[0].upper()
        if not default_ticker or len(default_ticker) > 10:
             default_ticker = "BTCUSDT"
        
        print(f"Adding missing 'ticker' column with value: {default_ticker}")
        df['ticker'] = default_ticker

    # --- End Preprocessing ---

    # Core OHLCV columns needed for kline_data
    ohlcv_cols = ["ticker", "candle_time", "open", "high", "low", "close", "volume"]
    
    if not set(ohlcv_cols).issubset(set(df.columns)):
        missing = set(ohlcv_cols) - set(df.columns)
        raise ValueError(f"CSV still missing essential OHLCV columns after renaming/adding ticker. Missing: {missing}. Found: {set(df.columns)}")

    # Final cleanup before processing
    df.rename(columns=lambda c: c.strip(), inplace=True)
    
    # *** FIX FOR MILLISECOND TIMESTAMPS ***
    print("Converting 'candle_time' (timestamp) using unit='ms'...")
    # This step is now the single, correct source for datetime objects (datetime64[ns])
    df["candle_time"] = pd.to_datetime(df["candle_time"], unit='ms')
    
    df["ticker"] = df["ticker"].astype(str).str.upper()


    # Get the engine using the imported function
    engine = get_db_engine(current_db_url)
    
    # STEP 1: UPSERT KLINE DATA (PARENT TABLE)
    print("--- STEP 1: Upserting OHLCV data into kline_data (Parent Table) ---")
    df_kline_data = df[ohlcv_cols].copy()
    # df_kline_data now contains correct datetime64[ns] objects
    upsert_kline_data(engine, df_kline_data)
    
    
    # STEP 2: COMPUTE INDICATORS AND UPSERT FLAGS (CHILD TABLE)
    print("\n--- STEP 2: Computing Indicators and Upserting into technical_indicators (Child Table) ---")
    out_rows = []
    
    for ticker, group in df.groupby("ticker"):
        print(f"Processing ticker={ticker} rows={len(group)}")
        g = group.sort_values("candle_time").reset_index(drop=True).copy()
        
        # 2a. compute indicators
        g_inds = compute_indicators(g)
        
        # 2b. build flags
        g_flags = build_flags(g_inds)
        
        # Final column preparation
        g_flags["ticker"] = ticker
        # Converting back to string for concatenation 
        g_flags["candle_time"] = g["candle_time"].astype(str) 
        out_rows.append(g_flags)

    if not out_rows:
        print("No data to upsert for technical indicators.")
        return

    df_out = pd.concat(out_rows, ignore_index=True)
    
    # 2c. Upsert flags
    upsert_technical_indicators(engine, df_out)
    
    print("\nProcess finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest CSV and compute/insert technical indicators.")
    parser.add_argument("--csv", help="Path to CSV file with OHLCV (ticker,candle_time,open,high,low,close,volume)", default=CSV_PATH)
    parser.add_argument("--db-url", help="Optional SQLAlchemy DB URL (overrides environment settings)", default=None)
    args = parser.parse_args()
    main(args.csv, db_url=args.db_url)
