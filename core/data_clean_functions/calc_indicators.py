import numpy as np
import pandas as pd

def calculate_indicators(df):
    """
    Enhances the input DataFrame with technical analysis indicators.

    This function calculates and appends various commonly used technical indicators 
    (e.g., moving averages, RSI, MACD, ATR, etc.) to the provided OHLCV DataFrame. 
    It is typically used as a preprocessing step before signal generation or model prediction.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing historical price data with columns like 
                        'open', 'high', 'low', 'close', and 'volume'.

    Returns:
        pd.DataFrame: The same DataFrame with additional columns representing calculated indicators.
    """
    df = df.copy()

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['hl_range'] = df['high'] - df['low']
    df['oc_range'] = df['close'] - df['open']

    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df.index).rolling(14).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()

    rolling_std_20 = df['close'].rolling(window=20).std()
    df['upper_bb'] = df['sma20'] + 2 * rolling_std_20
    df['lower_bb'] = df['sma20'] - 2 * rolling_std_20

    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    tp_mean = df['typical_price'].rolling(20).mean()
    tp_std = df['typical_price'].rolling(20).std()
    df['cci'] = (df['typical_price'] - tp_mean) / (0.015 * tp_std)

    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()

    df['log_volume'] = np.log1p(df['volume'])
    df['log_num_trades'] = np.log1p(df['num_trades'])
    df['log_taker_vol'] = np.log1p(df['taker_base_vol'])

    return df
