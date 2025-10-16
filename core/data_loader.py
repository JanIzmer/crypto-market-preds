import pandas as pd
from config import exchange
import time

def fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data for a given symbol and timeframe.

    Parameters:
    - symbol (str): Trading pair symbol (e.g. "BTC/USDT")
    - timeframe (str): Timeframe string compatible with exchange (e.g. "15m", "1h")
    - limit (int): Number of candles to fetch

    Returns:
    - pd.DataFrame: DataFrame containing columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """

    # Fetch OHLCV raw data from exchange
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    # Print local and server times for sync/debugging
    print('Local time:', int(time.time() * 1000))
    print('Server time:', exchange.fetch_time())
    print('Time difference:', exchange.timeout)

    # Convert raw data into a labeled DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Convert timestamps from milliseconds to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    return df
