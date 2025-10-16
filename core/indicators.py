from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange

def add_indicators(df):
    """
    Adds common technical indicators to a DataFrame with OHLCV data.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing at least 'high', 'low', and 'close' columns.

    Returns:
    - pd.DataFrame: Original DataFrame with additional indicator columns:
        - 'rsi': Relative Strength Index (momentum)
        - 'ema12': 12-period Exponential Moving Average
        - 'ema26': 26-period Exponential Moving Average
        - 'macd': MACD line (trend-following momentum)
        - 'macd_signal': MACD signal line
        - 'stoch_k': Stochastic %K (momentum)
        - 'stoch_d': Stochastic %D signal
        - 'atr': Average True Range (volatility)
    """

    df['rsi'] = RSIIndicator(df['close']).rsi()
    df['ema12'] = EMAIndicator(df['close'], window=12).ema_indicator()
    df['ema26'] = EMAIndicator(df['close'], window=26).ema_indicator()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    stoch = StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

    return df
