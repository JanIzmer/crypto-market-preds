import numpy as np
import pandas as pd
import ta
from sklearn.linear_model import LinearRegression

class MarketTrendDetector:
    """
    Detects the current market trend (bull, bear, or sideways) using technical indicators.
    Trend is only confirmed after a given number of consecutive confirmations (min_confirm).
    """

    def __init__(self, min_confirm=8):
        """
        Initializes the trend detector.

        Args:
            min_confirm (int): How many times the same trend must appear in a row to be confirmed.
        """
        self.min_confirm = min_confirm
        self.confirmed_trend = None
        self.potential_trend = None
        self.counter = 0

    def _calc_ema_slope(self, series: pd.Series, window: int = 7) -> float:
        """
        Calculates the slope (trend direction) of the EMA over the last N values using linear regression.

        Args:
            series (pd.Series): A pandas Series (e.g., EMA values).
            window (int): Number of most recent points to consider.

        Returns:
            float: Slope (positive for upward trend, negative for downward).
        """
        y = series.iloc[-window:].values.reshape(-1, 1)
        x = np.arange(window).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        return model.coef_[0][0]

    def _calculate_raw_trend(self, df: pd.DataFrame, current_trend) -> str:
        """
        Uses multiple indicators (EMA, RSI, ADX, ATR, return) to guess the current trend.

        Args:
            df (pd.DataFrame): Market data with 'close', 'high', and 'low' columns.
            current_trend (str): The last known trend, if any.

        Returns:
            str: One of 'bull', 'bear', 'sideways', or fallback to current trend.
        """
        if len(df) < 50:
            return 'sideways'

        df = df.copy()
        df['EMA20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['EMA100'] = ta.trend.ema_indicator(df['close'], window=100)
        df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

        last = df.iloc[-1]

        ema_slope = self._calc_ema_slope(df['EMA20'], window=7)
        atr_mean = df['ATR'].rolling(window=50).mean().iloc[-1]
        atr_ratio = last['ATR'] / atr_mean if atr_mean else 1
        recent_return = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]

        # Strong bullish case
        if last['EMA20'] > last['EMA100'] and ema_slope > 0:
            return 'bull'

        # Strong bearish case
        if last['EMA20'] < last['EMA100'] and ema_slope < 0:
            return 'bear'

        # Alternative bullish criteria
        if recent_return > 0.01 and ema_slope > 0 and df['RSI'].iloc[-1] > 55:
            return 'bull'
        if recent_return < -0.01 and ema_slope < 0 and df['RSI'].iloc[-1] < 45:
            return 'bear'

        # Volatility-driven direction
        if recent_return > 0.01 and atr_ratio > 0.7:
            return 'bull'
        if recent_return < -0.01 and atr_ratio > 0.7:
            return 'bear'

        return current_trend if current_trend else None

    def detect_market_trend(self, df: pd.DataFrame) -> str:
        """
        Detects the current trend and confirms it only after multiple consecutive matches.

        Args:
            df (pd.DataFrame): Market data.

        Returns:
            str: The confirmed or currently observed trend ('bull', 'bear', or 'sideways').
        """
        current_trend = self._calculate_raw_trend(df, None)

        if self.potential_trend is None:
            self.potential_trend = current_trend
            self.counter = 1
        elif current_trend == self.potential_trend:
            self.counter += 1
            if self.counter >= self.min_confirm:
                self.confirmed_trend = current_trend
        else:
            self.potential_trend = current_trend
            self.counter = 1

        return self.confirmed_trend if self.confirmed_trend else current_trend




