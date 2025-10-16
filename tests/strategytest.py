import sys
import os
import pandas as pd
import joblib
import numpy as np

# Add parent directory to path for core module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.market_type import MarketTrendDetector
from core.strategies.bull import generate_signal as bull_signal
from core.strategies.bear import generate_signal as bear_signal
from core.strategies.range import generate_signal as range_signal
from core.data_clean_functions.calc_indicators import calculate_indicators as calc_indicators

# Constants for ATR-based stop-loss and take-profit
ATR_MULT_SL = 1.75
ATR_MULT_TP = 3.5
HISTORY_LEN = 300  # Number of historical candles to feed into the models

# Load pre-trained machine learning models
model_st1 = joblib.load('stage1_model.pkl')
model_st2 = joblib.load('stage2_model.pkl')


def run_backtest(df_60min, starting_balance=1000, fee=0.0001):
    """
    Runs a backtest using technical indicators and ML-based trade signal generators.

    Parameters:
        df_60min (pd.DataFrame): DataFrame containing 60-minute OHLCV candles with timestamp.
        starting_balance (float): Initial capital for the backtest.
        fee (float): Trading fee per side (e.g. 0.001 for 0.1%).

    Returns:
        tuple:
            - final_balance (float): Final account balance after backtest.
            - balance_curve (List[float]): Balance value at each time step.
            - trades (List[Dict]): List of executed trades with their details.
    """
    balance = starting_balance
    position = None
    entry_price = 0
    entry_time = None
    market_type = None
    balance_curve = []
    trades = []

    trend_detector = MarketTrendDetector()
    df = calc_indicators(df_60min.copy())

    # Iterate over each time step after enough history is collected
    for i in range(HISTORY_LEN, len(df)):
        sliced = df.iloc[i - HISTORY_LEN + 1 : i + 1].copy()
        current_candle = df.iloc[i]
        timestamp = current_candle['timestamp']
        close_price = current_candle['close']
        high_price = current_candle['high']
        low_price = current_candle['low']

        # Detect current market regime: bull, bear or range
        trend = trend_detector.detect_market_trend(sliced)

        # Generate signal using the strategy model appropriate for current market regime
        if trend == 'bull':
            signal = bull_signal(sliced, model_st1, model_st2, position)
        elif trend == 'bear':
            signal = bear_signal(sliced, model_st1, model_st2, position)
        else:
            signal = range_signal(sliced, sliced)

        # Calculate average ATR over last 14 periods and cap it at 1.5% of close price
        atr = sliced['atr'].iloc[-14:].mean()
        atr = min(atr, close_price * 0.015)

        if position is None:
            # Entry conditions
            if signal == "BUY":
                position = 'long'
                entry_price = close_price * (1 + fee)
                entry_time = timestamp
                market_type = trend
                stop_loss_price = entry_price - ATR_MULT_SL * atr
                take_profit_price = entry_price + ATR_MULT_TP * atr

            elif signal == "SELL":
                position = 'short'
                entry_price = close_price * (1 - fee)
                entry_time = timestamp
                market_type = trend
                stop_loss_price = entry_price + ATR_MULT_SL * atr
                take_profit_price = entry_price - ATR_MULT_TP * atr

        else:
            # Manage open position
            exit_trade = False
            exit_price = None
            exit_reason = None

            if position == 'long':
                if low_price <= stop_loss_price:
                    exit_price = stop_loss_price * (1 - fee)
                    exit_reason = 'stop_loss'
                    exit_trade = True
                elif high_price >= take_profit_price:
                    exit_price = take_profit_price * (1 - fee)
                    exit_reason = 'take_profit'
                    exit_trade = True
                #COMMENTED FOR TEST
                elif signal == "SELL":
                    exit_price = close_price * (1 - fee)
                    exit_reason = 'signal_sell'
                    exit_trade = True

            elif position == 'short':
                if high_price >= stop_loss_price:
                    exit_price = stop_loss_price * (1 + fee)
                    exit_reason = 'stop_loss'
                    exit_trade = True
                elif low_price <= take_profit_price:
                    exit_price = take_profit_price * (1 + fee)
                    exit_reason = 'take_profit'
                    exit_trade = True
                #COMMENTED FOR TEST    
                elif signal == "BUY":
                    exit_price = close_price * (1 + fee)
                    exit_reason = 'signal_buy'
                    exit_trade = True

            if exit_trade:
                # Calculate profit and update balance
                if position == 'long':
                    profit = (exit_price - entry_price) / entry_price
                else:
                    profit = (entry_price - exit_price) / entry_price

                balance *= (1 + profit)

                trades.append({
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit * 100,
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'trend': market_type,
                    'position': position,
                    'exit_reason': exit_reason
                })

                position = None

        # Save balance at each time step
        balance_curve.append(balance)

    return balance, balance_curve, trades
