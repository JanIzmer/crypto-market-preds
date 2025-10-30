# basic imports
import pandas as pd
import numpy as np
import sys
import os
import joblib

# data loading
from core.data_loader import detect_earliest_time_with_news, load_data_from_db

# additional imports
from datetime import datetime
import plotly.graph_objects as go

# TEST CONFIG
SL_FACTOR = 1.5   # percent
TP_FACTOR = 3.0   # percent
INITIAL_CAPITAL = 10000.0
TRADE_SIZE_PERCENT = 1  # 1 == full capital 
OUTPUT_PARQUET = os.getenv('OUTPUT_PARQUET', 'combined_simple.parquet')
MODEL_PATH = 'models/random_forest_best_20251028_181817.joblib'

EXPECTED_FEATURE_ORDER = [
    "finbert_score", "sentiment_score", "finbert_label",
    "ema12_cross_ema26_up", "ema12_cross_ema26_down",
    "close_cross_sma50_up", "close_cross_sma50_down",
    "macd_cross_signal_up", "macd_cross_signal_down",
    "rsi_overbought", "rsi_oversold",
    "close_cross_upper_bb", "close_cross_lower_bb",
    "strong_trend"
]

EXCLUDE_COLS = ['open', 'high', 'low', 'close', 'volume', 'ticker', 'atr14']

# HELPER FUNCTIONS
def prepare_X_strict_order(X, expected_order=EXPECTED_FEATURE_ORDER, fill_value=0):
    """Return DataFrame with exact expected_order (add missing cols filled with fill_value)."""
    if not isinstance(X, pd.DataFrame):
        try:
            X = pd.DataFrame(X)
        except Exception:
            X = pd.DataFrame([X])
    for c in expected_order:
        if c not in X.columns:
            X[c] = fill_value
    return X.reindex(columns=expected_order)

# MODEL LOADING AND PREDICTION
class TrainedModelLoad:
    def __init__(self, path=MODEL_PATH):
        self.model = None
        self.path = path
        try:
            loaded = joblib.load(path)
            self.model = loaded.get("model", loaded) if isinstance(loaded, dict) else loaded
            print(f"Loaded ML model from {path}")
        except Exception as e:
            print(f"Warning: could not load model from {path} (error: {e}). Using stub fallback.")
            self.model = None

    def predict(self, features_df):
        """features_df: single-row DataFrame with columns in correct order."""
        if not isinstance(features_df, pd.DataFrame):
            features_df = prepare_X_strict_order(features_df)

        if self.model is not None:
            try:
                pred = self.model.predict(features_df)
                val = pred[0] if hasattr(pred, '__len__') else pred
                if isinstance(val, str):
                    v = val.upper()
                    if v in ('BUY', 'LONG'):
                        return 'BUY'
                    if v in ('SELL', 'SHORT'):
                        return 'HOLD'
                    return v
                try:
                    num = float(val)
                    return 'BUY' if num == 1.0 else 'HOLD'
                except Exception:
                    return 'HOLD'
            except Exception as e:
                print(f"Error during model prediction: {e}. Returning HOLD.")
                print(f"Features shape: {getattr(features_df, 'shape', None)}")
                return 'HOLD'

        # stub fallback: deterministic pseudo-signal
        try:
            idx0 = features_df.index.values[0]
            return 'BUY' if (int(idx0) % 10 == 0) else 'HOLD'
        except Exception:
            return 'HOLD'

# BACKTESTER CLASS
class Backtester:
    def __init__(self, data: pd.DataFrame):
        self.raw = data.copy()
        self.data = data.copy()
        self.capital = INITIAL_CAPITAL
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.trade_count = 0
        self.win_trades = 0
        self.trade_log = []
        self.model = TrainedModelLoad()

        if 'ticker' in self.data.columns and not self.data.empty:
            single = self.data['ticker'].iloc[0]
            self.data = self.data[self.data['ticker'] == single].copy()
            print(f"Backtesting for ticker {single}, rows: {len(self.data)}")

        if self.data.empty:
            print("No data after filtering.")

    def predict_signal(self, row: pd.Series):
        model_feature_excludes = EXCLUDE_COLS + ['candle_time']
        features = row.drop(model_feature_excludes, errors='ignore').to_frame().T
        features = prepare_X_strict_order(features)
        return self.model.predict(features)

    def run_backtest(self):
        if self.data.empty:
            return pd.DataFrame(self.trade_log)

        if 'candle_time' not in self.data.columns:
             print("Error: 'candle_time' column is missing, cannot log trade times correctly.")
             return pd.DataFrame(self.trade_log)

        for idx, row in self.data.iterrows():
            # check open position for SL/TP
            if self.position > 0:
                sl_level = self.entry_price * (1 - SL_FACTOR / 100)
                tp_level = self.entry_price * (1 + TP_FACTOR / 100)
                exit_price = None
                exit_reason = None

                if row.get('high', -np.inf) >= tp_level:
                    exit_price = tp_level
                    exit_reason = 'TP'
                elif row.get('low', np.inf) <= sl_level:
                    exit_price = sl_level
                    exit_reason = 'SL'

                if exit_price is not None:
                    pnl = (exit_price - self.entry_price) * self.position
                    self.capital += pnl
                    self.trade_count += 1
                    if pnl > 0:
                        self.win_trades += 1
                    self.trade_log.append({
                        'Entry_Time': self.entry_time,
                        'Exit_Time': row['candle_time'], # ИЗМЕНЕНИЕ 3: Используем candle_time
                        'Entry_Price': self.entry_price,
                        'Exit_Price': exit_price,
                        'PnL': pnl,
                        'Reason': exit_reason,
                        'Capital_After': self.capital
                    })
                    self.position = 0.0
                    self.entry_price = 0.0
                    self.entry_time = None

            # look for entry if flat
            if self.position == 0:
                sig = self.predict_signal(row)
                if sig == 'BUY':
                    capital_to_use = self.capital * TRADE_SIZE_PERCENT
                    price = row.get('close', None)
                    if not price or price == 0:
                        continue
                    position_size = capital_to_use / price
                    self.position = position_size
                    self.entry_price = price
                    self.entry_time = row['candle_time'] 

        # close leftover position at last close
        if self.position > 0 and not self.data.empty:
            exit_price = self.data.iloc[-1].get('close', self.entry_price)
            pnl = (exit_price - self.entry_price) * self.position
            self.capital += pnl
            self.trade_count += 1
            if pnl > 0:
                self.win_trades += 1
            self.trade_log.append({
                'Entry_Time': self.entry_time,
                'Exit_Time': self.data.iloc[-1]['candle_time'], 
                'Entry_Price': self.entry_price,
                'Exit_Price': exit_price,
                'PnL': pnl,
                'Reason': 'EOD',
                'Capital_After': self.capital
            })

        print("\n--- Backtest summary ---")
        print(f"Start capital: {INITIAL_CAPITAL:.2f}")
        print(f"Final capital: {self.capital:.2f}")
        total_return = (self.capital / INITIAL_CAPITAL - 1) * 100
        print(f"Total return: {total_return:.2f}%")
        print(f"Trades: {self.trade_count}")
        if self.trade_count > 0:
            print(f"Win rate: {self.win_trades / self.trade_count * 100:.2f}%")

        return pd.DataFrame(self.trade_log)

# PLOTTING FUNCTION
def plot_backtest(bt: Backtester, trades_df: pd.DataFrame, out_html="backtest_plot.html"):
    df_price = bt.data.copy()
    if df_price.empty:
        print("No price data to plot.")
        return

    if 'candle_time' in df_price.columns:
        df_price['candle_time'] = pd.to_datetime(df_price['candle_time'])
        x = df_price['candle_time']
    else:
        try:
            x = pd.to_datetime(df_price.index)
        except Exception:
            x = df_price.index

    y = df_price['close']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Close'))

    if trades_df is not None and not trades_df.empty:
        entries = trades_df[['Entry_Time', 'Entry_Price']].dropna()
        exits = trades_df[['Exit_Time', 'Exit_Price']].dropna()

        if not entries.empty:
            e_x = pd.to_datetime(entries['Entry_Time'], errors='coerce') 
            fig.add_trace(go.Scatter(
                x=e_x, y=entries['Entry_Price'],
                mode='markers', name='Entries',
                marker=dict(color='green', size=9, symbol='triangle-up')
            ))

        if not exits.empty:
            ex_x = pd.to_datetime(exits['Exit_Time'], errors='coerce') 
            fig.add_trace(go.Scatter(
                x=ex_x, y=exits['Exit_Price'],
                mode='markers', name='Exits',
                marker=dict(color='red', size=9, symbol='x')
            ))

    fig.update_layout(
        title=f"Backtest Close Price with Entries (green) and Exits (red) — {datetime.now().isoformat()}",
        xaxis_title="Time",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.write_html(out_html, include_plotlyjs='cdn')
    print(f"Saved interactive plot to {out_html}")
    try:
        fig.show()
    except Exception:
        pass

# MAIN FUNCTION
def main():
    print("🚀 Starting ETL + Backtest")

    try:
        start_time = detect_earliest_time_with_news()
    except SystemExit as e:
        print(e)
        sys.exit(1)

    df = load_data_from_db(start_time).iloc[15000:]  
    print("Loaded dataframe shape:", df.shape)
    print("NaNs per column (sample):")
    print(df.isna().sum()[lambda s: s > 0])

    if df.empty:
        print("No data to backtest.")
        return

    bt = Backtester(df)
    trades = bt.run_backtest()

    if trades is not None and not trades.empty:
        trades.to_csv("backtest_log.csv", index=False)
        print("Saved trade log to backtest_log.csv")
    else:
        print("No trades executed.")

    plot_backtest(bt, trades, out_html="backtest_plot.html")


if __name__ == "__main__":
    main()