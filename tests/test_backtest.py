import sys
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving plots
import plotly.graph_objects as go
import webbrowser


# Add parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.strategytest import run_backtest
from core.market_type import MarketTrendDetector


def generate_report(df60, trades, balance_curve):
    """
    Generates performance metrics and saves a visual report based on backtest results.

    Parameters:
        df60 (pd.DataFrame): Original 60-minute OHLCV candles with timestamp.
        trades (list): List of executed trades returned by the backtest.
        balance_curve (list): List of balance values over time.

    Output:
        Saves a chart with price, trades, and trend regions to 'static/reports/btc_close_price.png'
        Prints performance metrics to stdout.
    """
    df_trades = pd.DataFrame(trades)
    if trades != []:
        # Basic performance statistics
        df_trades['profit_abs'] = (df_trades['profit_pct'] / 100) + 1
        total_profit_pct = (df_trades['profit_abs'].prod() - 1) * 100
        win_rate = (df_trades['profit_pct'] > 0).mean() * 100
        avg_profit = df_trades['profit_pct'].mean()

        # Profit summary per trend
        breakdown = df_trades.groupby("trend")['profit_pct'].agg(['count', 'mean', 'sum'])
        breakdown_reason = df_trades.groupby(['trend', 'exit_reason']).size().unstack(fill_value=0)

        # Drawdown calculation
        equity = pd.Series(balance_curve)
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min() * 100

        # Print performance results
        print("\n📈 === OVERALL METRICS ===")
        print(f"Trades executed: {len(df_trades)}")
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Average profit per trade: {avg_profit:.2f}%")
        print(f"Total profit: {total_profit_pct:.2f}%")
        print(f"Max drawdown: {max_drawdown:.2f}%")

        print("\n📊 === BY MARKET TYPE ===")
        print(breakdown)
        print("\n")
        print(breakdown_reason)

        # Convert timestamp to datetime (and drop timezone)
        df60['timestamp'] = pd.to_datetime(df60['timestamp'], unit='ms')

        # Detect trend for each candle
        trends = []
        window = 200
        trend_detector = MarketTrendDetector()
        for i in range(len(df60)):
            if i < window - 1:
                trends.append(None)
            else:
                df_sub = df60.iloc[i - window + 1 : i + 1]
                trend = trend_detector.detect_market_trend(df=df_sub)
                trends.append(trend)
        df60['trend'] = trends


        fig = go.Figure()

        # Plot close price as a line
        fig.add_trace(go.Scatter(
            x=df60['timestamp'],
            y=df60['close'],
            mode='lines',
            name='BTC/USDT Close Price',
            line=dict(color='blue')
        ))

        # Background shading by trend type
        colors = {
            'bull': 'rgba(0, 255, 0, 0.2)',        # green
            'bear': 'rgba(255, 0, 0, 0.2)',        # red
            'sideways': 'rgba(128, 128, 128, 0.2)',# gray
            None: 'rgba(255, 255, 255, 0.2)'       # white (undefined)
        }

        # Group by consecutive trend segments
        start_idx = 0
        for i in range(1, len(df60)):
            if df60['trend'].iloc[i] != df60['trend'].iloc[i - 1]:
                trend = df60['trend'].iloc[i - 1]
                if trend:
                    fig.add_vrect(
                        x0=df60['timestamp'].iloc[start_idx],
                        x1=df60['timestamp'].iloc[i - 1],
                        fillcolor=colors.get(trend, 'white'),
                        opacity=0.2,
                        line_width=0
                    )
                start_idx = i

        
        final_trend = df60['trend'].iloc[-1]
        if final_trend:
            fig.add_vrect(
                x0=df60['timestamp'].iloc[start_idx],
                x1=df60['timestamp'].iloc[-1],
                fillcolor=colors.get(final_trend, 'white'),
                opacity=0.2,
                line_width=0
            )

        # Add trade entry and exit markers
        for trade in trades:
            trade_time = pd.to_datetime(trade['entry_time'], unit='ms')
            exit_time = pd.to_datetime(trade['exit_time'], unit='ms')
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']

            if trade['position'] == 'short':
                # Short entry (red)
                fig.add_trace(go.Scatter(
                    x=[trade_time], y=[entry_price],
                    mode='markers',
                    marker=dict(color='red', size=8),
                    name='Short Entry'
                ))
                # Exit (yellow)
                fig.add_trace(go.Scatter(
                    x=[exit_time], y=[exit_price],
                    mode='markers',
                    marker=dict(color='yellow', size=8),
                    name='Exit'
                ))
            elif trade['position'] == 'long':
                # Long entry (green)
                fig.add_trace(go.Scatter(
                    x=[trade_time], y=[entry_price],
                    mode='markers',
                    marker=dict(color='green', size=8),
                    name='Long Entry'
                ))
                # Exit (yellow)
                fig.add_trace(go.Scatter(
                    x=[exit_time], y=[exit_price],
                    mode='markers',
                    marker=dict(color='yellow', size=8),
                    name='Exit'
                ))

        # Layout settings
        fig.update_layout(
            title='BTC/USDT Close Price with Trades and Trend Background',
            xaxis_title='Time',
            yaxis_title='Price (USDT)',
            template='plotly_white',
            legend=dict(orientation="h")
        )

        # Export to interactive HTML file
        output_path = "static/reports/btc_close_price.html"
        fig.write_html(output_path)

        # Open the HTML file in the default browser
        file_path = os.path.abspath("static/reports/btc_close_price.html")
        webbrowser.open(f"file:///{file_path}")

    else:
        print('Trades is empty')


def test_backtest_runs(rows=5000):
    """
    Loads historical data, runs backtest, generates report, and performs basic assertions.

    Parameters:
        rows (int): Number of rows (candles) to read from the CSV file.

    Output:
        Prints summary and saves report image.
        Raises AssertionError if output is not as expected.
    """
    df = pd.read_csv("static/data/BTCUSDT_60m.csv", nrows=rows)
    df.columns = df.columns.str.lower()

    final_balance, curve, trades = run_backtest(df)

    print(f"\nFinal balance: {final_balance:.2f} USDT")
    print(f"Total trades: {len(trades)}")

    generate_report(df, trades, curve)

    # Basic sanity checks
    assert isinstance(final_balance, (int, float))
    assert isinstance(curve, list)
    assert isinstance(trades, list)
    assert final_balance > 0
    assert len(curve) > 0
    assert all('entry_price' in t and 'exit_price' in t and 'profit_pct' in t for t in trades)



