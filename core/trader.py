from config import API_CONFIG, exchange
from core.orders import place_order, place_stop_take_orders

def is_in_position(symbol):
    try:
        positions = exchange.fetch_positions([symbol], params={"category": "linear"})
        for pos in positions:
            if pos['symbol'] == symbol and float(pos['contracts']) != 0:
                return True
        return False
    except Exception as e:
        print("Error fetching position:", e)
        return False

def handle_trade(config, signal, df):
    print(f"DEBUG: Symbol: {config['symbol']}, Order amount: {config['order_amount']}, Type: {type(config['order_amount'])}")
    symbol = config['symbol']
    entry_price = df['close'].iloc[-1]

    try:
        positions = exchange.fetch_positions([symbol])
        current_leverage = None
        for pos in positions:
            if pos['symbol'] == symbol:
                current_leverage = int(pos['leverage'])
                break

        if current_leverage != config['leverage']:
            exchange.set_leverage(config['leverage'], symbol, params={'category': 'linear'})
            print(f"Leverage updated to {config['leverage']} for {symbol}")
        else:
            print(f"Leverage already set to {config['leverage']} for {symbol}")

    except Exception as e:
        print(f"Error setting/getting leverage for {symbol}: {e}")

    in_position = is_in_position(symbol)

    if signal == "BUY" and not in_position:
        order = place_order(symbol, "buy", config['order_amount'])
        if order:
            stop_loss_price = entry_price * (1 - config['stop_loss_pct'])
            take_profit_price = entry_price * (1 + config['take_profit_pct'])
            place_stop_take_orders(symbol, "buy", config['order_amount'], stop_loss_price, take_profit_price)

    elif signal == "SELL" and not in_position:
        order = place_order(symbol, "sell", config['order_amount'])
        if order:
            stop_loss_price = entry_price * (1 + config['stop_loss_pct'])
            take_profit_price = entry_price * (1 - config['take_profit_pct'])
            place_stop_take_orders(symbol, "sell", config['order_amount'], stop_loss_price, take_profit_price)


