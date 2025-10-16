import ccxt
from config import API_CONFIG

# Initialize Bybit exchange
exchange = ccxt.bybit(API_CONFIG)

def place_order(symbol, side, amount, price=None):
    """
    Places a market or limit order on Bybit.

    Args:
        symbol (str): Trading pair (e.g., "BTC/USDT").
        side (str): "buy" or "sell".
        amount (float): Quantity to trade.
        price (float, optional): If provided, places a limit order at this price. Otherwise, a market order.

    Returns:
        dict or None: Order object or None on error.
    """
    try:
        position_idx = 1 if side.lower() == "buy" else 2
        params = {
            "category": "linear",
            "positionIdx": position_idx,
        }
        amount = float(amount)
        if price is not None:
            price = float(price)
            order = exchange.create_limit_order(symbol, side, amount, price, params=params)
        else:
            order = exchange.create_market_order(symbol, side, amount, params=params)

        print(f"✅ Order {side.upper()} executed: {amount} {symbol}")
        return order
    except Exception as e:
        print("❌ Error placing order:", e)
        return None


def place_stop_take_orders(symbol, side, amount, stop_loss_price, take_profit_price):
    """
    Sets stop-loss and take-profit orders after a position is opened.

    Args:
        symbol (str): Trading pair (e.g., "BTC/USDT").
        side (str): The original position side ("buy" or "sell").
        amount (float): Quantity to close.
        stop_loss_price (float): Price at which to trigger stop-loss.
        take_profit_price (float): Price at which to trigger take-profit.
    """
    try:
        opposite_side = "sell" if side == "buy" else "buy"
        category = {"category": "linear"}

        # Stop-loss order
        exchange.create_order(
            symbol=symbol,
            type='STOP_MARKET',
            side=opposite_side,
            amount=amount,
            price=None,
            params={
                **category,
                'stopPrice': stop_loss_price,
                'reduceOnly': True,
                'closeOnTrigger': True,
            }
        )

        # Take-profit order
        exchange.create_order(
            symbol=symbol,
            type='TAKE_PROFIT_MARKET',
            side=opposite_side,
            amount=amount,
            price=None,
            params={
                **category,
                'stopPrice': take_profit_price,
                'reduceOnly': True,
                'closeOnTrigger': True,
            }
        )

        print(f"✅ Stop-loss and take-profit set for {symbol}")
    except Exception as e:
        print("❌ Error setting stop-loss/take-profit:", e)


