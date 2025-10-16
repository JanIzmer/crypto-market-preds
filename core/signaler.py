import requests
from config import TELEGRAM_CONFIG
from core.users import load_users, save_user

def send_signal(symbol, signal, df, strategy_type):
    """
    Sends a formatted trading signal message to all registered Telegram users.

    Args:
        symbol (str): Trading pair (e.g. BTC/USDT).
        signal (str): Type of signal (e.g. "BUY", "SELL").
        df (DataFrame): Market data with the latest candle.
        strategy_type (str): Name or type of the strategy that generated the signal.
    """
    last = df.iloc[-1]

    msg = (
        f"📢 Signal: {signal}\n"
        f"📊 Pair: {symbol}\n"
        f"💰 Price: {last['close']:.2f}\n\n"
        f"Strategy: {strategy_type}\n"
    )

    send_telegram(msg)

def send_telegram(message):
    """
    Sends a message to all registered Telegram users.

    Args:
        message (str): Message to send.
    """
    users = load_users()
    for chat_id in users:
        _send_message(chat_id, message)

def _send_message(chat_id, text):
    """
    Sends a single message to a Telegram user.

    Args:
        chat_id (int): Telegram chat ID.
        text (str): Message content.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_CONFIG['bot_token']}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': text
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram error (chat_id: {chat_id}):", e)

def register_user_from_update(update_json):
    """
    Registers a new Telegram user from an incoming update.

    Args:
        update_json (dict): Telegram update payload.
    """
    try:
        chat_id = update_json["message"]["chat"]["id"]
        save_user(chat_id)
    except:
        pass
