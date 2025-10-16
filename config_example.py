import ccxt


CONFIGS = {
    'BTC/USDT:USDT': {
        'symbol': 'BTC/USDT:USDT',
        'timeframe': '5m',
        'limit': 100,
        'interval_seconds': 100,
        'leverage': 5,
        'bull': {
            'order_amount': 0.0025,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.06,
            'strategy': 'bull_strategy',
        },
        'bear': {
            'order_amount': 0.0025,
            'stop_loss_pct': 0.025,
            'take_profit_pct': 0.04,
            'strategy': 'bear_strategy',
        },
        'sideways': {
            'order_amount': 0.0025,
            'stop_loss_pct': 0.015,
            'take_profit_pct': 0.03,
            'strategy': 'range_strategy',
        }
    },

    'ETH/USDT:USDT': {
        'symbol': 'ETH/USDT:USDT',
        'timeframe': '5m',
        'limit': 100,
        'interval_seconds': 100,
        'leverage': 5,
        'bull': {
            'order_amount': 0.02,
            'stop_loss_pct': 0.025,
            'take_profit_pct': 0.07,
            'strategy': 'bull_strategy',
        },
        'bear': {
            'order_amount': 0.02,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.045,
            'strategy': 'bear_strategy',
        },
        'sideways': {
            'order_amount': 0.02,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.035,
            'strategy': 'range_strategy',
        }
    }
}


API_CONFIG = {
    'apiKey': '',
    'secret': '',
    'enableRateLimit': True,
     "options": {
         'defaultSubType': 'linear',
        'defaultType': 'future', 
        "recvWindow": 1000000    
    },
    "adjustForTimeDifference": True
}

TELEGRAM_CONFIG = {
    'bot_token': ''
}
exchange = ccxt.bybit({
    'apiKey': '',
    'secret': '',
    'enableRateLimit': True,
    'adjustForTimeDifference': True,
    'options': {
        'defaultSubType': 'linear',
        'defaultType': 'future',
        'recvWindow': 1000000,
    },
})