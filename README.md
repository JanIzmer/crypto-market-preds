# Bybit_trading_bot
Trading Bot with Machine Learning

This repository contains a trading bot based on a two-stage machine learning model.  
It generates **BUY, SELL, or HOLD** signals using historical market data and technical indicators.

---

## 📁 Project Structure
```
├── core/               # Strategy logic and data preprocessing
├── static/             # Historical CSV files and generated reports
│   ├── data/           # Raw historical price data
│   ├── reports/        # Backtest results (plots, metrics)
├── tests/              # Backtesting and testing logic
├── .gitignore          # Ignored files for version control
├── README.md           # Project overview (you are here)
├── main.py             # Run backtests and generate reports
├── polling_bot.py      # Live trading signals with trained models
├── requirements.txt    # Python dependencies
├── stage1_model.pkl    # Stage 1 model (Trade vs Hold)
├── stage2_model.pkl    # Stage 2 model (Buy vs Sell)
├── users.json          # Sample user data for telegram bot (access control, signal delivery)
```

---

## 🔍 Description
- **Core module (`core/`)**:  
  Data preparation, feature engineering, and technical indicators.
- **Static (`static/`)**:  
  Contains historical price data and output reports with visualizations.
- **Models**:  
  - `stage1_model.pkl`: Decides whether to trade or hold  
  - `stage2_model.pkl`: Determines direction (BUY or SELL)

---

## ⚙️ Installation
```bash
git clone https://github.com/yourusername/Bybit_trading_bot.git
cd Bybit_trading_bot

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## 🚀 Usage

### Run backtests and generate reports (first you need to add your API key in settings)
```bash
pytest -s tests/
python main.py
```

### Run live signal generation
```bash
python polling_bot.py
```

---

## 🧠 Model Overview
The trading bot uses a **two-stage ML model**:

1. **Stage 1 – Trade vs Hold**  
   - LightGBM classifier  
   - Balanced with SMOTE  
   - Cross-validation with TimeSeriesSplit  

2. **Stage 2 – Buy vs Sell**  
   - LightGBM classifier  
   - Predicts trade direction  

**Features / Indicators used**:  
- Moving Averages (SMA, EMA, WMA)  
- RSI, MACD, Stochastic Oscillator  
- ATR (volatility measure)  
- Bollinger Bands  
- Volume-based indicators  

---

## 📊 Data
Input data format (`CSV`):
```
timestamp, open, high, low, close, volume
2024-01-01 00:00:00, 42000, 42100, 41900, 42050, 123.45
```

- Recommended timeframe: **1h**  
- Exchange: **Bybit**  

---

## 📈 Results  
- Win rate: 43%
- Total profit: -2.02% 


![Equity Curve](static/reports/btc_close_price.html)   

---

## 🔧 Configuration
- **`users.json`** – stores user/sample configuration (signal delivery, access control)  
- Parameters for training and testing can be adjusted in `config.py`  

---

## 🛠 Roadmap / TODO
- [ ] Add transformer-based models (TimesNet, Informer)      
- [ ] Web dashboard for live monitoring  

---

## ⚠️ Disclaimer
This project is for **educational and research purposes only**.  
It does **not constitute financial advice**. Use at your own risk.

---

## 📄 License
MIT License – feel free to use and modify.
