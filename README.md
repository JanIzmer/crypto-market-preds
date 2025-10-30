# Crypto Market Predictor & Trading System
Project: Algorithmic Trading Bot utilizing Technical and Sentiment Analysis

This repository documents a robust, scalable system designed for generating precise trading signals in the cryptocurrency market. The system integrates machine learning with both traditional technical indicators and specialized news sentiment analysis.

---

## Project Structure
```
├── core/               # Strategy logic and data preprocessing
├── database            # Database creation and data loading
├── tests/              # Backtesting and testing logic
├── notebooks           # Main data analyst/model training folder with ipynb
├── .gitignore          # Ignored files for version control
├── README.md           # Project overview (you are here)
├── requirements.txt    # Python dependencies
├── stage1_model.pkl    # Stage 1 model (Trade vs Hold)
├── stage2_model.pkl    # Stage 2 model (Buy vs Sell)
├── users.json          # Sample user data for telegram bot (access control, signal delivery)
```

---

## Description
The primary goal is to build a secure, readable, and scalable trading system focused on achieving a positive mathematical edge. This was accomplished by overcoming challenges like class imbalance and noisy raw data.
Trading Signal Logic
The model generates one of two signals:
BUY: A high-confidence signal to enter a long position.
HOLD: A signal to remain passive and await a favorable opportunity.
Key Performance Indicators (KPIs)
The minimum target was set to ensure profitable trading under a 1:2 Take Profit (TP) to Stop Loss (SL) ratio:
Precision (Accuracy) ≥ 0.34
Recall (Completeness) > 0.35

---

## Model Overview
The trading system uses a single-stage classification model optimized for high interpretability and stability.
Final Model Chosen: 
Algorithm: Random Forest Classifier (Selected for its robust performance and interpretability over Linear Regression and Gradient Boosting).
Best Validation Precision: ~0.375 (Exceeding the target of 0.34).
Feature Engineering Strategy
The predictive power comes from transforming raw data into high-quality features:
Technical Indicators: Calculated using standard OHLCV data.
Sentiment Analysis: Pre-trained Open Source models specialized for financial text are used to score news articles.
Binary Flags (Key Decision): To give the low-level model temporal intelligence without complex Attention mechanisms, all raw indicator values were converted into binary flag features based on thresholds. These flags also incorporate state information from past candles (historical lookback).

---

## Data

Data cleanliness and structure were critical to project success.
Data Sources: OHLCV data (Exchange APIs) and a news dataset covering 2021 to 2024.
Database: MySQL using the INNODB engine was selected for its proven scalability and transactional integrity. The database structure was mapped out in an ERD (Stage 2.1).
Handling Missing News Data
News data often contains critical missing values. The following strategy was implemented:
Strategy: Forward Fill (ffill) applied with a 5-hour limit.
Justification: A news item's impact is assumed to persist for up to 5 hours. If no news is present within that window, the feature is neutralized by being set to 0.

---

## Results  
From July 2023 to January 2024
Start capital: 10000.00
Final capital: 11826.21
Total return: 18.26%
Trades: 116
Win rate: 37.07%


---

## Roadmap / TODO
The primary goal (positive financial result and stability) has been achieved. Future improvements can focus on enhancing signal relevance and data volume:
News Source Expansion: Integrate new, more timely sources (e.g., Twitter/X) for faster sentiment capture.
NLP Model Retraining: Develop or retrain a specialized news assessment model for more accurate sentiment scoring tailored to crypto markets.
Dataset Expansion: Increase the training data volume to further improve the model's generalization ability.


## Disclaimer
This project is for **educational and research purposes only**.  
It does **not constitute financial advice**. Use at your own risk.

---

## 📄 License
MIT License – feel free to use and modify.
