import pandas as pd
from core.data_clean_functions.prepare_data_for_model import prepare_training_data as prepare_data

def generate_signal(df, model_stage1, model_stage2, position=None, threshold=0.6):
    """
    Generates a trading signal ('BUY', 'SELL', or 'HOLD') based on a two-stage machine learning process.

    Stage 1: Determines whether to trade or hold (binary classification).
    Stage 2: If a trade is decided, determines the direction ('BUY' or 'SELL') (multiclass classification).

    Parameters:
        df (pd.DataFrame): DataFrame containing market data and technical indicators.
        model_stage1: First-stage model to classify TRADE vs HOLD.
        model_stage2: Second-stage model to classify BUY (0), HOLD (1), or SELL (2).
        position(string): Short or long or None
        threshold (float): Probability threshold for initiating a trade (default is 0.6).

    Returns:
        str: A trading signal - one of 'BUY', 'SELL', or 'HOLD'.
    """
    # If we already in trade we increase threshold for initiating a trade to 0.75(test)
    if position == None:
        threshold = 0.7
    else:
        threshold = 0.8

    # Check if input data is too short or empty
    if df.empty or len(df) < 50:
        print("Too few rows in df")
        return 'HOLD'

    # Drop unnecessary columns and prepare features
    df = prepare_data(df.copy())
    last = df.iloc[-1]  # Use the last row (most recent data point)

    # Convert the last row into a single-row DataFrame for prediction
    features = last.to_dict()
    X = pd.DataFrame([features])

    # Check for missing values
    if X.isnull().values.any():
        print("NaNs found:", X.columns[X.isnull().any()])
        return 'HOLD'

    # Stage 1: Predict whether to trade
    try:
        prob_stage1 = model_stage1.predict_proba(X)[0][1]  # Probability of class 'TRADE'
        if prob_stage1 < threshold:
            return 'HOLD'  # If probability is too low, skip trading
    except Exception as e:
        print("Error in stage1 prediction:", e)
        return 'HOLD'

    # Stage 2: Predict the trade direction (BUY or SELL)
    try:
        pred_stage2 = model_stage2.predict(X)[0]
    except Exception as e:
        print("Error in stage2 prediction:", e)
        return 'HOLD'

    # Interpret the result
    if pred_stage2 == 0:
        return 'BUY'
    elif pred_stage2 == 2:
        return 'SELL'
    else:
        print("Unexpected label:", pred_stage2)
        return 'HOLD'
