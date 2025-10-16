def prepare_training_data(df):
    """
    Prepares the DataFrame for model training by removing non-feature columns.

    This function drops raw market data columns (such as OHLCV and related fields) 
    that are not intended to be used as features for machine learning models.

    Parameters:
        df (pd.DataFrame): DataFrame containing market data and indicators.

    Returns:
        pd.DataFrame: DataFrame with only feature columns, ready for training.
    """
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_trades', 'taker_base_vol']
    cols_to_drop = exclude_cols
    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df