# imports
import os
import pandas as pd
from sqlalchemy import create_engine, text
from database.db_config import DATABASE_URL  

# Environment variable overrides
KLINE_TABLE = os.getenv('KLINE_TABLE', 'kline_data')
TI_TABLE = os.getenv('TI_TABLE', 'technical_indicators')
MS_TABLE = os.getenv('MS_TABLE', 'market_sentiment')
CHUNKSIZE = int(os.getenv('CHUNKSIZE', '50000'))
OUTPUT_PARQUET = os.getenv('OUTPUT_PARQUET', 'combined_simple.parquet')


def detect_earliest_time_with_news():
    """
    Returns the earliest publication_time from the market_sentiment table.
    Raises SystemExit if no rows found or on DB error.
    """
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            earliest = conn.execute(text(f"SELECT MIN(publication_time) FROM `{MS_TABLE}`")).scalar()
    finally:
        engine.dispose()

    if earliest is None:
        raise SystemExit('No publication_time found in market_sentiment table')
    print('DB:', DATABASE_URL)
    return earliest


def load_data_from_db(start_time):
    """
    Load joined data in chunks starting from start_time (inclusive).
    start_time should be a string or datetime compatible with the DB.
    Returns a pandas DataFrame (possibly empty) and writes parquet to OUTPUT_PARQUET.
    """
    if start_time is None:
        raise ValueError("start_time must be provided")

    # If user passed a datetime, convert to string acceptable by DB
    if hasattr(start_time, "isoformat"):
        start_time_param = str(start_time)
    else:
        start_time_param = start_time

    print('Earliest publication_time:', start_time_param)

    sql = f"""
    SELECT
      k.*,
      ti.*,
      ms_agg.finbert_score_mean AS finbert_score,
      ms_agg.sentiment_score_mean AS sentiment_score,
      ms_single.finbert_label AS finbert_label
    FROM `{KLINE_TABLE}` k
    LEFT JOIN `{TI_TABLE}` ti
      ON k.ticker = ti.ticker AND k.candle_time = ti.candle_time
    LEFT JOIN (
      SELECT ticker, candle_time, AVG(finbert_score) AS finbert_score_mean, AVG(sentiment_score) AS sentiment_score_mean
      FROM `{MS_TABLE}`
      GROUP BY ticker, candle_time
    ) ms_agg
      ON k.ticker = ms_agg.ticker AND k.candle_time = ms_agg.candle_time
    LEFT JOIN (
      SELECT ticker, candle_time, MIN(finbert_label) AS finbert_label
      FROM `{MS_TABLE}`
      GROUP BY ticker, candle_time
    ) ms_single
      ON k.ticker = ms_single.ticker AND k.candle_time = ms_single.candle_time
    WHERE k.candle_time >= :start_time
    ORDER BY k.ticker, k.candle_time ASC
    """

    print('SQL prepared. Running query...')

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    chunks = []
    try:
        with engine.connect() as conn:
            try:
                for chunk in pd.read_sql_query(text(sql), conn, params={'start_time': start_time_param}, chunksize=CHUNKSIZE):
                    print('Loaded chunk:', len(chunk))
                    chunks.append(chunk)
            except Exception as e:
                # helpful debugging message
                raise RuntimeError(f"Error while executing query: {e}")
    finally:
        engine.dispose()

    if not chunks:
        df = pd.DataFrame()
    else:
        df = pd.concat(chunks, ignore_index=True)

    print('Total rows loaded:', len(df))

    if not df.empty:
        # remove duplicated columns that can arise from SELECT k.*, ti.* etc.
        df = df.loc[:, ~df.columns.duplicated()]    
        # dataclearing: fill NaNs in technical indicators with last valid value (forward fill)
        df['sentiment_score'] = df['sentiment_score'].ffill(limit=5)
        df['finbert_score'] = df['finbert_score'].ffill(limit=5)
        df['finbert_label'] = df['finbert_label'].ffill(limit=5)

        #replace remaining NaNs
        df['sentiment_score'].fillna(0, inplace=True)
        df['finbert_score'].fillna(0, inplace=True)
        df['finbert_label'].fillna('neutral', inplace=True)
        df['finbert_label'] = df['finbert_label'].map({'positive': 1, 'neutral': 0, 'negative': -1}).astype(int)
        # Try to write parquet with fastparquet first, fallback to pyarrow
        try:
            df.to_parquet(OUTPUT_PARQUET, index=False, engine='fastparquet')
            print('Saved to', OUTPUT_PARQUET, '(fastparquet)')
        except Exception as e_fp:
            try:
                df.to_parquet(OUTPUT_PARQUET, index=False, engine='pyarrow')
                print('Saved to', OUTPUT_PARQUET, '(pyarrow)')
            except Exception as e_pa:
                raise RuntimeError(f"Failed to write parquet with fastparquet ({e_fp}) and pyarrow ({e_pa})")

    else:
        print('No data to save')

    return df

