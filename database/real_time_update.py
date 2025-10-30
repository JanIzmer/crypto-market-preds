import os
import uuid
import sys
import time
import logging
import json
from urllib.parse import urljoin 
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sqlalchemy import create_engine, text, Engine

# Placeholder for db_config, assuming it provides these
try:
    from db_config import DATABASE_URL, get_db_engine 
except ImportError:
    # Fallback for self-contained execution if db_config is not available
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    def get_db_engine(db_url: str) -> Engine:
        return create_engine(db_url, pool_pre_ping=True)

# -----------------------------
# CONFIG / LOGGING
# -----------------------------
# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CRYPTOPANIC_API_TOKEN = os.getenv("CRYPTOPANIC_API_TOKEN", '80c24f44484b22771d0e63498b4d4cf9177d6f38')  # <-- required
CRYPTOPANIC_BASE = "https://cryptopanic.com/api/developer/v2/posts/"

START_DATE = datetime(2017, 8, 17, 4, 0, 0)  # naive UTC assumed for storage

# -----------------------------
# SENTIMENT
# -----------------------------
analyzer = SentimentIntensityAnalyzer()

def sentiment_score_vader(text: str) -> float:
    """Computes the VADER compound sentiment score for a given text."""
    if not text:
        return 0.0
    return float(analyzer.polarity_scores(text)["compound"])

# -----------------------------
# TABLE CREATION
# -----------------------------
def create_tables(engine: Engine):
    """
    Create news_source and market_sentiment tables if they do not exist.
    Adds indexes for performance on foreign keys and time-based queries.
    """
    logger.info("Checking/creating database tables...")
    with engine.connect() as conn:
        # Create news_source table with index on name
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS news_source (
                source_id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                url_base VARCHAR(500),
                INDEX idx_source_name (name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))
        
        # Create market_sentiment table with FK and indexes on time fields
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS market_sentiment (
                id VARCHAR(64) PRIMARY KEY,
                publication_time DATETIME NOT NULL,
                source_id VARCHAR(255) NOT NULL,
                headline TEXT NOT NULL,
                content TEXT,
                sentiment_score FLOAT,
                ticker VARCHAR(50),
                candle_time DATETIME NOT NULL,
                CONSTRAINT fk_sentiment_source FOREIGN KEY (source_id)
                    REFERENCES news_source(source_id)
                    ON UPDATE CASCADE
                    ON DELETE CASCADE,
                INDEX idx_sentiment_time (publication_time),
                INDEX idx_sentiment_candle (candle_time)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))
        conn.commit() # Commit DDL changes
    logger.info("Tables verified/created.")

# -----------------------------
# HELPERS
# -----------------------------
def to_naive_utc(dt_like: Any) -> datetime:
    """
    Convert pandas-parsed or datetime-like object to naive UTC datetime 
    for consistent DB storage.
    
    Args:
        dt_like: String, pandas.Timestamp, or datetime object.
        
    Returns:
        A timezone-naive datetime object representing UTC time.
    """
    ts = pd.to_datetime(dt_like, utc=True)
    # ts is timezone-aware in UTC; make naive by removing tzinfo
    dt = ts.to_pydatetime()
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt

def align_to_candle_time(dt: datetime, interval_hours: int = 1) -> datetime:
    """
    Aligns a datetime object to the start of the nearest hourly candle.
    
    Args:
        dt: The datetime object to align.
        interval_hours: The interval size in hours (e.g., 1 for hourly, 4 for 4-hour candles).
        
    Returns:
        The aligned datetime object (e.g., 2023-10-28 10:00:00).
    """
    aligned = dt.replace(minute=0, second=0, microsecond=0)
    if interval_hours > 1:
        hour = aligned.hour
        aligned = aligned.replace(hour=(hour // interval_hours) * interval_hours)
    return aligned

# -----------------------------
# CRYPTOPANIC FETCHER
# -----------------------------
def fetch_from_cryptopanic(max_requests: int = 2000, per_page: int = 500, sleep_between: float = 0.5) -> List[Dict[str, Any]]:
    """
    Cursor-based fetcher for CryptoPanic developer API.
    
    - Follows the 'next' link returned in the JSON response for pagination.
    - Stops when the 'next' link is None or when the newest article on a page 
      is older than the global START_DATE.
      
    Args:
        max_requests: Maximum number of API requests to make.
        per_page: Number of posts to request per page.
        sleep_between: Time in seconds to sleep between requests to respect rate limits.

    Returns:
        A list of normalized news items (dictionaries).
    """
    if not CRYPTOPANIC_API_TOKEN:
        logger.error("CRYPTOPANIC_API_TOKEN environment variable not set. Cannot fetch data.")
        return []

    all_items: List[Dict[str, Any]] = []

    # initial URL with params
    params = {
        "auth_token": CRYPTOPANIC_API_TOKEN,
        "public": "true",
        "per_page": per_page,
    }
    
    # Build initial request URL
    base_url = CRYPTOPANIC_BASE
    
    try:
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed initial request to CryptoPanic API: {e}")
        return all_items
        
    request_count = 1
    page_idx = 1
    data_root = None

    try:
        data_root = resp.json()
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing initial response JSON: {e}")
        return all_items

    while data_root:
        results = data_root.get("results") or []
        logger.info(f"CryptoPanic: page_index={page_idx} received {len(results)} results (per_page={per_page}).")

        if not results:
            logger.info(f"No results on current page_index {page_idx} — stopping pagination.")
            break

        page_pub_times = []
        added_this_page = 0
        skipped_old_on_page = 0

        for item in results:
            pub_raw = item.get("published_at") or item.get("created_at") or None
            pub_time: datetime
            
            if pub_raw:
                try:
                    pub_time = to_naive_utc(pub_raw)
                except Exception:
                    pub_time = datetime.now(timezone.utc).replace(tzinfo=None)
            else:
                pub_time = datetime.now(timezone.utc).replace(tzinfo=None)

            page_pub_times.append(pub_time)

            if pub_time < START_DATE:
                skipped_old_on_page += 1
                continue

            # --- Data Extraction and Normalization ---
            title = item.get("title") or ""
            content = item.get("description") or item.get("body") or item.get("excerpt") or ""
            
            currencies = item.get("currencies") or []
            ticker = "unknown"
            if currencies and isinstance(currencies, list):
                c0 = currencies[0]
                if isinstance(c0, dict):
                    ticker = c0.get("code") or c0.get("title") or c0.get("slug") or "unknown"
                else:
                    ticker = str(c0)

            source_data = item.get("source") or {}
            source_name = source_data.get("title") or item.get("domain") or "cryptopanic"
            source_url = source_data.get("url") or item.get("site") or ""

            cp_id = item.get("id") or item.get("slug") or str(uuid.uuid4())
            record_id = f"cryptopanic_{cp_id}"

            combined = f"{title} {content}"
            score = sentiment_score_vader(combined)

            all_items.append({
                "id": record_id,
                "cryptopanic_id": str(cp_id),
                "publication_time": pub_time,
                "source_name": source_name,
                "source_url": source_url,
                "headline": title,
                "content": content,
                "sentiment_score": score,
                "ticker": ticker,
            })
            added_this_page += 1

        logger.info(f"Page {page_idx}: added {added_this_page}, skipped_old {skipped_old_on_page}, total_collected {len(all_items)}")

        # Stop check 1: If the newest article on this page is already older than START_DATE
        if page_pub_times:
            # We check the newest for safety, but the API generally returns newest first
            newest_on_page = max(page_pub_times) 
            if newest_on_page < START_DATE:
                logger.info(f"Page {page_idx}: newest_on_page {newest_on_page} < START_DATE {START_DATE} -> stopping.")
                break

        # Stop check 2: Get next link from JSON (cursor-based)
        next_val = data_root.get("next")
        if not next_val:
            logger.info("No 'next' link in response -> reached end of pages.")
            break

        # Stop check 3: Request count cap
        if request_count >= max_requests:
            logger.warning(f"Reached max_requests={max_requests} -> stopping to avoid over-requesting.")
            break

        # Build absolute next_url if relative (safer implementation)
        next_url = next_val if next_val.startswith("http") else urljoin(base_url, next_val)
        
        # Sleep to respect rate limits
        time.sleep(sleep_between)

        # Fetch next page
        try:
            resp = requests.get(next_url, timeout=60)
            resp.raise_for_status()
            data_root = resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"ERROR fetching next page ({next_url}): {e}")
            break
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from next page ({next_url}): {e}")
            break

        request_count += 1
        page_idx += 1

    logger.info(f"Finished fetching. Total {len(all_items)} items collected (requests made: {request_count}).")
    return all_items

# -----------------------------
# DATAFRAME PREP / UPSERT
# -----------------------------
def prepare_source_data(raw_data: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the raw data list into two DataFrames: one for unique news sources 
    and one for the news articles themselves.
    
    Args:
        raw_data: List of dictionaries containing raw news data.
        
    Returns:
        A tuple: (df_sources, df_news)
    """
    if not raw_data:
        return pd.DataFrame(), pd.DataFrame()
    df_news = pd.DataFrame(raw_data)
    df_sources = df_news[["source_name", "source_url"]].drop_duplicates().copy()
    df_sources.rename(columns={"source_name": "name", "source_url": "url_base"}, inplace=True)
    
    # Create normalized, clean source_id
    df_sources["source_id"] = df_sources["name"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True).str.strip()
    
    # Handle cases where the normalization results in an empty ID
    df_sources['source_id'] = df_sources['source_id'].apply(lambda x: uuid.uuid4().hex if not x else x)

    df_sources = df_sources[["source_id", "name", "url_base"]].reset_index(drop=True)
    return df_sources, df_news

def get_source_map_from_db(engine: Engine) -> Dict[str, str]:
    """
    Fetches the mapping of source name to its canonical source_id from the database.
    
    Args:
        engine: The SQLAlchemy Engine object.
        
    Returns:
        A dictionary mapping source names (str) to source IDs (str).
    """
    logger.info("Fetching definitive source map from DB...")
    sql = text("SELECT name, source_id FROM news_source;")
    source_map: Dict[str, str] = {}
    with engine.connect() as conn:
        try:
            result = conn.execute(sql)
            source_map = {row.name: row.source_id for row in result}
            logger.info(f"Successfully fetched {len(source_map)} sources.")
        except Exception as e:
            logger.error(f"Failed to fetch source map from news_source: {e}")
            
    return source_map

def finalize_sentiment_data(df_news: pd.DataFrame, source_map: Dict[str, str]) -> pd.DataFrame:
    """
    Finalizes the news DataFrame by mapping source names to source IDs and 
    calculating the candle_time for aggregation.
    
    Args:
        df_news: DataFrame containing news records.
        source_map: Dictionary mapping source names to source IDs.
        
    Returns:
        DataFrame ready for upsert into market_sentiment.
    """
    if df_news.empty:
        return pd.DataFrame()
        
    df_news["source_id"] = df_news["source_name"].map(source_map)
    df_news["candle_time"] = df_news["publication_time"].apply(align_to_candle_time)
    
    df_sentiment = df_news[[
        "id", "publication_time", "source_id", "headline", "content",
        "sentiment_score", "ticker", "candle_time"
    ]].copy()
    
    # Only keep records that successfully mapped to a source_id (i.e., were upserted)
    df_sentiment = df_sentiment[df_sentiment["source_id"].notnull()]
    
    # Ensure all datetime objects are Python's native datetime (naive) for SQLAlchemy consistency
    for col in ['publication_time', 'candle_time']:
        df_sentiment[col] = df_sentiment[col].apply(
            lambda x: x.to_pydatetime() if isinstance(x, pd.Timestamp) else x
        )
        
    return df_sentiment

def upsert_news_source(engine: Engine, df_sources: pd.DataFrame):
    """
    Inserts or updates news source records in the news_source table using 
    ON DUPLICATE KEY UPDATE.
    
    Args:
        engine: The SQLAlchemy Engine object.
        df_sources: DataFrame containing the unique source records to upsert.
    """
    if df_sources.empty:
        logger.info("No sources to upsert.")
        return
        
    insert_cols = ["source_id", "name", "url_base"]
    cols_sql = ", ".join(f"`{c}`" for c in insert_cols)
    vals_sql = ", ".join(f":{c}" for c in insert_cols)
    update_sql = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in insert_cols[1:])
    
    sql = text(f"""
        INSERT INTO news_source ({cols_sql})
        VALUES ({vals_sql})
        ON DUPLICATE KEY UPDATE {update_sql};
    """)
    
    records = df_sources[insert_cols].to_dict(orient="records")
    
    with engine.connect() as conn:
        try:
            with conn.begin():
                if records:
                    conn.execute(sql, records)
                    logger.info(f"Upserted {len(records)} unique sources into news_source.")
        except Exception as e:
            logger.error(f"ERROR upserting news_source: {e}")
            raise

def upsert_market_sentiment(engine: Engine, df_sentiment: pd.DataFrame):
    """
    Upsert market_sentiment records. It performs FK-safety checks by querying 
    the 'kline_data' table to ensure that a required (ticker, candle_time) pair exists.
    If the pair does not exist, the 'ticker' is set to NULL to prevent FK violation.
    
    Args:
        engine: The SQLAlchemy Engine object.
        df_sentiment: DataFrame containing the sentiment records to upsert.
    """
    if df_sentiment.empty:
        logger.info("No sentiment records to upsert.")
        return

    df = df_sentiment.copy()

    # 1. Normalize placeholder tickers to None
    df['ticker'] = df['ticker'].replace({'': None, 'unknown': None}).where(df['ticker'].notnull(), None)

    # 2. Build set of unique non-null (ticker, candle_time) pairs to check in kline_data
    pairs = df[['ticker', 'candle_time']].dropna().drop_duplicates().reset_index(drop=True)

    existing_pairs = set()
    if not pairs.empty:
        # Build optimized WHERE clause using parameterized queries
        conds = []
        params = {}
        for i, row in enumerate(pairs.itertuples(index=False), start=0):
            t_key = f"t{i}"
            c_key = f"c{i}"
            conds.append(f"(ticker = :{t_key} AND candle_time = :{c_key})")
            params[t_key] = row.ticker 
            # ensure datetime is python datetime (not pandas Timestamp)
            params[c_key] = (row.candle_time.to_pydatetime() if isinstance(row.candle_time, pd.Timestamp) else row.candle_time)

        where_clause = " OR ".join(conds)
        sql = text(f"SELECT ticker, candle_time FROM kline_data WHERE {where_clause};")
        
        with engine.connect() as conn:
            try:
                res = conn.execute(sql, params)
                for r in res:
                    existing_pairs.add((r.ticker, r.candle_time)) 
                logger.debug(f"Validated {len(existing_pairs)} existing ticker pairs in kline_data.")
            except Exception as e:
                # If the kline_data table doesn't exist or query fails, warn and proceed without validation
                logger.warning(f"Could not query kline_data for FK validation (table likely missing/schema issue). Setting tickers to NULL for safety. Error: {e}")
                existing_pairs = set() 
                
    # 3. Validate each row: if its (ticker, candle_time) not in existing_pairs -> nullify ticker
    def _validate_ticker(row):
        t = row['ticker']
        ct = row['candle_time']
        
        if t is None:
            return None 
            
        # Convert pandas Timestamp to python datetime for comparison if necessary
        ct_py = ct.to_pydatetime() if isinstance(ct, pd.Timestamp) else ct
        
        # Check against the set of existing pairs
        if (t, ct_py) in existing_pairs:
            return t
        
        # If the pair is not found, nullify the ticker
        return None

    df['ticker'] = df.apply(_validate_ticker, axis=1)

    # 4. Proceed with upsert
    insert_cols = ["id", "publication_time", "source_id", "headline", "content",
                   "sentiment_score", "ticker", "candle_time"]
    cols_sql = ", ".join(f"`{c}`" for c in insert_cols)
    vals_sql = ", ".join(f":{c}" for c in insert_cols)
    update_sql = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in insert_cols[1:])
    
    sql = text(f"""
        INSERT INTO market_sentiment ({cols_sql})
        VALUES ({vals_sql})
        ON DUPLICATE KEY UPDATE {update_sql};
    """)

    records = df[insert_cols].to_dict(orient="records")
    
    with engine.connect() as conn:
        try:
            with conn.begin():
                if records:
                    conn.execute(sql, records)
                    logger.info(f"Upserted {len(records)} rows into market_sentiment.")
        except Exception as e:
            logger.critical(f"CRITICAL ERROR upserting market_sentiment: {e}")
            raise

# -----------------------------
# MAIN
# -----------------------------
def main(db_url: Optional[str] = None):
    """Main function to run the CryptoPanic data ingestion pipeline."""
    logger.info("--- Starting Real-Time Ingestion Pipeline ---")
    try:
        engine = get_db_engine(db_url or DATABASE_URL)
        create_tables(engine)  
        
        logger.info(f"Starting CryptoPanic data fetch from START_DATE: {START_DATE}...")
        raw = fetch_from_cryptopanic()
        
        if not raw:
            logger.info("No records fetched from CryptoPanic. Exiting pipeline.")
            return
            
        df_sources, df_news = prepare_source_data(raw)
        
        logger.info("Step 1/4: Upserting news sources...")
        upsert_news_source(engine, df_sources)
        
        logger.info("Step 2/4: Fetching source map for ID resolution...")
        source_map = get_source_map_from_db(engine)
        
        logger.info("Step 3/4: Finalizing sentiment data (aligning time, mapping source IDs)...")
        df_sent = finalize_sentiment_data(df_news, source_map)
        
        logger.info("Step 4/4: Upserting market sentiment records...")
        upsert_market_sentiment(engine, df_sent)
        
        logger.info("--- Pipeline finished successfully. ---")
        
    except Exception as e:
        logger.critical(f"Pipeline failed due to a critical error: {e}", exc_info=True)


if __name__ == "__main__":
    main()