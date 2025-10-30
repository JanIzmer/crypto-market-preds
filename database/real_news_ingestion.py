# IMPORTS
import json
import re
import uuid
import logging
import sys
import os
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import Engine

# Assuming these helpers are available in a module named real_time_update
from real_time_update import (
    to_naive_utc, 
    sentiment_score_vader, 
    prepare_source_data, 
    finalize_sentiment_data, 
    upsert_news_source, 
    upsert_market_sentiment, 
    get_source_map_from_db
)
from db_config import DATABASE_URL, get_db_engine 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _extract_url_from_field(url_field: Optional[str]) -> str:
    """
    Extracts a clean URL string from a potentially noisy field that might contain 
    extra text before the actual 'http' or 'https' start of the URL.

    Args:
        url_field: The raw string field containing the URL.

    Returns:
        The cleaned URL string, or an empty string if no valid URL is found.
    """
    if not isinstance(url_field, str) or not url_field:
        return ""
    # Find the last occurrence of "http" or "https" to handle cases where 
    # the URL is appended to other text or metadata.
    idx_http = url_field.rfind("http")
    
    # If "http" is found, return the substring starting from that index.
    if idx_http != -1:
        return url_field[idx_http:].strip()
    
    # Otherwise, return the field content (or the original field if it doesn't look like a URL)
    return url_field.strip()


def ingest_csv_to_db(csv_path: str, engine: Engine, save_raw_json: bool = True) -> None:
    """
    Reads a CSV file, processes news article data, computes VADER sentiment, 
    and ingests the structured data into the 'news_source' and 'market_sentiment' 
    tables in the database.

    The script makes educated guesses about column names based on common headers 
    (date, title, text, source, url).

    Args:
        csv_path: The file path to the CSV file
        engine: The SQLAlchemy Engine object connected to the database.
        save_raw_json: Flag (currently unused but kept for potential future expansion) 

    Raises:
        ValueError: If essential columns (date, title, text, source) are missing 
                    from the CSV file.
    """
    logger.info(f"Starting ingestion process for CSV: {csv_path}")

    try:
        # Attempt to read CSV, using `dtype=str` to prevent Pandas inferring incorrect types
        df = pd.read_csv(csv_path, dtype=str)
    except FileNotFoundError:
        logger.error(f"Error: CSV file not found at path: {csv_path}")
        return
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return

    # Normalize column names to lowercase for case-insensitive lookup
    df.columns = [c.strip().lower() for c in df.columns]

    # COLUMN MAPPING
    col_map = {
        "date": ["date", "datetime", "publication_time", "pub_date"],
        "title": ["title", "headline", "subject"],
        "text": ["text", "content", "body", "description"],
        "source": ["source", "source_name"],
        "url": ["url", "link", "source_url"],
        "original_sentiment": ["sentiment"]
    }

    # Find the actual column names in the DataFrame
    found_cols: Dict[str, Optional[str]] = {}
    
    for key, aliases in col_map.items():
        found = next((c for c in df.columns if c in aliases), None)
        found_cols[key] = found

    # Check for required columns
    required_keys = ["date", "title", "text", "source"]
    missing_cols = [k for k in required_keys if found_cols[k] is None]
    
    if missing_cols:
        raise ValueError(f"CSV missing one or more required columns (aliases checked): {', '.join(missing_cols)}")

    # DATA PROCESSING
    raw_items: List[Dict[str, Any]] = []
    skipped_count = 0
    
    for _, row in df.iterrows():
        # 1. Date Parsing
        raw_date = row.get(found_cols["date"], None)
        pub_time: Optional[datetime] = None
        
        if pd.isna(raw_date) or raw_date is None:
            # Fallback to current time if date is missing
            pub_time = datetime.now(timezone.utc).replace(tzinfo=None)
            logger.debug(f"Row {_+1}: Missing date, defaulting to current UTC time.")
        else:
            try:
                # Try custom function first
                pub_time = to_naive_utc(raw_date)
            except Exception:
                try:
                    # Fallback to pandas robust parsing
                    pd_dt = pd.to_datetime(raw_date, utc=True)
                    pub_time = pd_dt.to_pydatetime().astimezone(timezone.utc).replace(tzinfo=None)
                except Exception as e:
                    logger.warning(f"Row {_+1}: Failed to parse date '{raw_date}'. Skipping row. Error: {e}")
                    skipped_count += 1
                    continue

        # 2. Extract Core Content
        title = str(row.get(found_cols["title"], "") or "").strip()
        content = str(row.get(found_cols["text"], "") or "").strip()

        if not title and not content:
            logger.warning(f"Row {_+1}: Both title and content are empty. Skipping row.")
            skipped_count += 1
            continue
            
        # 3. Compute VADER Sentiment
        vader_score = sentiment_score_vader(f"{title} {content}")

        # 4. Source and URL
        source_name = str(row.get(found_cols["source"], "") or "unknown").strip()
        url_field = row.get(found_cols["url"], "") if found_cols["url"] else ""
        source_url = _extract_url_from_field(str(url_field or ""))

        # 5. Optional Metadata
        original_sentiment = row.get(found_cols["original_sentiment"]) if found_cols["original_sentiment"] else None

        # 6. Build raw item dict
        raw_items.append({
            "id": f"csv_{uuid.uuid4().hex}", # Unique ID for this record
            "publication_time": pub_time,
            "source_name": source_name,
            "source_url": source_url,
            "headline": title,
            "content": content,
            "sentiment_score": vader_score,
            # Set ticker None by default to avoid ForeignKey constraints when importing generic data
            "ticker": None,
            "original_sentiment": original_sentiment
        })

    logger.info(f"Successfully processed {len(raw_items)} valid rows. Skipped {skipped_count} invalid rows.")
    
    if not raw_items:
        logger.info("No valid records to ingest into the database. Exiting.")
        return

    # --- Database Operations ---
    
    # 1. Prepare source and news DataFrames using external helper
    df_sources, df_news = prepare_source_data(raw_items)

    # 2. Upsert (Insert or Update) news sources
    logger.info(f"Upserting {len(df_sources)} unique news sources from CSV...")
    upsert_news_source(engine, df_sources)

    # 3. Fetch the updated source map (name -> id)
    source_map = get_source_map_from_db(engine)

    # 4. Finalize sentiment data (maps source name to ID)
    df_sent = finalize_sentiment_data(df_news, source_map)

    # 5. Ensure ticker column is properly set to None
    if "ticker" not in df_sent.columns:
        df_sent["ticker"] = None
    else:
        # Normalize empty/unknown tickers to None (required for the FK constraint setup)
        df_sent["ticker"] = df_sent["ticker"].replace({"": None, "unknown": None})

    # 6. Upsert market_sentiment records
    logger.info(f"Upserting {len(df_sent)} market sentiment records...")
    upsert_market_sentiment(engine, df_sent)

    logger.info(f"Ingestion complete. Total {len(df_sent)} records successfully loaded into DB.")

# MAIN EXECUTION
if __name__ == "__main__":
    try:
        # Initialize database engine
        engine = get_db_engine(DATABASE_URL)
        # Use a placeholder path for demonstration, assuming the user will correct it
        ingest_csv_to_db("notebooks/data/training_news.csv", engine)
    except Exception as e:
        logger.error(f"A critical error occurred during the CSV ingestion process: {e}", exc_info=True)