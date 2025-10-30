# IMPORTS
import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, Engine
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Assuming db_config.py is in the core structure
from db_config import DATABASE_URL, get_db_engine 

# Configure logging for better output control
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIGURATION
# HF model to use: FinBERT-CRYPTO 
HF_MODEL = os.getenv("HF_MODEL", "burakutf/finetuned-finbert-crypto")

# Batch size for DB fetch (how many records to process per main loop)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
# Batch size for Hugging Face inference (smaller for VRAM efficiency)
HF_BATCH = int(os.getenv("HF_BATCH", "16"))

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FinBERT-CRYPTO label map
FINBERT_LABELS = ["negative", "neutral", "positive"]

# --- Initialization ---
logger.info(f"Connecting to DB: {DATABASE_URL}")
# Use get_db_engine() if available, otherwise create directly
try:
    engine: Engine = get_db_engine()
except NameError:
    engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True)

logger.info(f"Loading tokenizer & model: {HF_MODEL} on device: {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL).to(DEVICE)
model.eval()

# DATABASE HELPERS

def ensure_finbert_columns_exist(db_engine: Engine) -> None:
    """
    Ensures that the FinBERT scoring columns exist in the market_sentiment table.
    Handles MySQL versions that might not support `IF NOT EXISTS` in ALTER TABLE.
    
    Args:
        db_engine: SQLAlchemy Engine object for database connection.
    """
    logger.info("Ensuring FinBERT columns exist in market_sentiment...")
    
    # Attempt combined ALTER command first (supported by most modern SQL DBs)
    alter_sql = """
    ALTER TABLE market_sentiment
      ADD COLUMN IF NOT EXISTS finbert_label VARCHAR(32) NULL,
      ADD COLUMN IF NOT EXISTS finbert_score FLOAT NULL,
      ADD COLUMN IF NOT EXISTS finbert_probs JSON NULL,
      ADD COLUMN IF NOT EXISTS finbert_updated_at DATETIME NULL;
    """
    try:
        with db_engine.connect() as conn:
            conn.execute(text(alter_sql))
            conn.commit()
    except Exception as e:
        logger.warning(f"NOTICE: Combined ALTER with IF NOT EXISTS failed (error: {e}). Trying per-column add...")
        
        # Fallback for older MySQL versions: attempt to add columns one by one
        cols = [
            ("finbert_label", "VARCHAR(32) NULL"),
            ("finbert_score", "FLOAT NULL"),
            ("finbert_probs", "JSON NULL"),
            ("finbert_updated_at", "DATETIME NULL")
        ]
        with db_engine.connect() as conn:
            for name, ddl in cols:
                try:
                    conn.execute(text(f"ALTER TABLE market_sentiment ADD COLUMN {name} {ddl};"))
                    conn.commit()
                except Exception:
                    # Ignore error if column already exists
                    pass
    logger.info("FinBERT columns verified.")

def fetch_unsored_batch(db_engine: Engine, limit: int, offset: int = 0) -> pd.DataFrame:
    """
    Fetches a batch of unscored news articles for 'BTCUSDT' from the database.
    
    Args:
        db_engine: SQLAlchemy Engine object.
        limit: The maximum number of rows to return (BATCH_SIZE).
        offset: The starting point for the query.
        
    Returns:
        A pandas DataFrame containing 'id' and 'content'.
    """
    query = text("""
        SELECT id, content
        FROM market_sentiment
        WHERE content IS NOT NULL
          AND finbert_score IS NULL
          AND ticker IS NOT NULL
          AND UPPER(ticker) = 'BTCUSDT'
        ORDER BY publication_time ASC
        LIMIT :limit OFFSET :offset;
    """)
    with db_engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"limit": limit, "offset": offset})
    return df

def update_scored_rows(db_engine: Engine, updates: List[Dict[str, Any]]) -> None:
    """
    Updates the market_sentiment table with the calculated FinBERT scores and labels.
    
    Args:
        db_engine: SQLAlchemy Engine object.
        updates: A list of dictionaries, each containing update fields and 'id'.
    """
    sql = text("""
        UPDATE market_sentiment
        SET finbert_label = :label,
            finbert_score = :score,
            finbert_probs = :probs,
            finbert_updated_at = :updated
        WHERE id = :id
    """)
    
    # Prepare parameters for bulk execution
    params = []
    for u in updates:
        params.append({
            "label": u["label"],
            "score": u["score"],
            "probs": json.dumps(u["probs"], ensure_ascii=False),
            "updated": u["updated"],
            "id": u["id"]
        })
        
    with db_engine.connect() as conn:
        # Use a transaction for atomic batch update
        with conn.begin():
            conn.execute(sql, params)
    logger.info(f"Successfully updated {len(params)} rows.")


# MODEL INFERENCE HELPERS

def hf_batch_infer(texts: List[str], model, tokenizer, device: str, batch_size: int = HF_BATCH) -> List[Dict[str, Any]]:
    """
    Performs batch inference using the loaded Hugging Face model.
    
    Args:
        texts: A list of news article content strings.
        model: The loaded FinBERT model.
        tokenizer: The loaded FinBERT tokenizer.
        device: The compute device ('cuda' or 'cpu').
        batch_size: Internal batch size for inference.
        
    Returns:
        A list of dictionaries with 'label', 'score', and 'probs' for each text.
    """
    results: List[Dict[str, Any]] = []
    
    for i in range(0, len(texts), batch_size):
        sub_texts = texts[i:i + batch_size]
        
        # Tokenization
        enc = tokenizer(
            sub_texts, 
            truncation=True, 
            padding=True, 
            return_tensors="pt", 
            max_length=512
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        
        # Inference
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits.cpu().numpy()
            
        for logit in logits:
            # Softmax calculation
            exp = np.exp(logit - np.max(logit))
            probs = (exp / exp.sum()).tolist()
            
            # FinBERT-CRYPTO uses [negative, neutral, positive] mapping
            labels = FINBERT_LABELS
            label_idx = int(np.argmax(probs))
            label = labels[label_idx]
            
            # FinBERT Score: positive probability minus negative probability
            neg_prob = probs[0]
            pos_prob = probs[2]
            score = float(pos_prob - neg_prob)
            
            results.append({"label": label, "score": score, "probs": probs})
            
    return results

def get_total_unsored_count(db_engine: Engine) -> int:
    """
    Queries the database for the total number of news articles pending FinBERT scoring.
    """
    query = text("""
        SELECT COUNT(*) FROM market_sentiment
        WHERE content IS NOT NULL
          AND finbert_score IS NULL
          AND ticker IS NOT NULL
          AND UPPER(ticker) = 'BTCUSDT';
    """)
    with db_engine.connect() as conn:
        total = conn.execute(query).scalar_one()
    return int(total)

# MAIN PROCESSING

def main() -> None:
    """
    Main function to run the FinBERT scoring process in batches.
    """
    ensure_finbert_columns_exist(engine)
    
    total_to_process = get_total_unsored_count(engine)
    
    logger.info(f"Total articles to process: {total_to_process}")
    if total_to_process == 0:
        logger.info("Nothing to do. Exiting.")
        return

    processed_count = 0
    
    # We use offset=0 and rely on the UPDATE statement setting finbert_score IS NOT NULL 
    # to make the next batch query start from the remaining unscored records.
    while processed_count < total_to_process:
        # Fetch the next batch from the beginning of the unscored list
        df = fetch_unsored_batch(db_engine=engine, limit=BATCH_SIZE, offset=0)
        
        if df.empty:
            # This should only happen if total_to_process count was inaccurate or a concurrent process ran
            break

        texts = df["content"].astype(str).tolist()
        ids = df["id"].tolist()
        
        logger.info(f"Processing batch of {len(texts)} articles...")

        hf_results = hf_batch_infer(
            texts, 
            model=model, 
            tokenizer=tokenizer, 
            device=DEVICE, 
            batch_size=HF_BATCH
        )

        # Prepare update payload
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        updates = []
        for idx, article_id in enumerate(ids):
            result = hf_results[idx]
            updates.append({
                "id": article_id,
                "label": result["label"],
                "score": result["score"],
                "probs": result["probs"],
                "updated": now
            })

        update_rows(db_engine=engine, updates=updates)
        
        processed_count += len(updates)
        logger.info(f"Batch completed. Total processed: {processed_count}/{total_to_process}")

    logger.info("Scoring complete. Total processed: %d", processed_count)

if __name__ == "__main__":
    main()