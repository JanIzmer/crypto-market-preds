# IMPORTS
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from pathlib import Path

# DATABASE CONFIGURATION
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")        
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "")
DB_DRIVER = os.getenv("DB_DRIVER", "mysql+pymysql")

# SQLAlchemy URL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

# FUNCTION TO GET DB ENGINE
def get_db_engine(db_url: str = DATABASE_URL):
    """
    Creates and returns a SQLAlchemy engine instance using the defined database URL.
    
    Args:
        db_url: The connection string. Defaults to DATABASE_URL.
        
    Returns:
        SQLAlchemy Engine instance.
    """
    print(f"Connecting to database...")
    # pool_recycle is crucial for handling stale MySQL connections
    return create_engine(db_url, pool_recycle=3600)