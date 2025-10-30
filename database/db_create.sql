CREATE DATABASE IF NOT EXISTS market_db
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;
USE market_db;

-- ===========================================
-- Drop tables in reverse order (to avoid FK constraint issues)
-- ===========================================
DROP TABLE IF EXISTS market_sentiment;
DROP TABLE IF EXISTS technical_indicators;
DROP TABLE IF EXISTS news_source;
DROP TABLE IF EXISTS kline_data;

-- ===========================================
-- 1. Table: kline_data
-- ===========================================
CREATE TABLE IF NOT EXISTS kline_data (
    ticker VARCHAR(10) NOT NULL DEFAULT 'BTCUSDT',
    candle_time DATETIME NOT NULL,
    open DECIMAL(18, 8) NULL,
    high DECIMAL(18, 8) NULL,
    low DECIMAL(18, 8) NULL,
    close DECIMAL(18, 8) NULL,
    volume BIGINT NULL,
    
    PRIMARY KEY (ticker, candle_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===========================================
-- 2. Table: technical_indicators
--    1:1 relation with kline_data (ticker + candle_time)
-- ===========================================
CREATE TABLE IF NOT EXISTS technical_indicators (
    ticker VARCHAR(10) NOT NULL DEFAULT 'BTCUSDT',
    candle_time DATETIME NOT NULL,
    
    -- Numeric feature for volatility/risk
    atr14 DECIMAL(18, 8) NULL,
    
    -- 11 Core Binary Indicators
    ema12_cross_ema26_up TINYINT(1) NOT NULL DEFAULT 0,
    ema12_cross_ema26_down TINYINT(1) NOT NULL DEFAULT 0,
    close_cross_sma50_up TINYINT(1) NOT NULL DEFAULT 0,
    close_cross_sma50_down TINYINT(1) NOT NULL DEFAULT 0,
    macd_cross_signal_up TINYINT(1) NOT NULL DEFAULT 0,
    macd_cross_signal_down TINYINT(1) NOT NULL DEFAULT 0,
    rsi_overbought TINYINT(1) NOT NULL DEFAULT 0,
    rsi_oversold TINYINT(1) NOT NULL DEFAULT 0,
    close_cross_upper_bb TINYINT(1) NOT NULL DEFAULT 0,
    close_cross_lower_bb TINYINT(1) NOT NULL DEFAULT 0,
    strong_trend TINYINT(1) NOT NULL DEFAULT 0,

    -- Primary and Foreign Keys
    PRIMARY KEY (ticker, candle_time),
    FOREIGN KEY (ticker, candle_time) 
        REFERENCES kline_data(ticker, candle_time) 
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===========================================
-- 3. Table: news_source
-- ===========================================
CREATE TABLE IF NOT EXISTS news_source (
    source_id INT NOT NULL AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    url_base VARCHAR(255) DEFAULT NULL,
    PRIMARY KEY (source_id),
    UNIQUE KEY uq_source_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===========================================
-- 4. Table: market_sentiment
-- ===========================================
CREATE TABLE IF NOT EXISTS market_sentiment (
    id INT NOT NULL AUTO_INCREMENT,
    publication_time DATETIME NOT NULL,
    source_id INT NOT NULL,
    headline VARCHAR(500) NOT NULL,
    content TEXT,
    sentiment_score DECIMAL(5,4) NOT NULL,
    
    -- Ticker must allow NULL to match ON DELETE SET NULL
    ticker VARCHAR(10) DEFAULT NULL, 
    
    -- candle_time column already allows NULL
    candle_time DATETIME DEFAULT NULL, 
    
    PRIMARY KEY (id),
    KEY idx_sentiment_time (publication_time),
    KEY idx_sentiment_source (source_id),
    
    CONSTRAINT fk_sentiment_source
        FOREIGN KEY (source_id) REFERENCES news_source(source_id)
        ON DELETE RESTRICT
        ON UPDATE CASCADE,
    
    CONSTRAINT fk_sentiment_kline
        FOREIGN KEY (ticker, candle_time) REFERENCES kline_data(ticker, candle_time)
        -- Requires 'ticker' and 'candle_time' columns in this table to be nullable
        ON DELETE SET NULL 
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===========================================
-- Optional: additional indexes/optimizations
-- ===========================================

-- 1) For frequent queries by time range and source
CREATE INDEX idx_sentiment_time_source ON market_sentiment (publication_time, source_id);

-- 2) For frequent joins between kline_data and indicators
CREATE INDEX idx_tech_kline_ts ON technical_indicators (ticker, candle_time);

-- Indicate completion
SELECT 'Schema created' AS message;
