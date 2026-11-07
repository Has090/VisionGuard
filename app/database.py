import os
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Check if PostgreSQL environment variables are provided
DB_HOST = os.getenv("DB_HOST")
POSTGRES_DB = os.getenv("POSTGRES_DB", "visionguard_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "vg_admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "VisionGuardSecure2026!")
DB_PORT = os.getenv("DB_PORT", "5432")

USE_POSTGRES = bool(DB_HOST)

if USE_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor

# SQLite Fallback Path
SQLITE_PATH = Path("reports") / "visionguard.db"


def get_db_connection():
    if USE_POSTGRES:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        return conn
    else:
        SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if USE_POSTGRES:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp VARCHAR(64) NOT NULL,
                model_name VARCHAR(128) NOT NULL,
                filename VARCHAR(256) NOT NULL,
                prediction VARCHAR(64) NOT NULL,
                confidence REAL NOT NULL,
                raw_score REAL NOT NULL,
                threshold_applied REAL NOT NULL,
                latency_ms REAL NOT NULL,
                client_ip VARCHAR(64) NOT NULL
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                prediction_id INTEGER REFERENCES predictions(id) ON DELETE SET NULL,
                timestamp VARCHAR(64) NOT NULL,
                prediction VARCHAR(64) NOT NULL,
                user_verdict VARCHAR(32) NOT NULL CHECK(user_verdict IN ('correct', 'incorrect')),
                notes TEXT,
                client_ip VARCHAR(64) NOT NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_model ON predictions(model_name);")
    else:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                raw_score REAL NOT NULL,
                threshold_applied REAL NOT NULL,
                latency_ms REAL NOT NULL,
                client_ip TEXT NOT NULL
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                timestamp TEXT NOT NULL,
                prediction TEXT NOT NULL,
                user_verdict TEXT NOT NULL CHECK(user_verdict IN ('correct', 'incorrect')),
                notes TEXT,
                client_ip TEXT NOT NULL,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE SET NULL
            );
        """)
    
    conn.commit()
    cursor.close()
    conn.close()

def get_system_analytics() -> Dict[str, Any]:
    conn = get_db_connection()
    pred_query = """
        SELECT 
            COUNT(*) as total_predictions,
            COALESCE(AVG(latency_ms), 0.0) as avg_latency_ms,
            SUM(CASE WHEN prediction = 'AiArtData' THEN 1 ELSE 0 END) as ai_count,
            SUM(CASE WHEN prediction = 'RealArt' THEN 1 ELSE 0 END) as real_count
        FROM predictions;
    """
    fb_query = """
        SELECT 
            COUNT(*) as total_feedback,
            SUM(CASE WHEN user_verdict = 'correct' THEN 1 ELSE 0 END) as correct_count,
            SUM(CASE WHEN user_verdict = 'incorrect' THEN 1 ELSE 0 END) as incorrect_count
        FROM feedback;
    """
    
    if USE_POSTGRES:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(pred_query)
        pred_stats = dict(cursor.fetchone())
        cursor.execute(fb_query)
        fb_stats = dict(cursor.fetchone())
    else:
        cursor = conn.cursor()
        cursor.execute(pred_query)
        pred_stats = dict(cursor.fetchone())
        cursor.execute(fb_query)
        fb_stats = dict(cursor.fetchone())
        
    cursor.close()
    conn.close()

    total_fb = fb_stats["total_feedback"] or 0
    correct_fb = fb_stats["correct_count"] or 0
    feedback_accuracy = round((correct_fb / total_fb) * 100, 2) if total_fb > 0 else 0.0

    return {
        "total_predictions": pred_stats["total_predictions"] or 0,
        "ai_predictions": pred_stats["ai_count"] or 0,
        "real_predictions": pred_stats["real_count"] or 0,
        "avg_latency_ms": round(float(pred_stats["avg_latency_ms"]), 2),
        "total_feedback": total_fb,
        "feedback_accuracy_pct": feedback_accuracy
    }


def record_prediction(model_name, filename, prediction, confidence, raw_score, threshold, latency_ms, client_ip) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    now_ts = datetime.utcnow().isoformat() + "Z"
    
    if USE_POSTGRES:
        cursor.execute("""
            INSERT INTO predictions 
            (timestamp, model_name, filename, prediction, confidence, raw_score, threshold_applied, latency_ms, client_ip)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (now_ts, model_name, filename, prediction, confidence, raw_score, threshold, latency_ms, client_ip))
        pred_id = cursor.fetchone()[0]
    else:
        cursor.execute("""
            INSERT INTO predictions 
            (timestamp, model_name, filename, prediction, confidence, raw_score, threshold_applied, latency_ms, client_ip)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now_ts, model_name, filename, prediction, confidence, raw_score, threshold, latency_ms, client_ip))
        pred_id = cursor.lastrowid
        
    conn.commit()
    cursor.close()
    conn.close()
    return pred_id


def record_feedback(prediction_id, prediction, user_verdict, notes, client_ip) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    now_ts = datetime.utcnow().isoformat() + "Z"
    
    if USE_POSTGRES:
        cursor.execute("""
            INSERT INTO feedback 
            (prediction_id, timestamp, prediction, user_verdict, notes, client_ip)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (prediction_id, now_ts, prediction, user_verdict, notes, client_ip))
        fb_id = cursor.fetchone()[0]
    else:
        cursor.execute("""
            INSERT INTO feedback 
            (prediction_id, timestamp, prediction, user_verdict, notes, client_ip)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (prediction_id, now_ts, prediction, user_verdict, notes, client_ip))
        fb_id = cursor.lastrowid
        
    conn.commit()
    cursor.close()
    conn.close()
    return fb_id


def fetch_recent_history(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    query = """
        SELECT 
            p.id,
            p.timestamp,
            p.model_name,
            p.filename,
            p.prediction,
            p.confidence,
            p.raw_score,
            p.latency_ms,
            f.user_verdict,
            f.notes
        FROM predictions p
        LEFT JOIN feedback f ON p.id = f.prediction_id
        ORDER BY p.id DESC
        LIMIT %s;
    """ if USE_POSTGRES else """
        SELECT 
            p.id,
            p.timestamp,
            p.model_name,
            p.filename,
            p.prediction,
            p.confidence,
            p.raw_score,
            p.latency_ms,
            f.user_verdict,
            f.notes
        FROM predictions p
        LEFT JOIN feedback f ON p.id = f.prediction_id
        ORDER BY p.id DESC
        LIMIT ?;
    """
    
    if USE_POSTGRES:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
    else:
        cursor = conn.cursor()
        cursor.execute(query, (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        
    cursor.close()
    conn.close()
    return rows