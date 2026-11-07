import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_DIR = Path("reports")
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "visionguard.db"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Predictions table fulfilling: What, When, By Which Model, and How Confidently
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

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_verdict ON predictions(prediction);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_model ON predictions(model_name);")
        conn.commit()


def record_prediction(
    model_name: str,
    filename: str,
    prediction: str,
    confidence: float,
    raw_score: float,
    threshold: float,
    latency_ms: float,
    client_ip: str
) -> int:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions 
            (timestamp, model_name, filename, prediction, confidence, raw_score, threshold_applied, latency_ms, client_ip)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat() + "Z",
            model_name,
            filename,
            prediction,
            confidence,
            raw_score,
            threshold,
            latency_ms,
            client_ip
        ))
        conn.commit()
        return cursor.lastrowid


def record_feedback(
    prediction_id: Optional[int],
    prediction: str,
    user_verdict: str,
    notes: str,
    client_ip: str
) -> int:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback 
            (prediction_id, timestamp, prediction, user_verdict, notes, client_ip)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            prediction_id,
            datetime.utcnow().isoformat() + "Z",
            prediction,
            user_verdict,
            notes,
            client_ip
        ))
        conn.commit()
        return cursor.lastrowid


def fetch_recent_history(limit: int = 50) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
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
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


def get_system_analytics() -> Dict[str, Any]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                COALESCE(AVG(latency_ms), 0.0) as avg_latency_ms,
                SUM(CASE WHEN prediction = 'AiArtData' THEN 1 ELSE 0 END) as ai_count,
                SUM(CASE WHEN prediction = 'RealArt' THEN 1 ELSE 0 END) as real_count
            FROM predictions;
        """)
        pred_stats = dict(cursor.fetchone())

        cursor.execute("""
            SELECT 
                COUNT(*) as total_feedback,
                SUM(CASE WHEN user_verdict = 'correct' THEN 1 ELSE 0 END) as correct_count,
                SUM(CASE WHEN user_verdict = 'incorrect' THEN 1 ELSE 0 END) as incorrect_count
            FROM feedback;
        """)
        fb_stats = dict(cursor.fetchone())

        total_fb = fb_stats["total_feedback"]
        feedback_accuracy = round((fb_stats["correct_count"] / total_fb) * 100, 2) if total_fb > 0 else 0.0

        return {
            "total_predictions": pred_stats["total_predictions"],
            "ai_predictions": pred_stats["ai_count"] or 0,
            "real_predictions": pred_stats["real_count"] or 0,
            "avg_latency_ms": round(pred_stats["avg_latency_ms"], 2),
            "total_feedback": total_fb,
            "feedback_accuracy_pct": feedback_accuracy
        }