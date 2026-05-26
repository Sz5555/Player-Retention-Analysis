import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "game_analytics.db"


def init_db():
    """Create SQLite database from CSV files. Returns (conn, table_info)."""
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)

    # --- player_data table ---
    processed_csv = db_path.parent / "processed" / "player_data_processed.csv"
    if processed_csv.exists():
        df_players = pd.read_csv(processed_csv)
        df_players.to_sql("player_data", conn, if_exists="replace", index=False)

    # --- player_reviews table ---
    reviews_csv = db_path.parent / "processed" / "reviews.csv"
    if reviews_csv.exists():
        df_reviews = pd.read_csv(reviews_csv)
        df_reviews.to_sql("player_reviews", conn, if_exists="replace", index=False)

    return conn


def get_schema(conn):
    """Return table and column info for display."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    info = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        info[table] = [(row[1], row[2]) for row in cursor.fetchall()]
    return info


def run_query(conn, sql):
    """Execute a SELECT-only query. Returns (success, data/error_message, columns)."""
    cleaned = sql.strip().rstrip(";")

    dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
    for word in dangerous:
        if word in cleaned.upper().split():
            return False, f"只允许 SELECT 查询，不允许 {word}", []

    try:
        df = pd.read_sql_query(cleaned, conn)
        return True, df, list(df.columns)
    except Exception as e:
        return False, str(e), []
