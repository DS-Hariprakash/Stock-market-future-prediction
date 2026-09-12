"""SQLite-backed run history log."""

import sqlite3
from datetime import datetime, timezone

import pandas as pd

DB_PATH = "run_history.db"


def init_db(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            best_model TEXT NOT NULL,
            rmse REAL NOT NULL,
            mae REAL NOT NULL,
            mape REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def log_run(ticker: str, best_model: str, rmse: float, mae: float, mape: float,
            db_path: str = DB_PATH) -> None:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO run_history (timestamp, ticker, best_model, rmse, mae, mape) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), ticker, best_model,
         rmse, mae, mape),
    )
    conn.commit()
    conn.close()


def fetch_history(db_path: str = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT timestamp, ticker, best_model, rmse, mae, mape "
        "FROM run_history ORDER BY id DESC", conn,
    )
    conn.close()
    return df


def clear_history(db_path: str = DB_PATH) -> None:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM run_history")
    conn.commit()
    conn.close()
