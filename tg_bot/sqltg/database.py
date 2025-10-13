import sqlite3
import numpy as np
from contextlib import contextmanager


@contextmanager
def get_db_connection():
    conn = sqlite3.connect("bot_base.db")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db_connection() as conn:
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS skidki(
            message_id INTEGER,
            channel TEXT NOT NULL,
            date TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB,
            PRIMARY KEY (message_id, channel)
        )
        """
        )


def save_message(message_id, channel, date, text, embedding):
    """Сохраняет сообщение в базу данных"""
    try:
        with get_db_connection() as conn:

            cursor = conn.execute(
                "INSERT OR REPLACE INTO skidki (message_id, channel, date, text, embedding) VALUES (?, ?, ?, ?, ?)",
                (message_id, channel, date, text, embedding),
            )
            return cursor.rowcount > 0
    except Exception as e:
        print(f"Ошибка сохранения в БД: {e}")
        return False
