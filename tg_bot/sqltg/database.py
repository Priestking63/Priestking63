import sqlite3
import numpy as np
from contextlib import contextmanager
from typing import Optional, Generator, Any


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
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


def init_db() -> None:
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


def save_message(
    message_id: int, channel: str, date: str, text: str, embedding: Optional[Any] = None
) -> bool:
    try:
        with get_db_connection() as conn:
            embedding_blob = None
            if embedding is not None:
                if hasattr(embedding, "tobytes"):
                    embedding_blob = embedding.tobytes()
                elif isinstance(embedding, (list, np.ndarray)):
                    embedding_array = np.array(embedding, dtype=np.float32)
                    embedding_blob = embedding_array.tobytes()

            cursor = conn.execute(
                "INSERT OR IGNORE INTO skidki (message_id, channel, date, text, embedding) VALUES (?, ?, ?, ?, ?)",
                (message_id, channel, date, text, embedding_blob),
            )
            return cursor.rowcount > 0
    except Exception as e:
        print(f"Ошибка сохранения в БД: {e}")
        return False
