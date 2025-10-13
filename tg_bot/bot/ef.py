# add_test_data.py
import sqlite3
from transform import TextEmbedder
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def add_test_messages():
    """Добавляет тестовые сообщения в базу данных"""

    embedder = TextEmbedder()
    conn = sqlite3.connect("bot_base.db")
    cursor = conn.cursor()

    # Тестовые сообщения о скидках и акциях
    test_messages = [
        {
            "message_id": 100001,
            "channel": "https://t.me/test_channel",
            "date": "2024-01-10 10:00:00",
            "text": "Большая распродажа! Скидки до 70% на всю одежду. Успей купить!",
        },
        {
            "message_id": 100002,
            "channel": "https://t.me/test_channel",
            "date": "2024-01-10 11:00:00",
            "text": "Специальная акция: вторая вещь со скидкой 50%. Только сегодня!",
        },
        {
            "message_id": 100003,
            "channel": "https://t.me/test_channel",
            "date": "2024-01-10 12:00:00",
            "text": "Новогодние скидки на все товары. Цены снижены на 30-60%.",
        },
        {
            "message_id": 100004,
            "channel": "https://t.me/test_channel",
            "date": "2024-01-10 13:00:00",
            "text": "Распродажа зимней коллекции. Футболки от 299 рублей.",
        },
        {
            "message_id": 100005,
            "channel": "https://t.me/test_channel",
            "date": "2024-01-10 14:00:00",
            "text": "Акция недели: покупайте 3 вещи по цене 2. Выгодные предложения!",
        },
    ]

    added_count = 0

    for msg in test_messages:
        try:
            # Создаем эмбединг для текста
            embedding = embedder.get_embeddings(msg["text"])

            # Добавляем в базу
            cursor.execute(
                "INSERT OR REPLACE INTO skidki (message_id, channel, date, text, embedding) VALUES (?, ?, ?, ?, ?)",
                (
                    msg["message_id"],
                    msg["channel"],
                    msg["date"],
                    msg["text"],
                    embedding,
                ),
            )
            added_count += 1
            print(f"✅ Добавлено тестовое сообщение: {msg['text'][:50]}...")

        except Exception as e:
            print(f"❌ Ошибка добавления сообщения: {e}")

    conn.commit()
    conn.close()
    print(f"\n📊 Добавлено тестовых сообщений: {added_count}")


if __name__ == "__main__":
    add_test_messages()
