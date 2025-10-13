# check_compatibility.py
import sqlite3
import numpy as np
from transform import TextEmbedder


def check_embedding_compatibility():
    """Проверяет совместимость эмбедингов из базы и от TextEmbedder"""

    # Создаем тестовый эмбединг
    embedder = TextEmbedder()
    test_text = "тестовый текст для проверки"
    test_blob = embedder.get_embeddings(test_text)

    print(f"✅ Тестовый BLOB от TextEmbedder: {len(test_blob)} байт")

    # Проверяем базу данных
    conn = sqlite3.connect("bot_base.db")
    cursor = conn.cursor()

    cursor.execute("SELECT embedding FROM skidki WHERE embedding IS NOT NULL LIMIT 1")
    result = cursor.fetchone()

    if result:
        db_blob = result[0]
        print(f"✅ BLOB из базы данных: {len(db_blob)} байт")

        # Проверяем можно ли преобразовать оба в numpy arrays
        try:
            test_array = np.frombuffer(test_blob, dtype=np.float32)
            db_array = np.frombuffer(db_blob, dtype=np.float32)

            print(f"✅ Тестовый массив: {test_array.shape}")
            print(f"✅ Массив из БД: {db_array.shape}")

            # Проверяем косинусное сходство
            dot_product = np.dot(test_array, db_array)
            norm1 = np.linalg.norm(test_array)
            norm2 = np.linalg.norm(db_array)

            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
                print(f"✅ Косинусное сходство: {similarity:.4f}")
            else:
                print("❌ Нулевые векторы")

        except Exception as e:
            print(f"❌ Ошибка преобразования: {e}")
    else:
        print("❌ В базе нет сообщений с эмбедингами")

    conn.close()


if __name__ == "__main__":
    check_embedding_compatibility()
