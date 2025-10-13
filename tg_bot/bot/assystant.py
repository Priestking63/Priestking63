import sqlite3 
import numpy as np
import logging
from typing import List, Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transform import TextEmbedder

logging.basicConfig(level= logging.INFO)
logger = logging.getLogger(__name__)

class SmartTelegramAssistant():
    def __init__(self, db = 'bot_base.db', similarity_threshold = 0.7):
        self.db_path = db
        self.threshold = similarity_threshold

        self.embedder = TextEmbedder()

        self.generator = pipeline(
            "text2text-generation",
            model="IlyaGusev/rut5_base_sum_gazeta",
            tokenizer="IlyaGusev/rut5_base_sum_gazeta",
            max_length=200,
        )
        self.init_db()
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='skidki'
        """
        )

    def cosine_similarity(self, vec1: bytes, vec2: bytes):
        vec1 = np.frombuffer(vec1, dtype=np.float32)
        vec2 = np.frombuffer(vec2, dtype=np.float32)
        dot_product = vec1@vec2
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        sim = float(dot_product/(norm1*norm2))
        return max(0.0, min(1.0, sim))

    def search_similar_mes(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding_blob = self.embedder.get_embeddings(query)
        if query_embedding_blob is None:
            logger.error("Не удалось создать эмбединг для запроса")
            return []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT message_id, channel, date, text, embedding FROM skidki WHERE embedding IS NOT NULL")
        message = cursor.fetchall()
        conn.close()

        similarities = []
        for mes in message:
            message_id, channel, date, text, embedding = mes
            if not embedding:
                continue
            score = self.cosine_similarity(query_embedding_blob, embedding)
            if score > self.threshold:
                similarities.append({'message_id': message_id,
                                     'channel':channel,
                                     'date': date,
                                     'text': text,
                                     'similarity': score})

        similarities.sort(key = lambda x: x["similarity"], reverse=True)

        return similarities[:top_k]

    def generate_ai_response(self, query: str, context_message: List[Dict[str, Any]]):
        if not context_message:
            return "К сожалению, я не нашел подходящей информации в базе данных."

        if self.generator is None:
            return self._generate_simple_response(context_message)

        context = " | ".join([msg["text"] for msg in context_message[:3]])
        prompt = f"Вопрос: {query} Контекст: {context} Ответ:"

        return self.generator(prompt, max_length = 150, do_sample = True, temperature = 0.7)

    def _generate_simple_response(self, messages: List[Dict[str, Any]]) -> str:
        best_match = messages[0] if messages else None
        if best_match:
            return f"По вашему запросу я нашел:\n\n{best_match['text']}\n\nИсточник: {best_match['channel']}"
        return "Информация по вашему запросу не найдена."

    def format_similar_messages_response(self, similar_messages: List[Dict[str, Any]]) -> str:
        """Форматирует найденные похожие сообщения в читаемый текст"""
        if not similar_messages:
            return "❌ По вашему запросу ничего не найдено."

        response = "🔍 **Найденные похожие предложения:**\n\n"

        for i, msg in enumerate(similar_messages, 1):
            response += f"**{i}. Канал:** {msg['channel']}\n"
            response += f"**Дата:** {msg['date']}\n"
            response += f"**Сходство:** {msg['similarity']:.2%}\n"
            response += f"**Текст:** {msg['text']}\n"
            response += "─" * 50 + "\n\n"

        return response

    def response(self, query: str, top_k: int = 5, use_ai: bool = False) -> str:
        """Основной метод для получения ответа на запрос"""
        logger.info(f"Обработка запроса: {query}")

        similar_messages = self.search_similar_mes(query, top_k)

        if use_ai and self.generator and similar_messages:
            context = " | ".join([msg["text"] for msg in similar_messages[:3]])
            prompt = (
                f"На основе следующей информации: {context} Ответь на вопрос: {query}"
            )

            try:
                ai_response = self.generator(
                    prompt, max_length=150, do_sample=True, temperature=0.7
                )
                if ai_response and len(ai_response) > 0:
                    generated_text = ai_response[0]["generated_text"]

                    response = "🤖 **AI-ответ на основе найденной информации:**\n\n"
                    response += f"{generated_text}\n\n"
                    response += "─" * 50 + "\n\n"
                    response += self.format_similar_messages_response(similar_messages)
                    return response
            except Exception as e:
                logger.error(f"Ошибка генерации AI-ответа: {e}")
                return self.format_similar_messages_response(similar_messages)

        return self.format_similar_messages_response(similar_messages)


if __name__ == "__main__":
    assistant = SmartTelegramAssistant()

    print("🤖 Ассистент запущен! Задавайте вопросы о скидках и акциях.")
    print("Доступные команды:")
    print("- 'ai on' - включить AI-генерацию ответов")
    print("- 'ai off' - выключить AI-генерацию ответов")
    print("- 'exit' - выход\n")

    use_ai = False

    while True:
        try:
            query = input("\n🧐 Ваш вопрос: ").strip()

            if query.lower() in ["exit", "выход", "quit"]:
                break
            elif query.lower() == "ai on":
                use_ai = True
                print("✅ AI-генерация ответов включена")
                continue
            elif query.lower() == "ai off":
                use_ai = False
                print("❌ AI-генерация ответов выключена")
                continue

            if query:
                response = assistant.response(query, use_ai=use_ai)
                print(f"\n{response}")

        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Произошла ошибка: {e}")
