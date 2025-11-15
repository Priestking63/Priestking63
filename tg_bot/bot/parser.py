import csv
import os
import sys
import re
from datetime import datetime
from telethon import TelegramClient
import logging
import asyncio
from telethon.tl.types import MessageService
import re
import html
from urllib.parse import unquote

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
logger = logging.getLogger(__name__)


from sqltg.database import init_db, save_message
from transform import TextEmbedder


api_id = 23801227
api_hash = "53c7949eaea365a820ec814c2971f87c"
channels = [
    "https://t.me/skidki_nnov_me08",
    "https://t.me/plohie_skidki",
    "https://t.me/besfree",
    "https://t.me/dealfinder",
    "https://t.me/+xypZbUFHvp84NDQ6",
    "https://t.me/poblat",
    "https://t.me/sliv_halyavy",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def clean_text(text):
    if text is None:
        return ""

    text = html.unescape(text)
    text = unquote(text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[*_~`#\[\]()]", "", text)  
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)  
    text = re.sub(r"[\U0001F600-\U0001F64F]", "", text) 
    text = re.sub(r"[\U0001F300-\U0001F5FF]", "", text)  
    text = re.sub(r"[\U0001F680-\U0001F6FF]", "", text) 
    text = re.sub(r"[\U0001F1E0-\U0001F1FF]", "", text) 
    text = re.sub(r"[\U00002700-\U000027BF]", "", text)  
    text = re.sub(r"[♥♦♣♠•◘○◙♂♀♪♫☼►◄↕‼¶§▬↨↑↓→←∟↔▲▼]", "", text)
    text = re.sub(r"([!?.,])\1+", r"\1", text)  # !!! -> !, ??? -> ?
    text = re.sub(r"[-—]{2,}", " ", text)
    text = re.sub(
        r"[^\w\s\.\,\!\?\:\;\-\+\(\)\"\'\@\%\=\/\\\&\#\<\>]", "", text
    )

    text = re.sub(r"(\d)[\s]*%", r"\1%", text) 
    text = re.sub(
        r"(\d)[\s]*-[\s]*(\d)", r"\1-\2", text
    ) 

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


async def parser_tgchanels(client, channels, limit=500):  
    """Парсит каналы и сохраняет в базу данных"""
    logger.info("Инициализация базы данных...")
    init_db()

    processed_messages = set()

    embedder = TextEmbedder()

    total_messages = 0
    skipped_messages = 0

    for channel in channels:
        logger.info(f"Парсинг канала {channel}")
        try:
            all_mes = await client.get_messages(channel, limit=limit)
            channel_message_count = 0

            for message in all_mes:
                if isinstance(message, MessageService):
                    skipped_messages += 1
                    continue

                if not message.text or len(message.text.strip()) <= 5:
                    skipped_messages += 1
                    continue

                message_uid = f"{channel}_{message.id}"

                if message_uid in processed_messages:
                    continue

                try:
                    cleaned_text = clean_text(message.text)
                    if len(cleaned_text) < 30:
                        continue

                    embeddings = embedder.get_embeddings(cleaned_text)
                    
                    if save_message(
                        message_id=message.id,
                        channel=channel,
                        date=str(message.date),
                        text=cleaned_text,
                        embedding= embeddings

                    ):
                        processed_messages.add(message_uid)
                        channel_message_count += 1
                        total_messages += 1
                        logger.info(f"Сохранено сообщение {message.id}")
                    else:
                        logger.warning(f"Не удалось сохранить сообщение {message_uid}")

                except Exception as e:
                    logger.error(
                        f"Ошибка при обработке сообщения {message_uid}: {str(e)}",
                        exc_info=True,
                    )

            logger.info(f"Успешно обработано {channel_message_count} сообщений")

        except Exception as e:
            logger.error(
                f"Ошибка при парсинге канала {channel}: {str(e)}", exc_info=True
            )

    logger.info(f"Всего обработано сообщений: {total_messages}")
    logger.info(f"Пропущено сообщений: {skipped_messages}")


async def main():
    try:
        logger.info("Запуск парсера...")
        async with TelegramClient(
            "my", api_id, api_hash, system_version="4.10.5 beta x64"
        ) as client:
            logger.info("Клиент Telegram успешно создан")
            await parser_tgchanels(client, channels)
        logger.info("Парсинг завершен")
    except Exception as e:
        logger.error(f"Ошибка в main: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
