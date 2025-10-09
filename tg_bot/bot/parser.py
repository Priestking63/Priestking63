import asyncio
import logging
import os
import sys
import re
from typing import Optional
from telethon import TelegramClient, events
from telethon.tl.types import MessageService

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from sqltg.database import init_db, save_message
from emb.transform import TextEmbedder
from config import Config

Config.validate()

channels = [
    "https://t.me/skidki_nnov_me08",
    "https://t.me/skidki_iz_pitera",
    "https://t.me/myfavoritejumoreski",
    "https://t.me/dvachannel"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class OnlineParser:
    def __init__(self):
        self.client: Optional[TelegramClient] = None  
        self.embedder = TextEmbedder()
        self.channel_dict = {}

    def clean_text(self, text: str) -> str:
        """Очищает текст от лишних символов"""
        if text is None:
            return ""
        text = re.sub(r"[^\w\s\.\,\!\?\:\;\-\+\(\)\[\]\{\}\"\'\@\$\%\=\/\\]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    async def start_parser(self) -> TelegramClient:
        """Запуск парсера в реальном времени"""
        init_db()

        self.client = TelegramClient("online_session", Config.API_ID, Config.API_HASH)

        await self.client.start()

        for channel_url in channels:
            try:
                entity = await self.client.get_entity(channel_url)
                self.channel_dict[entity.id] = channel_url
                logger.info(f"Канал {channel_url} имеет ID {entity.id}")
            except Exception as e:
                logger.error(f"Ошибка получения entity для {channel_url}: {e}")

        @self.client.on(events.NewMessage(chats=list(self.channel_dict.keys())))
        async def handler(event):
            message = event.message
            if isinstance(message, MessageService):
                return

            text = message.text
            if not text or len(text.strip()) <= 5:
                return

            cleaned_text = self.clean_text(text)

            try:
                embedding = self.embedder.get_embeddings(cleaned_text)
                channel_url = self.channel_dict[event.chat_id]

                if save_message(
                    message_id=message.id,
                    channel=channel_url,
                    date=str(message.date),
                    text=cleaned_text,
                    embedding=embedding,
                ):
                    logger.info(f"Сохранено сообщение {message.id} из {channel_url}")
                else:
                    logger.warning(f"Не удалось сохранить сообщение {message.id}")

            except Exception as e:
                logger.error(f"Ошибка обработки сообщения: {e}")

        logger.info("Парсер запущен и слушает сообщения в реальном времени...")
        return self.client

    async def stop_parser(self):
        """Остановка парсера"""
        if self.client:
            await self.client.disconnect()
            self.client = None
