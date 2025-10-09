import asyncio
import logging
import signal
import sys
from typing import Any
import os

import uvicorn
from parser import OnlineParser
from api import app
from config import Config

import uvicorn
from api import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class Application:
    def __init__(self):
        self.parser = OnlineParser()
        self.server = None

    async def start_parser_only(self):
        """Запуск только парсера (без API сервера)"""
        try:
            logger.info("Запуск парсера...")
            await self.parser.start_parser()

            # Бесконечный цикл чтобы парсер продолжал работать
            logger.info("Парсер запущен. Нажмите Ctrl+C для остановки.")
            while True:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки парсера...")
        except Exception as e:
            logger.error(f"Ошибка в парсере: {e}")
        finally:
            await self.shutdown()

    async def start_api_only(self):
        """Запуск только API сервера (без парсера)"""
        try:
            config_uvicorn = uvicorn.Config(
                app,
                host=Config.HOST,
                port=Config.PORT,
                log_level="info",
                access_log=True,
            )
            self.server = uvicorn.Server(config_uvicorn)

            logger.info(
                f"Запуск API сервера на {Config.HOST}:{Config.PORT}"
            )
            logger.info("Нажмите Ctrl+C для остановки")

            await self.server.serve()

        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки API сервера...")
        except Exception as e:
            logger.error(f"Ошибка в API сервере: {e}")
        finally:
            await self.shutdown()

    async def start_both(self):
        """Запуск и парсера и API сервера"""
        try:
            # Запускаем парсер в фоне
            parser_task = asyncio.create_task(self.parser.start_parser())

            # Даем время парсеру запуститься
            await asyncio.sleep(2)

            # Запускаем API сервер
            config_uvicorn = uvicorn.Config(
                app,
                host=Config.HOST,
                port=Config.PORT,
                log_level="info",
                access_log=True,
            )
            self.server = uvicorn.Server(config_uvicorn)

            logger.info(
                f"Запуск API сервера на {Config.HOST}:{Config.PORT}"
            )
            logger.info("Парсер и API сервер запущены. Нажмите Ctrl+C для остановки.")

            # Запускаем сервер - он будет работать до прерывания
            await self.server.serve()

        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки...")
        except Exception as e:
            logger.error(f"Ошибка: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Корректное завершение работы"""
        logger.info("Завершение работы приложения...")
        try:
            await self.parser.stop_parser()
        except Exception as e:
            logger.error(f"Ошибка при остановке парсера: {e}")
        logger.info("Приложение остановлено")


async def main():
    """Главная функция с выбором режима работы"""
    print("\n" + "=" * 50)
    print("Telegram Parser & API Server")
    print("=" * 50)
    print("1. Запустить только парсер")
    print("2. Запустить только API сервер")
    print("3. Запустить парсер и API сервер вместе")
    print("4. Выход")
    print("=" * 50)

    choice = input("Выберите режим (1-4): ").strip()

    app = Application()

    if choice == "1":
        await app.start_parser_only()
    elif choice == "2":
        await app.start_api_only()
    elif choice == "3":
        await app.start_both()
    elif choice == "4":
        print("Выход...")
        return
    else:
        print("Неверный выбор. Запускаю оба сервиса...")
        await app.start_both()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Приложение остановлено пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
