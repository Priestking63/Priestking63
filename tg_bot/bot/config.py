import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    API_ID = int(os.getenv("API_ID", 0))
    API_HASH = os.getenv("API_HASH", "")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))

    @classmethod
    def validate(cls):
        if not cls.API_ID or not cls.API_HASH:
            raise ValueError("API_ID и API_HASH должны быть установлены в .env файле")
