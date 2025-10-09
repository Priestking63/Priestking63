from fastapi import FastAPI, HTTPException,Query
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import json
import numpy as np
from typing import List, Optional
from pydantic import BaseModel
import logging


app = FastAPI(title='TG bot')

app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*']
)

class MessageResponse(BaseModel):
    message_id: int
    channel: str
    date: str
    text: str


class MessageWithEmbeddingResponse(BaseModel):
    message_id: int
    channel: str
    date: str
    text: str
    embedding: Optional[List[float]] = None
    embedding_size: Optional[int] = None


class SearchResponse(BaseModel):
    result: List[MessageResponse]
    total: int

class EmbeddingResponse(BaseModel):
    message_id: int
    channel: str
    embedding: List[float]
    embedding_size: int


class SearchWithEmbeddingsResponse(BaseModel):
    results: List[MessageWithEmbeddingResponse]
    total: int


class SimilaritySearchRequest(BaseModel):
    query_text: str
    limit: int = 10
    threshold: float = 0.7

class SimilarMessageResponse(BaseModel):
    message_id: int
    channel: str
    date: str
    text: str
    similarity_score: float

def get_db_connection():
    conn = sqlite3.connect('bot_base.db')
    conn.row_factory = sqlite3.Row
    return conn

def blob_to_embedding(embedding_blob):
    if embedding_blob is None:
        return None
    return np.frombuffer(embedding_blob, dtype=np.float32).tolist()

@app.get("/")
async def root():
    return {"message": "Telegram Parser API", "status": "online"}


@app.get("/messages", response_model=SearchResponse)
async def get_messages(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    channel: Optional[str] = None,
    search_text: Optional[str] = None,
):
    """Получить сообщения с пагинацией и фильтрацией"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Базовый запрос
        query = "SELECT * FROM skidki WHERE 1=1"
        params = []

        # Фильтр по каналу
        if channel:
            query += " AND channel = ?"
            params.append(channel)

        # Поиск по тексту
        if search_text:
            query += " AND text LIKE ?"
            params.append(f"%{search_text}%")

        # Получаем общее количество
        count_query = f"SELECT COUNT(*) as total FROM ({query})"
        total = cursor.execute(count_query, params).fetchone()["total"]

        # Получаем данные с пагинацией
        query += " ORDER BY date DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        messages = cursor.execute(query, params).fetchall()

        result = []
        for msg in messages:
            result.append(
                MessageResponse(
                    message_id=msg["message_id"],
                    channel=msg["channel"],
                    date=msg["date"],
                    text=msg["text"],
                )
            )

        conn.close()

        return SearchResponse(results=result, total=total)

    except Exception as e:
        logging.error(f"Ошибка получения сообщений: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/messages/with-embeddings", response_model=SearchWithEmbeddingsResponse)
async def get_messages_with_embeddings(
    limit: int = Query(50, ge=1, le=100),  # Меньший лимит из-за размера эмбедингов
    offset: int = Query(0, ge=0),
    channel: Optional[str] = None,
    has_embedding: bool = Query(True, description="Только сообщения с эмбедингами")
):
    """Получить сообщения с эмбедингами"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Базовый запрос
        if has_embedding:
            query = "SELECT * FROM skidki WHERE embedding IS NOT NULL"
        else:
            query = "SELECT * FROM skidki WHERE 1=1"
            
        params = []
        
        # Фильтр по каналу
        if channel:
            query += " AND channel = ?"
            params.append(channel)
        
        # Получаем общее количество
        count_query = f"SELECT COUNT(*) as total FROM ({query})"
        total = cursor.execute(count_query, params).fetchone()["total"]
        
        # Получаем данные с пагинацией
        query += " ORDER BY date DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        messages = cursor.execute(query, params).fetchall()
        
        result = []
        for msg in messages:
            embedding = None
            embedding_size = 0
            
            if msg["embedding"] is not None:
                embedding = blob_to_embedding(msg["embedding"])
                embedding_size = len(embedding) if embedding else 0
            
            result.append(MessageWithEmbeddingResponse(
                message_id=msg["message_id"],
                channel=msg["channel"],
                date=msg["date"],
                text=msg["text"],
                embedding=embedding,
                embedding_size=embedding_size
            ))
        
        conn.close()
        
        return SearchWithEmbeddingsResponse(results=result, total=total)
        
    except Exception as e:
        logging.error(f"Ошибка получения сообщений с эмбедингами: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/messages/{message_id}/embedding")
async def get_message_embedding(message_id: int, channel: str = Query(..., description="URL канала")):
    """Получить эмбединг конкретного сообщения"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        message = cursor.execute(
            "SELECT message_id, channel, embedding FROM skidki WHERE message_id = ? AND channel = ?", 
            (message_id, channel)
        ).fetchone()
        
        conn.close()
        
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        if message["embedding"] is None:
            raise HTTPException(status_code=404, detail="Embedding not found for this message")
            
        embedding = blob_to_embedding(message["embedding"])
        
        return EmbeddingResponse(
            message_id=message["message_id"],
            channel=message["channel"],
            embedding=embedding,
            embedding_size=len(embedding)
        )
        
    except Exception as e:
        logging.error(f"Ошибка получения эмбединга: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/search/similar")
async def search_similar_messages(request: SimilaritySearchRequest):
    """Поиск семантически похожих сообщений"""
    try:
        # Для этого эндпоинта нужно импортировать TextEmbedder
        from emb.transform import TextEmbedder
        
        embedder = TextEmbedder()
        query_embedding = embedder.get_embeddings(request.query_text)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Получаем все сообщения с эмбедингами
        messages = cursor.execute(
            "SELECT message_id, channel, date, text, embedding FROM skidki WHERE embedding IS NOT NULL"
        ).fetchall()
        
        similar_messages = []
        
        for msg in messages:
            msg_embedding = blob_to_embedding(msg["embedding"])
            
            # Вычисляем косинусное сходство
            similarity = cosine_similarity(query_embedding, msg_embedding)
            
            if similarity >= request.threshold:
                similar_messages.append({
                    "message": msg,
                    "similarity": similarity
                })
        
        # Сортируем по сходству
        similar_messages.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Берем топ-N результатов
        results = similar_messages[:request.limit]
        
        response = []
        for item in results:
            msg = item["message"]
            response.append(SimilarMessageResponse(
                message_id=msg["message_id"],
                channel=msg["channel"],
                date=msg["date"],
                text=msg["text"],
                similarity_score=round(item["similarity"], 4)
            ))
        
        conn.close()
        
        return {"query": request.query_text, "results": response}
        
    except Exception as e:
        logging.error(f"Ошибка семантического поиска: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/channels")
async def get_channels():
    """Получить список всех каналов"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        channels = cursor.execute(
            "SELECT DISTINCT channel FROM skidki ORDER BY channel"
        ).fetchall()
        
        conn.close()
        
        return {"channels": [channel["channel"] for channel in channels]}
        
    except Exception as e:
        logging.error(f"Ошибка получения каналов: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/stats/embeddings")
async def get_embedding_stats():
    """Статистика по эмбедингам"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Общая статистика
        total_messages = cursor.execute("SELECT COUNT(*) as count FROM skidki").fetchone()["count"]
        messages_with_embeddings = cursor.execute("SELECT COUNT(*) as count FROM skidki WHERE embedding IS NOT NULL").fetchone()["count"]
        
        # Статистика по каналам
        channel_stats = cursor.execute("""
            SELECT channel, 
                   COUNT(*) as total,
                   SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as with_embeddings
            FROM skidki 
            GROUP BY channel
            ORDER BY total DESC
        """).fetchall()
        
        conn.close()
        
        return {
            "total_messages": total_messages,
            "messages_with_embeddings": messages_with_embeddings,
            "coverage_percentage": round((messages_with_embeddings / total_messages * 100) if total_messages > 0 else 0, 2),
            "channels": [
                {
                    "channel": stat["channel"],
                    "total_messages": stat["total"],
                    "with_embeddings": stat["with_embeddings"],
                    "coverage": round((stat["with_embeddings"] / stat["total"] * 100) if stat["total"] > 0 else 0, 2)
                }
                for stat in channel_stats
            ]
        }
        
    except Exception as e:
        logging.error(f"Ошибка получения статистики: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Проверяем доступность базы данных
        cursor.execute("SELECT COUNT(*) as count FROM skidki")
        total_count = cursor.fetchone()["count"]
        
        cursor.execute("SELECT COUNT(*) as count FROM skidki WHERE embedding IS NOT NULL")
        embedding_count = cursor.fetchone()["count"]
        
        conn.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "total_messages": total_count,
            "messages_with_embeddings": embedding_count,
            "embedding_coverage": f"{(embedding_count / total_count * 100) if total_count > 0 else 0:.2f}%",
            "timestamp": str(datetime.now())
        }
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

def cosine_similarity(vec1, vec2):
    """Вычисляет косинусное сходство между двумя векторами"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

if __name__ == "__main__":
    import uvicorn
    from datetime import datetime
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
