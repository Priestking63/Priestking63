from sentence_transformers import SentenceTransformer
import numpy as np


class TextEmbedder:
    def __init__(
        self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts):
        try:
            if not texts or not texts.strip():
                return None

            if isinstance(texts, str):
                texts = [texts]

            embedding = self.model.encode(texts)

            if len(embedding) == 0:
                return None

            # Преобразуем в BLOB
            blob = embedding.tobytes()
            return blob

        except Exception as e:
            return None
