from sentence_transformers import SentenceTransformer
import numpy as np


class TextEmbedder:
    def __init__(
        self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts):
        embedding = self.model.encode(texts)
        blob = embedding.tobytes()
        return blob

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычисляет косинусное сходство между двумя векторами"""
        if vec1 is None or vec2 is None or len(vec1) != len(vec2):
            return 0.0

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
