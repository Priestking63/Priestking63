from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, cross_val_score
from transformers import PreTrainedTokenizer
from typing import List, Generator, Tuple, Optional
import pandas as pd
import math
import torch
from transformers import PreTrainedModel


@dataclass
class DataLoader:
    path: str
    tokenizer: PreTrainedTokenizer
    batch_size: int = 512
    max_length: int = 128
    padding: Optional[str] = None

    def __post_init__(self):
        # Подсчет общего количества строк (минус заголовок)
        with open(self.path, "r", encoding="utf-8") as f:
            self.total_rows = sum(1 for _ in f) - 1

    def __iter__(self) -> Generator[Tuple[List[List[int]], List[int]], None, None]:
        """Iterate over batches"""
        for i in range(len(self)):
            yield self.batch_tokenized(i)

    def __len__(self):
        """Number of batches"""
        return math.ceil(self.total_rows / self.batch_size)

    def tokenize(self, batch: List[str]) -> List[List[int]]:
        """Tokenize list of texts"""
        if self.padding is None:
            # Старое поведение - без паддинга
            encoded = self.tokenizer(
                batch,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
                add_special_tokens=True,
            )
            return encoded["input_ids"]

        elif self.padding == "max_length":
            # Паддинг до фиксированной длины
            encoded = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
                add_special_tokens=True,
            )
            return encoded["input_ids"]

        elif self.padding == "batch":
            # Динамический паддинг до самой длинной последовательности в батче
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
                add_special_tokens=True,
            )
            return encoded["input_ids"]

        else:
            raise ValueError(f"Unsupported padding type: {self.padding}")

    def batch_loaded(self, i: int) -> Tuple[List[str], List[int]]:
        """Return loaded i-th batch of data (text, label)"""
        skip_rows = 1 + i * self.batch_size
        n_rows = min(self.batch_size, self.total_rows - i * self.batch_size)

        texts = []
        labels = []

        with open(self.path, "r", encoding="utf-8") as f:
            # Пропускаем заголовок
            header = f.readline()

            # Пропускаем нужное количество строк
            for _ in range(skip_rows - 1):
                f.readline()

            # Читаем n_rows строк
            for _ in range(n_rows):
                line = f.readline()
                if not line:
                    break

                # Разделяем строку по запятым
                parts = line.strip().split(",", 4)

                if len(parts) < 5:
                    continue

                # Извлекаем поля
                review_id, dt, rating, sentiment, review = parts

                # Преобразуем текстовые метки в числовые
                sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
                numeric_label = sentiment_map.get(sentiment, 0)

                labels.append(numeric_label)
                texts.append(review)

        return texts, labels

    def batch_tokenized(self, i: int) -> Tuple[List[List[int]], List[int]]:
        """Return tokenized i-th batch of data"""
        texts, labels = self.batch_loaded(i)
        tokens = self.tokenize(texts)
        return tokens, labels


def attention_mask(padded: List[List[int]]) -> List[List[int]]:
    """Create attention mask for padded sequences"""
    mask = []
    for sequence in padded:
        # 1 для реальных токенов, 0 для паддинга (токены с ID 0)
        seq_mask = [1 if token_id != 0 else 0 for token_id in sequence]
        mask.append(seq_mask)
    return mask


def review_embedding(
    tokens: List[List[int]], model: PreTrainedModel
) -> List[List[float]]:
    """Return embedding for batch of tokenized texts"""
    # Преобразуем в тензоры
    tokens_tensor = torch.tensor(tokens)

    # Создаем attention mask
    mask = attention_mask(tokens)
    mask_tensor = torch.tensor(mask)

    # Вычисляем эмбеддинги без градиентов
    with torch.no_grad():
        outputs = model(tokens_tensor, attention_mask=mask_tensor)
        last_hidden_states = outputs.last_hidden_state

    # Берем эмбеддинги для [CLS]-токенов (первый токен в каждой последовательности)
    cls_embeddings = last_hidden_states[:, 0, :]

    return cls_embeddings.tolist()


def evaluate(model,
    embeddings: List[List[float]], labels: List[int], cv: int = 5
) -> List[float]:
    """
    Evaluate embeddings using KFold cross-validation and return Cross-Entropy Loss for each fold.
    """
    X = np.array(embeddings)
    y = np.array([label + 1 for label in labels])

    kf = KFold(n_splits=cv, shuffle=False)

    fold_losses = [
        log_loss(
            y[test_idx],
            model.fit(X[train_idx], y[train_idx]).predict_proba(X[test_idx]),
        )
        for train_idx, test_idx in kf.split(X)
    ]

    return fold_losses
