from typing import List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


class SentimentModel:
    def __init__(
        self,
        name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    ):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.name)

    def predict_proba(self, inputs: List[str]) -> np.ndarray:
        tokens = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            logits = self.model(**tokens).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs.detach().numpy()
