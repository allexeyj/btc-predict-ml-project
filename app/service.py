import os
import pickle
from typing import Dict, List, Optional, Union
import logging
from logger import get_logger
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel

MODEL_STORE = "model_store"


class BertEmbedder:
    def __init__(self, model_name="sergeyzh/BERTA", pooling_method="mean"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.pooling_method = pooling_method

    def pool(self, hidden_state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_method == "mean":
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(dim=1, keepdim=True).float()
            return s / d
        elif self.pooling_method == "cls":
            return hidden_state[:, 0]
        elif self.pooling_method == "max":
            masked = hidden_state * mask.unsqueeze(-1).float()
            masked_fill = masked + (1 - mask.unsqueeze(-1).float()) * -1e9
            return torch.max(masked_fill, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

    def embed(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            hidden_state = outputs.last_hidden_state
        pooled = self.pool(hidden_state, attention_mask)
        return pooled.squeeze(0).cpu().numpy()


class ModelWrapper:
    def __init__(
        self,
        model_id: str,
        model,
        description: str,
        embedder: Optional[BertEmbedder] = None,
    ):
        self.model_id = model_id
        self.model = model
        self.description = description
        self.embedder = embedder

    def predict_from_features(
        self, features: Union[List[float], np.ndarray, pd.DataFrame]
    ) -> float:
        if isinstance(features, list):
            features = np.array(features).reshape(1, -1)
        elif isinstance(features, pd.DataFrame):
            features = features.values
        elif isinstance(features, np.ndarray) and features.ndim == 1:
            features = features.reshape(1, -1)

        pred = self.model.predict(features)
        return float(pred[0])

    def predict(
        self, input_data: Union[str, List[float], np.ndarray, pd.DataFrame]
    ) -> float:
        if isinstance(input_data, str):
            features = self.embedder.embed(input_data).reshape(1, -1)
        else:
            features = input_data

        return self.predict_from_features(features)


class ModelService:
    def __init__(self):
        self.models: Dict[str, ModelWrapper] = {}
        self.active_model_id: Optional[str] = None
        self.logger = get_logger()
        self.embedder = BertEmbedder()

    async def load_models(self):
        for filename in os.listdir(MODEL_STORE):
            if filename.endswith(".pkl"):
                path = os.path.join(MODEL_STORE, filename)
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    model_id = filename.replace(".pkl", "")
                    wrapper = ModelWrapper(
                        model_id=model_id,
                        model=data,
                        description=f"Модель {filename}",
                        embedder=self.embedder,
                    )
                    self.models[model_id] = wrapper
                    self.logger.info(f"Загружена модель {model_id}")

        if self.models:
            self.active_model_id = list(self.models.keys())[0]
            self.logger.info(f"Активирована модель {self.active_model_id}")

    def list_models(self) -> List[Dict]:
        return [
            {"model_id": m.model_id, "description": m.description}
            for m in self.models.values()
        ]

    def set_active_model(self, model_id: str) -> bool:
        if model_id in self.models:
            self.active_model_id = model_id
            self.logger.info(f"Модель изменена на {model_id}")
            return True
        return False

    def predict(
        self, input_data: Union[str, List[float], np.ndarray, pd.DataFrame]
    ) -> float:
        model = self.models[self.active_model_id]
        return model.predict(input_data)

    def fit_model(self, hyperparameters: dict):
        self.logger.info(f"Train с параметрами: {hyperparameters}")
        import time

        time.sleep(5)
        self.logger.info("Train завершена")
