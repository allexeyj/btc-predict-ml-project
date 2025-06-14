import os
import pickle
from typing import Dict, List, Optional
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer
import logging
from .logger import get_logger

MODEL_STORE = "app/model_store"


class ModelWrapper:
    def __init__(self, model_id: str, model, vectorizer, description: str):
        self.model_id = model_id
        self.model = model
        self.vectorizer = vectorizer
        self.description = description

    def predict(self, text: str) -> float:
        X = self.vectorizer.transform([text])
        return float(self.model.predict(X)[0])


class ModelService:
    def __init__(self):
        self.models: Dict[str, ModelWrapper] = {}
        self.active_model_id: Optional[str] = None
        self.logger = get_logger()

    async def load_models(self):
        for filename in os.listdir(MODEL_STORE):
            if filename.endswith(".pkl"):
                path = os.path.join(MODEL_STORE, filename)
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    model_id = filename.replace(".pkl", "")
                    wrapper = ModelWrapper(
                        model_id=model_id,
                        model=data["model"],
                        vectorizer=data["vectorizer"],
                        description=f"Модель {filename}",
                    )
                    self.models[model_id] = wrapper
                    self.logger.info(f"загружена модель {model_id}")

        if self.models:
            self.active_model_id = list(self.models.keys())[0]
            self.logger.info(f"активирована модель {self.active_model_id}")

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

    def predict(self, text: str) -> float:
        if not self.active_model_id:
            raise RuntimeError("Модель не выбрана")
        model = self.models[self.active_model_id]
        return model.predict(text)

    def fit_model(self, hyperparameters: dict):
        self.logger.info(f"Train с параметрами: {hyperparameters}")
        import time

        time.sleep(5)
        self.logger.info("Train завершена")
