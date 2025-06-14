from pydantic import BaseModel
from typing import List, Dict

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: float

class FitRequest(BaseModel):
    hyperparameters: Dict[str, float]  

class ModelInfo(BaseModel):
    model_id: str
    description: str

class SetActiveModelRequest(BaseModel):
    model_id: str
