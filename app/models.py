from pydantic import BaseModel
from typing import List

class ModelInfo(BaseModel):
    id: str
    name: str
    description: str

class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
