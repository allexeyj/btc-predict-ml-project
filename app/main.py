from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from service import ModelService
from models import (
    PredictRequest,
    PredictResponse,
    FitRequest,
    ModelInfo,
    SetActiveModelRequest,
)
from logger import get_logger

app = FastAPI()
logger = get_logger()

service = ModelService()


@app.on_event("startup")
async def load_models():
    await service.load_models()
    logger.info("Начало загрузки моделей")


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    return service.list_models()


@app.post("/set_active_model")
async def set_active_model(req: SetActiveModelRequest):
    success = service.set_active_model(req.model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"message": f"Active model set to {req.model_id}"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    prediction = service.predict(req.text)
    return PredictResponse(prediction=prediction)


@app.post("/fit")
async def fit(req: FitRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(service.fit_model, req.hyperparameters)
    return JSONResponse(content={"message": "Training started in background"})
