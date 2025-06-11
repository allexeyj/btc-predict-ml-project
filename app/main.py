from fastapi import FastAPI
from app.api import router as api_router
from app.logger import logger
from app.service import load_models

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting FastAPI service")
    load_models()
    logger.info("Models загружены")

app.include_router(api_router)
