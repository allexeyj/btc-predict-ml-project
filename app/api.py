from fastapi import APIRouter
from app.models import ModelsListResponse, ModelInfo
from app.service import get_models_list

router = APIRouter()

@router.get("/", tags=["Main check"])
async def root():
    return {"message": "FastAPI starts"}

@router.get("/models", response_model=ModelsListResponse, tags=["Models"])
async def get_models():
    models = get_models_list()
    return ModelsListResponse(models=[ModelInfo(**m) for m in models])
