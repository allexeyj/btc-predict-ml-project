from typing import Dict

models_store: Dict[str, dict] = {}

def load_models():
    models_store["model_1"] = {"name": "Model 1", "description": "-", "status": "загружена"}

def get_models_list():
    return [
        {"id": model_id, **info}
        for model_id, info in models_store.items()
    ]
