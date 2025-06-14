import logging
import os


def get_logger():
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("app_logger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("app/logs/app.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
