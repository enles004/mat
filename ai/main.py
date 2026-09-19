import logging

import uvicorn

from src.api.dependencies import load_model_from_artifact
from src.api.server import create_app
from src.core.settings import Settings

logging.basicConfig(level=logging.INFO)

settings = Settings()
app = create_app(settings, load_model_from_artifact)


def run() -> None:
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
