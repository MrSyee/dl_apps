"""Handwritten image to Text API Server."""

import utils
from fastapi import FastAPI, UploadFile
from PIL import Image

from tasks import TrOCRInferencer
from models import OCRResponse

app = FastAPI()
logger = utils.get_logger()

inferencer = TrOCRInferencer()

###################
# APIs
###################
@app.get("/")
def healthcheck() -> bool:
    """Ping and pong for healthcheck."""
    logger.info("Health Check Requested.")
    return True


@app.post("/infer", response_model=OCRResponse, status_code=200)
async def infer(file: UploadFile) -> OCRResponse:
    """Inference model."""
    print(file)
    image = Image.open(file.file).convert("RGB")
    text = inferencer.inference(image)

    return OCRResponse(text=text)