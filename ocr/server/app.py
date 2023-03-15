"""Handwritten image to Text API Server."""

import utils
from fastapi import FastAPI, UploadFile

app = FastAPI()
logger = utils.get_logger()


###################
# APIs
###################
@app.get("/")
def healthcheck() -> bool:
    """Ping and pong for healthcheck."""
    logger.info("Health Check Requested.")
    return True
