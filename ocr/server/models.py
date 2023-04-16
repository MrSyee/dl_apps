"""API models."""

from pydantic import BaseModel, Field


class OCRResponse(BaseModel):
    """Response of API."""

    text: str = Field(
        ...,
        example="Sample text",
        description="Generated text from image.",
    )
