"""Handwritten image OCR App."""

import os

import gradio as gr
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class TrOCRInferencer:
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

    def inference(self, image: Image) -> str:
        """Inference using model."""
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return generated_text


inferencer = TrOCRInferencer()


def image_to_text(image: np.ndarray) -> str:
    image = Image.fromarray(image).convert("RGB")
    # NOTE: Can't save in colab
    # image.save("inputs/canvas.png", format="PNG")
    text = inferencer.inference(image)
    return text


# Set gradio app
with gr.Blocks() as app:
    gr.Markdown("# Handwritten Image OCR")
    with gr.Tab("Image upload"):
        image = gr.Image(label="Handwritten image file")
        output = gr.Textbox(label="Output Box")
        convert_btn = gr.Button("Convert")
        convert_btn.click(
            fn=image_to_text, inputs=image, outputs=output, api_name="image_to_text"
        )

        gr.Markdown("## Image Examples")
        gr.Examples(
            examples=[
                os.path.join(os.path.dirname(__file__), "examples/Red.png"),
                os.path.join(os.path.dirname(__file__), "examples/sentence.png"),
                os.path.join(os.path.dirname(__file__), "examples/i_love_you.png"),
                os.path.join(os.path.dirname(__file__), "examples/merrychristmas.png"),
                os.path.join(os.path.dirname(__file__), "examples/Rock.png"),
                os.path.join(os.path.dirname(__file__), "examples/Bob.png"),
            ],
            inputs=image,
            outputs=output,
            fn=image_to_text,
        )

    with gr.Tab("Drawing"):
        sketchpad = gr.Sketchpad(
            label="Handwritten Sketchpad",
            shape=(600, 192),
            brush_radius=2,
            invert_colors=False,
        )
        output = gr.Textbox(label="Output Box")
        convert_btn = gr.Button("Convert")
        convert_btn.click(
            fn=image_to_text, inputs=sketchpad, outputs=output, api_name="image_to_text"
        )


if __name__ == "__main__":
    app.launch()
