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
    image = np.bitwise_not(image)
    image = Image.fromarray(image).convert("RGB")
    image.save("inputs/canvas.png", format="PNG")
    text = inferencer.inference(image)
    return text


# Set gradio app
with gr.Blocks() as app:
    name = gr.Sketchpad(label="Handwritten", shape=(600, 192), brush_radius=2)
    output = gr.Textbox(label="Output Box")
    convert_btn = gr.Button("Convert")
    convert_btn.click(
        fn=image_to_text, inputs=name, outputs=output, api_name="image_to_text"
    )


if __name__ == "__main__":
    app.launch()
