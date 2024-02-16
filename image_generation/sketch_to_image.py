import os

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    DiffusionPipeline,
    UniPCMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    AutoPipelineForImage2Image,
    StableDiffusionImg2ImgPipeline,
)
from diffusers.utils import load_image
from PIL import Image

os.makedirs("outputs", exist_ok=True)

WIDTH = 768
HEIGHT = 1024

DEVICE = "cuda"

# "models/anything_inkBase.safetensors",
# "models/anythingV3_fp16.ckpt",

def init_pipeline() -> DiffusionPipeline:
    pipeline = StableDiffusionImg2ImgPipeline.from_single_file(
        "models/anythingV3_fp16.ckpt",
        torch_dtype=torch.float16,
        variant="fp16",
        load_safety_checker=False,
        use_safetensors=True,
    ).to("cuda")

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    return pipeline

pipeline = init_pipeline()


# pipe.save_pretrained(save_path, safe_serialization=True)
# self.a_prompt = "1girl, masterpiece, best quality, solo, standing, bangs, blush, breasts, choker, closed mouth, cone hair bun, frills, jewelry, long hair, on head, bird, sleeveless, smile, solo, twintails, virtual youtuber"
# self.n_prompt = "nsfw, multiple girls, 2girls, 3girls, 4girls, (ugly:1.3), (fused fingers), (((cropped))), (bad anatomy:1.5), (watermark:1.5), (words), letters, untracked eyes, asymmetric eyes, floating head, (logo:1.5), (bad hands:1.3), (mangled hands:1.2), (missing hands), (missing arms), backward hands, floating jewelry, unattached jewelry, unattached head, doubled head, head in body, (misshapen body:1.1), (badly fitted headwear:1.2), floating arms, (too many arms:1.5), limbs fused with body, (facial blemish:1.5), badly fitted clothes, imperfect eyes, crossed eyes, hair growing from clothes, partial faces, hair not attached to head"


def sketch_to_image(sketch: Image.Image, prompt: str):
    width, height = sketch.size
    return pipeline(
        image=sketch,
        prompt=prompt,
        negative_prompt="nsfw, multiple girls, 2girls, 3girls, 4girls, ugly, fused fingers, cropped, bad anatomy, watermark, words, letters, untracked eyes, asymmetric eyes, floating head, logo, bad hands, mangled hands, missing hands, missing arms, backward hands, floating jewelry, unattached jewelry, unattached head, doubled head, head in body, misshapen body, badly fitted headwear, floating arms, too many arms, limbs fused with body, facial blemish, badly fitted clothes, imperfect eyes, crossed eyes, hair growing from clothes, partial faces, hair not attached to head",
        height=height,
        width=width,
        num_images_per_prompt=4,
        num_inference_steps=20,
        strength=0.65,
        guidance_scale=7.5,
    ).images


print("[INFO] Gradio app ready")
with gr.Blocks() as app:
    gr.Markdown("# Sketch to Cartoon Image")

    with gr.Row():
        with gr.Column():
            with gr.Tab("Canvas"):
                with gr.Row():
                    canvas = gr.Image(
                        label="Draw",
                        source="canvas",
                        image_mode="RGB",
                        tool="color-sketch",
                        interactive=True,
                        width=WIDTH,
                        height=HEIGHT,
                        shape=(WIDTH, HEIGHT),
                        brush_radius=20,
                        type="pil",
                    )
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", value="human, masterpiece, best quality, solo, standing", placeholder="Type here")
                with gr.Row():
                    canvas_run_btn = gr.Button(label="Run")

            with gr.Tab("File"):
                with gr.Row():
                    file = gr.Image(
                        label="Upload",
                        source="upload",
                        image_mode="RGB",
                        tool="color-sketch",
                        interactive=True,
                        width=WIDTH,
                        height=HEIGHT,
                        shape=(WIDTH, HEIGHT),
                        type="pil",
                    )
                with gr.Row():
                    file_run_btn = gr.Button(label="Run")

        with gr.Column():
            result_gallery = gr.Gallery(
                label="Output", elem_id="gallery", rows=2, height=1024
            )

        canvas_run_btn.click(
            sketch_to_image,
            [canvas, prompt],
            [result_gallery],
        )
        file_run_btn.click(
            sketch_to_image,
            [file, prompt],
            [result_gallery],
        )

if __name__ == "__main__":
    app.launch(inline=False, share=True)
