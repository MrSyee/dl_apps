from typing import Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting, UniPCMultistepScheduler
from PIL import Image

# SD15: runwayml/stable-diffusion-inpainting
# SDXL: diffusers/stable-diffusion-xl-1.0-inpainting-0.1
pipeline = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16
).to("cuda")
pipeline.enable_model_cpu_offload()

pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

seed = 7777
generator = [
    torch.Generator(device="cuda").manual_seed(seed + i) for i in range(4)
]


def inpaint_from_mask(image_dict: Image, prompt: str):
    image = image_dict["image"]
    mask = image_dict["mask"]

    height, width = image.shape[:2]

    generated_images = pipeline(
        prompt=prompt,
        image=Image.fromarray(image),
        mask_image=Image.fromarray(mask),
        num_inference_steps=20,
        num_images_per_prompt=4,
        original_size=(height, width),
        target_size=(height, width),
        # height=height,
        # width=width,
        generator=generator
    ).images

    generated_images[0].save("./outputs/output.png")
    print("Complete")
    return generated_images


with gr.Blocks() as app:
    gr.Markdown("# AI 이미지 에디터")
    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            lines=3,
            value="a black cat with glowing eyes on the bench, big cat, cute, highly detailed, 8k",
        )
    with gr.Row():
        with gr.Tab("Mask"):
            input_img = gr.Image(
                label="Input image",
                height=600,
                tool="sketch",
                source="upload",
                brush_radius=100,
            )
            inpaint_btn = gr.Button(value="Inpaint!")

    with gr.Row():
        gallary = gr.Gallery(rows=1)

    inpaint_btn.click(inpaint_from_mask, [input_img, prompt], [gallary])

app.launch(inline=False, share=True)
