from typing import IO

import os
import requests

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
)
from diffusers.utils import load_image
from PIL import Image

os.makedirs("outputs", exist_ok=True)

WIDTH = 512
HEIGHT = 512
ROOT_URL = "https://civitai.com"

DEVICE = "cuda"
PROMPT = "1girl, masterpiece, best quality, solo, standing, blush, breasts, closed mouth, frills, jewelry, long hair, sleeveless, smile, solo, twintails"
NEGATIVE_PROMPT = "nsfw, multiple girls, 2girls, 3girls, 4girls, ugly, fused fingers, cropped, bad anatomy, watermark, words, letters, untracked eyes, asymmetric eyes, floating head, logo, bad hands, mangled hands, missing hands, missing arms, backward hands, floating jewelry, unattached jewelry, unattached head, doubled head, head in body, misshapen body, badly fitted headwear, floating arms, too many arms, limbs fused with body, facial blemish, badly fitted clothes, imperfect eyes, crossed eyes, hair growing from clothes, partial faces, hair not attached to head"

PIPELINE = None

# "models/anything_inkBase.safetensors",
# "models/anythingV3_fp16.ckpt",

# pipe.save_pretrained(save_path, safe_serialization=True)
# self.a_prompt = "1girl, masterpiece, best quality, solo, standing, bangs, blush, breasts, choker, closed mouth, cone hair bun, frills, jewelry, long hair, on head, bird, sleeveless, smile, solo, twintails, virtual youtuber"
# self.n_prompt = "nsfw, multiple girls, 2girls, 3girls, 4girls, (ugly:1.3), (fused fingers), (((cropped))), (bad anatomy:1.5), (watermark:1.5), (words), letters, untracked eyes, asymmetric eyes, floating head, (logo:1.5), (bad hands:1.3), (mangled hands:1.2), (missing hands), (missing arms), backward hands, floating jewelry, unattached jewelry, unattached head, doubled head, head in body, (misshapen body:1.1), (badly fitted headwear:1.2), floating arms, (too many arms:1.5), limbs fused with body, (facial blemish:1.5), badly fitted clothes, imperfect eyes, crossed eyes, hair growing from clothes, partial faces, hair not attached to head"


def init_pipeline(model_file: IO) -> str:
    print("[INFO] Initializing pipeline")
    global PIPELINE
    PIPELINE = StableDiffusionImg2ImgPipeline.from_single_file(
        model_file.name,
        torch_dtype=torch.float16,
        variant="fp16",
        load_safety_checker=False,
        use_safetensors=True,
    ).to("cuda")

    PIPELINE.scheduler = EulerAncestralDiscreteScheduler.from_config(PIPELINE.scheduler.config)
    print("[INFO] Initialized pipeline")
    return "Model loaded!"


def sketch_to_image(sketch: Image.Image, prompt: str, negative_prompt: str, strength: float):
    width, height = sketch.size
    return PIPELINE(
        image=sketch,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_images_per_prompt=4,
        num_inference_steps=20,
        strength=strength,
        guidance_scale=7.5,
    ).images


def download_model(model_url: str) -> str:
    parsed = model_url.replace(f"{ROOT_URL}/", "").split("?modelVersionId=")
    model_id = parsed[0].replace("models/", "").split("/")[0]

    try:
        response = requests.get(
            f"{ROOT_URL}/api/v1/models/{model_id}", stream=True, timeout=600
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    download_url = response.json()["modelVersions"][0]["downloadUrl"]
    filename = response.json()["modelVersions"][0]["files"][0]["name"]

    file_path = f"models/{filename}"
    if os.path.exists(file_path):
        print(f"[INFO] File already exists: {file_path}")
        return file_path

    os.makedirs("models", exist_ok=True)
    os.system(f"wget -O '{file_path}' '{download_url}'")
    print(f"[INFO] File downloaded: {file_path}")
    return file_path


print("[INFO] Gradio app ready")
with gr.Blocks() as app:
    gr.Markdown("# Sketch to Image 애플리케이션")

    gr.Markdown("## 모델 다운로드")
    with gr.Row():
        model_url = gr.Textbox(label="Model Link", placeholder="https://civitai.com/")
        download_model_btn = gr.Button(value="Download model")
    with gr.Row():
        model_file = gr.File(label="Model File")

    gr.Markdown("## 모델 불러오기")
    with gr.Row():
        load_model_btn = gr.Button(value="Load model")
    with gr.Row():
        is_model_check = gr.Textbox(label="Model Load Check", value="Model Not loaded")

    gr.Markdown("## 프롬프트 입력")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value=PROMPT)
    with gr.Row():
        n_prompt = gr.Textbox(label="Negative Prompt", value=NEGATIVE_PROMPT)

    gr.Markdown("## 이미지 입력")
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
                    canvas_run_btn = gr.Button(value="Generate")

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
                    file_run_btn = gr.Button(value="Generate")

        with gr.Column():
            result_gallery = gr.Gallery(
                label="Output", elem_id="gallery", rows=2, height=768,
            )

    with gr.Row():
        with gr.Column():
            with gr.Accordion("Advanced options", open=False):
                strength = gr.Slider(
                    label="Strength",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.65,
                    step=0.01,
                )

    # Event
    download_model_btn.click(
        download_model,
        [model_url],
        [model_file],
    )
    load_model_btn.click(
        init_pipeline,
        [model_file],
        [is_model_check],
    )
    canvas_run_btn.click(
        sketch_to_image,
        [canvas, prompt, n_prompt, strength],
        [result_gallery],
    )
    file_run_btn.click(
        sketch_to_image,
        [file, prompt, n_prompt, strength],
        [result_gallery],
    )

if __name__ == "__main__":
    app.launch(inline=False, share=True)
