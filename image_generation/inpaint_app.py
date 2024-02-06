from typing import Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting, UniPCMultistepScheduler
from PIL import Image

pipeline = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()

pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

seed = 7777
generator = [
    torch.Generator(device="cuda").manual_seed(seed + i) for i in range(4)
]

def resize_and_pad(
    image: np.ndarray, mask: np.ndarray, target_size: int = 1024
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    이미지와 마스크를 리사이즈합니다.
    가로와 세로 중 긴 부분이 target_size가 되도록 리사이즈하고, 짧은 쪽도 target_size가 될 수 있도록 패딩합니다.
    """
    # Resize
    height, width, _ = image.shape
    max_dim = max(height, width)
    scale = target_size / max_dim
    new_height = int(height * scale)
    new_width = int(width * scale)
    image_resized = cv2.resize(image, (new_width, new_height))
    mask_resized = cv2.resize(mask, (new_width, new_height))

    # Pad
    pad_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_image[:new_height, :new_width, :] = image_resized

    pad_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    pad_mask[:new_height, :new_width] = mask_resized

    return pad_image, pad_mask, (height, width), (new_height, new_width)


def restore(
    pad_image: np.ndarray,
    pad_mask: np.ndarray,
    origin_shape: Tuple[int, int],
    resize_shape: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    리사이즈된 이미지와 마스크를 원본 사이즈로 리사이즈.
    """
    # Unpadding
    resize_height, resize_width = resize_shape

    image = pad_image[:resize_height, :resize_width]
    mask = pad_mask[:resize_height, :resize_width]

    # Resize
    origin_height, origin_width = origin_shape
    image = cv2.resize(image, dsize=(origin_width, origin_height))
    mask = cv2.resize(mask, dsize=(origin_width, origin_height))

    return image, mask


def inpaint_from_mask(image_dict: Image, prompt: str):
    image = image_dict["image"]
    mask = image_dict["mask"]

    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    resized_image, resized_mask, origin_shape, new_shape = resize_and_pad(image, mask)

    generated_images = pipeline(
        prompt=prompt,
        image=Image.fromarray(resized_image),
        mask_image=Image.fromarray(resized_mask),
        num_inference_steps=20,
        num_images_per_prompt=4,
        generator=generator,
    ).images

    output_images = []
    for generated_image in generated_images:
        generated_image = np.asarray(generated_image)
        resized_mask = np.asarray(resized_mask)

        restored_image, restored_mask = restore(generated_image, resized_mask, origin_shape, new_shape)

        restored_mask = np.expand_dims(restored_mask, -1) / 255
        output_image = restored_image * restored_mask + image * (1 - restored_mask)
        output_images.append(Image.fromarray(output_image.astype(np.uint8)))

    output_images[0].save("./outputs/output.png")
    print("Complete")
    return output_images


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
