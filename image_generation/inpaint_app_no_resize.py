import os
import urllib
from typing import Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting, UniPCMultistepScheduler
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

#############
# Initialize
#############
# Pipeline
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipeline = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16
).to(DEVICE)
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_vae_tiling()

seed = 7777
num_images = 4
generator = [
    torch.Generator(device="cuda").manual_seed(seed + i) for i in range(num_images)
]


# SAM
CHECKPOINT_PATH = "checkpoint"
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODEL_TYPE = "default"


class SAMInferencer:
    def __init__(
        self,
        checkpoint_path: str,
        checkpoint_name: str,
        checkpoint_url: str,
        model_type: str,
        device: torch.device,
    ):
        print("[INFO] Initailize inferencer.")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint = os.path.join(checkpoint_path, checkpoint_name)
        if not os.path.exists(checkpoint):
            urllib.request.urlretrieve(checkpoint_url, checkpoint)
        sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
        self.predictor = SamPredictor(sam)
        print("[INFO] Initialization complete!")

    def inference(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        points_labels: np.ndarray,
    ) -> np.ndarray:
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(point_coords, points_labels)
        mask, _ = self.select_mask(masks, scores)
        return mask

    def select_mask(
        self, masks: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Determine if we should return the multiclick mask or not from the number of points.
        # The reweighting is used to avoid control flow.
        # Reference: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/utils/onnx.py#L92-L105
        score_reweight = np.array([-1000] + [0] * 2)
        score = scores + score_reweight
        best_idx = np.argmax(score)
        selected_mask = np.expand_dims(masks[best_idx, :, :], axis=-1)
        selected_score = np.expand_dims(scores[best_idx], axis=0)
        return selected_mask, selected_score


sam_inferencer = SAMInferencer(
    CHECKPOINT_PATH, CHECKPOINT_NAME, CHECKPOINT_URL, MODEL_TYPE, DEVICE
)


def extract_object(image: np.ndarray, point_x: int, point_y: int):
    point_coords = np.array([[point_x, point_y], [0, 0]])
    point_label = np.array([1, -1])

    # Get mask
    mask = sam_inferencer.inference(image, point_coords, point_label)

    # Postprocess mask
    mask = (mask > 0).astype(np.uint8) * 255

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    return mask


def extract_object_by_event(image: np.ndarray, evt: gr.SelectData):
    click_x, click_y = evt.index
    return extract_object(image, click_x, click_y)


def inpaint_drawing_mask(image_dict: Image, prompt: str):
    image = image_dict["image"]
    mask = image_dict["mask"]

    height, width = image.shape[:2]

    generated_images = pipeline(
        prompt=prompt,
        image=Image.fromarray(image),
        mask_image=Image.fromarray(mask),
        num_inference_steps=20,
        num_images_per_prompt=num_images,
        original_size=(height, width),
        target_size=(height, width),
        generator=generator,
    ).images
    return generated_images


def inpaint_click_mask(image: np.ndarray, mask: np.ndarray, prompt: str):
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    height, width = image.shape[:2]

    generated_images = pipeline(
        prompt=prompt,
        image=Image.fromarray(image),
        mask_image=Image.fromarray(mask),
        original_size=(height, width),
        target_size=(height, width),
        num_inference_steps=20,
        num_images_per_prompt=num_images,
        generator=generator,
    ).images
    return generated_images


with gr.Blocks() as app:
    gr.Markdown("# AI 이미지 에디터")
    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            lines=3,
            value="black cat on the bench, big cat, cute cat, highly detailed, high quality photography",
        )
    with gr.Row():
        with gr.Tab("Drawing Mask"):
            input_img_for_drawing = gr.Image(
                label="Input image",
                height=600,
                tool="sketch",
                source="upload",
                brush_radius=100,
            )
            drawing_inpaint_btn = gr.Button(value="Inpaint!")

        with gr.Tab("Click Mask"):
            with gr.Row():
                input_img_for_click = gr.Image(
                    label="Input image",
                    height=600,
                )
            with gr.Row():
                mask_img = gr.Image(
                    label="Mask image",
                    height=600,
                )
            click_inpaint_btn = gr.Button(value="Inpaint!")

    with gr.Row():
        gallery = gr.Gallery(
            height=600,
            show_download_button=True,
            preview=True,
        )

    drawing_inpaint_btn.click(
        inpaint_drawing_mask, [input_img_for_drawing, prompt], [gallery]
    )

    input_img_for_click.select(
        extract_object_by_event, [input_img_for_click], [mask_img]
    )
    click_inpaint_btn.click(
        inpaint_click_mask, [input_img_for_click, mask_img, prompt], [gallery]
    )
