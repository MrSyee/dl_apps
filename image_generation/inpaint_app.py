from typing import Tuple
import os
import urllib

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
pipeline = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()

pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

seed = 7777
generator = [
    torch.Generator(device="cuda").manual_seed(seed + i) for i in range(4)
]

# SAM
CHECKPOINT_PATH = "checkpoint"
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODEL_TYPE = "default"
IMAGE_PATH = "examples/mannequin.jpg"
OUTPUT_PATH = "outputs/output.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAMInferencer:
    def __init__(
        self,
        checkpoint_path: str,
        checkpoint_name: str,
        checkpoint_url: str,
        model_type: str,
        device: torch.device,
    ):
        print("[INFO] Initailize inferencer")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint = os.path.join(checkpoint_path, checkpoint_name)
        if not os.path.exists(checkpoint):
            urllib.request.urlretrieve(checkpoint_url, checkpoint)
        sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
        self.predictor = SamPredictor(sam)

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

    print("mask", mask.shape, mask.dtype, np.unique(mask))
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    print("after mask", mask.shape, mask.dtype, np.unique(mask))

    return mask


def extract_object_by_event(image: np.ndarray, evt: gr.SelectData):
    click_x, click_y = evt.index

    return extract_object(image, click_x, click_y)


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


def inpaint_from_mask(image_dict: dict, prompt: str):
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


def inpaint_with_mask(image: np.ndarray, mask: np.ndarray, prompt: str):
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

        with gr.Tab("Click"):
            with gr.Row():
                input_img = gr.Image(
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
        gallary = gr.Gallery(rows=1)

    inpaint_btn.click(inpaint_from_mask, [input_img, prompt], [gallary])

    input_img.select(extract_object_by_event, [input_img], [mask_img])
    click_inpaint_btn.click(inpaint_with_mask, [input_img, mask_img, prompt], [gallary])

app.launch(inline=False, share=True)
