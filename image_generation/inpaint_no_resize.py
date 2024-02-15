from typing import Tuple

import cv2
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image

pipeline = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()


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
    image_resized = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    mask_resized = cv2.resize(
        mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

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
    image = cv2.resize(
        image, dsize=(origin_width, origin_height), interpolation=cv2.INTER_LINEAR
    )
    mask = cv2.resize(
        mask, dsize=(origin_width, origin_height), interpolation=cv2.INTER_LINEAR
    )

    return image, mask


def inpaint_from_mask(image_dict: Image, prompt: str):
    image = image_dict["image"]
    mask = image_dict["mask"]

    output_images = pipeline(
        prompt=prompt,
        image=Image.fromarray(image),
        mask_image=Image.fromarray(mask),
        num_images_per_prompt=4,
    ).images

    output_images[0].save("./outputs/output.png")
    return output_images


def main():
    image = Image.open("examples/dog_on_the_bench.png")
    image = np.array(image)

    mask = Image.open("outputs/mask.png")
    mask = np.array(mask)

    image_dict = {
        "image": image,
        "mask": mask,
    }

    prompt = "a black cat with glowing eyes on the bench, cute, highly detailed, 8k"

    output_images = inpaint_from_mask(image_dict, prompt)


if __name__ == "__main__":
    main()
