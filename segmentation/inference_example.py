"""
Simple Segment Anything(SAM) inference code.
"""


import os
import urllib

import cv2
import numpy as np
import PIL
import torch
from segment_anything import SamPredictor, sam_model_registry

CHECKPOINT_PATH = os.path.join("checkpoint")
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODEL_TYPE = "default"
MAX_WIDTH = MAX_HEIGHT = 1024
IMAGE_PATH = "examples/dog.jpg"
OUTPUT_PATH = "outputs/output.jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def adjust_image_size(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height > width:
        if height > MAX_HEIGHT:
            height, width = MAX_HEIGHT, int(MAX_HEIGHT / height * width)
    else:
        if width > MAX_WIDTH:
            height, width = int(MAX_WIDTH / width * height), MAX_WIDTH
    image = cv2.resize(image, (width, height))
    return image


def draw_contour(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # draw contour
    print("draw_mask: ", mask.shape)
    contours, _ = cv2.findContours(
        np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
    return image


def main():
    print("[INFO] Initialize")
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    checkpoint = os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME)
    if not os.path.exists(checkpoint):
        urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint).to(device)
    predictor = SamPredictor(sam)

    # Load image
    print("[INFO] Load sample image")
    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("origin image shape", image.shape)
    image = adjust_image_size(image)
    print("adjust image size: ", image.shape)

    # Set points
    point_coords = np.array([[450, 390]])
    points_labels = np.array([1])
    print("coords: ", point_coords.shape)
    print("labels: ", points_labels.shape)

    # Inference
    print("[INFO] Generate mask")
    predictor.set_image(image)
    masks, _, _ = predictor.predict(point_coords, points_labels)
    merged_mask = np.logical_or.reduce(masks, axis=0)

    # Draw contour of mask
    image = draw_contour(image, merged_mask)
    image = cv2.circle(
        image, point_coords[0], radius=15, color=(0, 215, 255), thickness=-1
    )

    # Save results
    print("[INFO] Save results")
    os.makedirs("outputs", exist_ok=True)
    image = PIL.Image.fromarray(image)
    image.save(OUTPUT_PATH)


if __name__ == "__main__":
    main()
