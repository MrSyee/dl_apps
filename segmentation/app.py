import os
import urllib
from typing import Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

CHECKPOINT_PATH = os.path.join("checkpoint")
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODEL_TYPE = "default"
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
        # merged_mask = np.logical_or.reduce(masks, axis=0)
        mask, score = self.select_masks(masks, scores, point_coords.shape[0])
        return mask

    def select_masks(
        self, masks: np.ndarray, iou_preds: np.ndarray, num_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Determine if we should return the multiclick mask or not from the number of points.
        # The reweighting is used to avoid control flow.
        score_reweight = np.array([1000] + [0] * 2)
        score = iou_preds + (num_points - 2.5) * score_reweight
        best_idx = np.argmax(score)
        masks = np.expand_dims(masks[best_idx, :, :], axis=-1)
        iou_preds = np.expand_dims(iou_preds[best_idx], axis=0)
        return masks, iou_preds


inferencer = SAMInferencer(
    CHECKPOINT_PATH, CHECKPOINT_NAME, CHECKPOINT_URL, MODEL_TYPE, DEVICE
)


def draw_contour(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # draw contour
    contour_image = image.copy()
    contours, _ = cv2.findContours(
        np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 3)
    return contour_image, contours


def extract_object(image: np.ndarray, point_h: int, point_w: int):
    point_coords = np.array([[point_h, point_w], [0, 0]])
    point_label = np.array([1, -1])
    # image_pil = Image.fromarray(image).convert("RGB")
    # image_pil.save("inputs/origin.png", format="PNG")

    # Get mask
    mask = inferencer.inference(image, point_coords, point_label)

    # Extract object
    mask = mask.astype(np.uint8) * 255
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    return segmented_image


def extract_object_by_event(image: np.ndarray, evt: gr.SelectData):
    click_h, click_w = evt.index

    return extract_object(image, click_h, click_w)


def get_coords(evt: gr.SelectData):
    return evt.index[0], evt.index[1]


print("[INFO] Gradio app ready")
with gr.Blocks() as demo:
    gr.Markdown("# Interactive Extracting Object from Image")
    with gr.Row():
        coord_h = gr.Number(label="Mouse coords h")
        coord_w = gr.Number(label="Mouse coords w")

    with gr.Row():
        input_img = gr.Image(label="Input image").style(height=600)
        output_img = gr.Image(label="Output image").style(height=600)

    input_img.select(extract_object_by_event, [input_img], output_img)
    input_img.select(get_coords, None, [coord_h, coord_w])

    gr.Markdown("## Image Examples")
    gr.Examples(
        examples=[
            [os.path.join(os.path.dirname(__file__), "examples/dog.jpg"), 1013, 786, 1]
        ],
        inputs=[input_img, coord_h, coord_w],
        outputs=output_img,
        fn=extract_object,
        run_on_click=True,
    )

if __name__ == "__main__":
    demo.launch()
