import os
import urllib

import cv2
import torch
import gradio as gr
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image

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
        masks, _, _ = self.predictor.predict(point_coords, points_labels)
        merged_mask = np.logical_or.reduce(masks, axis=0)
        return merged_mask

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


with gr.Blocks() as demo:
    gr.Markdown("# Interactive Extracting Object from Image")
    coords = gr.Textbox(label="Mouse coords")

    with gr.Row():
        input_img = gr.Image(label="Input image").style(height=1000, width=1000)
        output_img = gr.Image(label="Output image").style(height=1000, width=1000)

    extract_btn = gr.Button("Extract")

    def extract_object_by_click(image: np.ndarray, evt: gr.SelectData):
        print("image: ", image.shape)
        image_pil = Image.fromarray(image).convert("RGB")
        image_pil.save("inputs/origin.png", format="PNG")
        click_h, click_w = evt.index

        point_coords = np.array([[click_h, click_w]])
        points_labels = np.array([1])
        mask = inferencer.inference(image, point_coords, points_labels)
        print("mask: ", mask.shape)
        output_image, contours = draw_contour(image, mask)
        print("output_image: ", output_image.shape)
        output_image_pil = Image.fromarray(output_image).convert("RGB")
        output_image_pil.save("inputs/contour.png", format="PNG")

        # Extract object
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, contours, (255, 255, 255))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        result = cv2.bitwise_and(image, image, mask=mask)

        return result

    def get_coords(evt: gr.SelectData):
        return f"(x, y): ({evt.index[1]}, {evt.index[0]})"

    input_img.select(extract_object_by_click, [input_img], output_img)
    input_img.select(get_coords, None, coords)

if __name__ == "__main__":
    demo.launch()
