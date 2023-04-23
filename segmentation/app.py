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


def extract_object(
    image: np.ndarray, point_h: int, point_w: int, point_label: int
):
    point_coords = np.array([[point_h, point_w]])
    point_label = np.array([point_label])
    # image_pil = Image.fromarray(image).convert("RGB")
    # image_pil.save("inputs/origin.png", format="PNG")

    # Get mask
    mask = inferencer.inference(image, point_coords, point_label)
    overlay_image, contours = draw_contour(image, mask)
    # overlay_image_pil = Image.fromarray(overlay_image).convert("RGB")
    # overlay_image_pil.save("inputs/contour.png", format="PNG")

    # Extract object
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, contours, (255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    return segmented_image

def extract_object_by_event(image: np.ndarray, evt: gr.SelectData):
    click_h, click_w = evt.index

    return extract_object(image, click_h, click_w, 1)


def get_coords(evt: gr.SelectData):
    return f"(h, w): ({evt.index[0]}, {evt.index[1]})"


with gr.Blocks() as demo:
    gr.Markdown("# Interactive Extracting Object from Image")
    coords = gr.Textbox(label="Mouse coords")
    with gr.Row():
        coord_h = gr.Number(label="Mouse coords h")
        coord_w = gr.Number(label="Mouse coords w")
        click_label = gr.Number(label="label")

    with gr.Row():
        input_img = gr.Image(label="Input image").style(height=1000, width=1000)
        output_img = gr.Image(label="Output image").style(height=1000, width=1000)

    extract_btn = gr.Button("Extract")

    input_img.select(extract_object_by_event, [input_img], output_img)
    input_img.select(get_coords, None, coords)



    gr.Markdown("## Image Examples")
    gr.Examples(
        examples=[
            [os.path.join(os.path.dirname(__file__), "examples/dog.jpg"), 1013, 786, 1]
        ],
        inputs=[input_img, coord_h, coord_w, click_label],
        outputs=output_img,
        fn=extract_object,
        run_on_click=True,
    )

if __name__ == "__main__":
    demo.launch()
