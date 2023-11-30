import os

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    DiffusionPipeline,
    EulerDiscreteScheduler,
    StableDiffusionControlNetPipeline,
)
from diffusers.utils import load_image
from PIL import Image

os.makedirs("outputs", exist_ok=True)


def init_pipeline() -> DiffusionPipeline:
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    return pipe


pipe = init_pipeline()

# image = load_image(
#     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
# )

# image = np.array(image)

# low_threshold = 100
# high_threshold = 200

# image = cv2.Canny(image, low_threshold, high_threshold)
# image = image[:, :, None]
# image = np.concatenate([image, image, image], axis=2)
# canny_image = Image.fromarray(image)


def generate_image(sketch_image: Image, prompt: str = "", negative_prompt: str = ""):
    image = sketch_image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    sketch_image = Image.fromarray(image)
    sketch_image.save("outputs/sketch_image.png")

    outputs = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=sketch_image,
        num_images_per_prompt=4,
        num_inference_steps=30,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ).images
    outputs[0].save("outputs/output.png")
    return outputs


print("[INFO] Gradio app ready")
with gr.Blocks() as demo:
    gr.Markdown("# Sketch to Cartoon Image")

    with gr.Row():
        with gr.Column():
            # sketchpad = gr.Sketchpad(
            #     label="Handwritten Sketchpad",
            #     shape=(600, 192),
            #     brush_radius=2,
            #     invert_colors=False,
            # )
            sketchpad = gr.Image(
                label="Draw sketch",
                image_mode="L",
                source="canvas",
                interactive=True,
                height=512,
                shape=(512, 512),
                brush_radius=2,
                invert_colors=False,
            )
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(
                    label="Images", minimum=1, maximum=12, value=1, step=1
                )
                image_resolution = gr.Slider(
                    label="Image Resolution",
                    minimum=256,
                    maximum=768,
                    value=512,
                    step=64,
                )
                strength = gr.Slider(
                    label="Control Strength",
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.01,
                )
                guess_mode = gr.Checkbox(label="Guess Mode", value=False)
                ddim_steps = gr.Slider(
                    label="Steps", minimum=1, maximum=100, value=20, step=1
                )
                scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=0.1,
                    maximum=30.0,
                    value=9.0,
                    step=0.1,
                )
                seed = gr.Slider(
                    label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True
                )
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(
                    label="Added Prompt", value="best quality, extremely detailed"
                )
                n_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                )
        with gr.Column():
            result_gallery = gr.Gallery(
                label="Output", elem_id="gallery", rows=2, height="auto"
            )

        run_button.click(
            fn=generate_image,
            inputs=[sketchpad, prompt, n_prompt],
            outputs=[result_gallery],
        )

if __name__ == "__main__":
    demo.launch(inline=False, share=True)
