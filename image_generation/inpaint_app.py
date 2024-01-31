import gradio as gr
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()


def inpaint_from_mask(image_dict: Image, prompt: str):
    image_dict["image"].save("./outputs/image.png")
    image_dict["mask"].save("./outputs/mask.png")

    images = pipeline(
        prompt=prompt,
        image=image_dict["image"],
        mask_image=image_dict["mask"],
        num_images_per_prompt=4,
    ).images

    images[0].save("./outputs/output.png")
    return images


with gr.Blocks() as app:
    gr.Markdown("# AI 이미지 에디터")
    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            lines=3,
            value="a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k",
        )
    with gr.Row():
        with gr.Tab("Mask"):
            input_img = gr.Image(
                label="Input image",
                height=600,
                type="pil",
                tool="sketch",
                source="upload",
                brush_radius=100,
            )
            inpaint_btn = gr.Button(value="Inpaint!")

    with gr.Row():
        gallary = gr.Gallery(rows=1)

    inpaint_btn.click(inpaint_from_mask, [input_img, prompt], [gallary])

app.launch(inline=False, share=True)
