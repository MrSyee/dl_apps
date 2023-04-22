import gradio as gr

with gr.Blocks() as demo:
    coords = gr.Textbox(label="Mouse coords")
    with gr.Row():
        input_img = gr.Image(label="Input")

    def get_coords(evt: gr.SelectData):
        return f"({evt.index[1]}, {evt.index[0]})"

    input_img.select(get_coords, None, coords)

if __name__ == "__main__":
    demo.launch()
