import json
import os
import urllib

import requests
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import numpy as np

if os.environ.get("BACKEND_URL") is not None:
    BACKEND_URL = str(os.environ.get("BACKEND_URL"))
else:
    BACKEND_URL = "http://localhost:8888/"
PREDICT_URL = urllib.parse.urljoin(BACKEND_URL, "infer")


st.title("Handwritten image to Text")
st.markdown("Try to write some sentences!")

CANVAS_WIDTH = 600
CANVAS_HEIGHT = 192
CanvasResult = st_canvas(
    fill_color="#000000",
    stroke_width=5,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=CANVAS_WIDTH,
    height=CANVAS_HEIGHT,
    drawing_mode="freedraw",
    key="canvas",
)

if CanvasResult.image_data is not None:
    img_array = CanvasResult.image_data.astype("uint8")


image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
if image_file is not None:
    # To See details
    file_details = {
        "filename": image_file.name,
        "filetype": image_file.type,
        "filesize": image_file.size,
    }
    st.write(file_details)

    # Send image to backend
    files = {"file": image_file.getvalue()}

text_value = ""
if st.button("Predict"):
    try:
        # image_byte = img_array.tobytes()
        # frame_file = io.BytesIO(image_byte)

        image = Image.fromarray(img_array)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        image_bytes = buffered.getvalue()

        files = {"file": image_bytes}
        print("files: ", files)


        print("PREDICT_URL: ", PREDICT_URL)
        response_predict = requests.post(
            url=PREDICT_URL,
            files=files,
        )

        print("response_predict: ", response_predict)
        res = response_predict.json()
        print("res: ", res)
        text_value = res['text']
        st.write(f"Prediction: {res['text']}")

    except ConnectionError:
        st.write("Couldn't reach backend")

st.text_area("Result", text_value)
