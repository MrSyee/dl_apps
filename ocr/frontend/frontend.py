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

CANVAS_WIDTH = 600
CANVAS_HEIGHT = 192
os.makedirs("inputs", exist_ok=True)

# Set title
st.title("Handwritten image to Text")
st.markdown("Try to write some sentences!")

# Set canvas
CanvasResult = st_canvas(
    fill_color="#FFFFFF",
    stroke_width=3,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=CANVAS_WIDTH,
    height=CANVAS_HEIGHT,
    drawing_mode="freedraw",
    key="canvas",
)

# Get handwritten text image fron canvas
if CanvasResult.image_data is not None:
    img_array = CanvasResult.image_data.astype("uint8")

text_value = ""
if st.button("Predict"):
    try:
        image = Image.fromarray(img_array)
        image.save("inputs/canvas.png", format="PNG")

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        image_bytes = buffered.getvalue()

        files = {"file": image_bytes}

        # Request to API
        response_predict = requests.post(
            url=PREDICT_URL,
            files=files,
        )

        # Get text
        res = response_predict.json()
        text_value = res['text']
        st.write(f"Prediction: {res['text']}")

    except ConnectionError:
        st.write("Couldn't reach backend")

# Set text area
st.text_area("Result", text_value)
