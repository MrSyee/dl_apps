import json
import os
import urllib

import requests
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

if os.environ.get("BACKEND_URL") is not None:
    BACKEND_URL = str(os.environ.get("BACKEND_URL"))
else:
    BACKEND_URL = "http://localhost:8888"
PREDICT_URL = urllib.parse.urljoin(BACKEND_URL, "predict")


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

text_value = ""
if st.button("Predict"):
    try:
        response_predict = requests.post(
            url=PREDICT_URL,
            data=json.dumps({"image": img_array.tolist()}),
        )

        print("response_predict: ", response_predict)
        res = response_predict.json()
        print("res: ", res)
        text_value = res['text']
        st.write(f"Prediction: {res['text']}")

    except ConnectionError:
        st.write("Couldn't reach backend")

st.text_area("Result", text_value)
