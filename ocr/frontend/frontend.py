import json
import os
import urllib

import pyperclip
import requests
import streamlit as st
from streamlit_drawable_canvas import st_canvas

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
    img = CanvasResult.image_data.astype("uint8")

text_value = ""
if st.button("Predict"):
    try:
        text_value = "Predict!"
        # response_predict = requests.post(
        #     url=PREDICT_URL,
        #     data=json.dumps({"image": img.tolist()}),
        # )

        # if response_predict.ok:
        #     res = response_predict.json()
        #     st.write(f"Prediction: {res['label']}")
        # else:
        #     st.write("Some error occured")

    except ConnectionError:
        st.write("Couldn't reach backend")
st.text_area("Result", text_value)
