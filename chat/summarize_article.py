import gradio as gr
import openai
from dotenv import load_dotenv
import os

load_dotenv()

example_text = """
The Ukrainian military is doubling down on efforts to break through thick Russian defenses in its counteroffensive in the south, which has struggled to gain momentum since being launched at the beginning of June.
Ukrainian officials have said little about what fresh units are being committed to the offensive, but the military has clearly added recently-minted units equipped with western armor in at least one important segment of the southern front.
The challenges faced by the Ukrainians are perhaps less to do with numbers and more to do with capabilities, training and coordination, factors that are critical when an attacking force is faced with such an array of defenses.
Fragments of geolocated video show that western armor such as Bradley fighting vehicles have been part of the renewed assault and that experienced units have been brought into the fray. But tight operational security on the part of the Ukrainians precludes a full assessment of what is being done to reboot the counteroffensive – and where.
There's still debate about the size of the additional effort.
George Barros of the Institute for the Study of War – a Washington-based group – told CNN: “We had not seen any evidence of a battalion-level attack and certainly no brigade-level attacks. If the Ukrainians are indeed committing full battalions and brigades now as reported, that would mark a clear new phase of the Ukrainian counteroffensive.”
"""

class GPTInferencer:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "gpt-3.5-turbo"

    def summarize(self, texts: str):
        query = f"""
            Summarize the sentences below.
            ---
            {texts}
            """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]

        # ChatGPT API 호출하기
        response = openai.ChatCompletion.create(model=self.model, messages=messages)
        return response["choices"][0]["message"]["content"]

    def translate(self, texts: str):
        query = f"""
            Translate the sentences below into Korean.
            ---
            {texts}
            """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]

        # ChatGPT API 호출하기
        response = openai.ChatCompletion.create(model=self.model, messages=messages)
        return response["choices"][0]["message"]["content"]

gpt_inferencer = GPTInferencer()

with gr.Blocks() as app:
    gr.Markdown("# Summarize article & Translate")
    gr.Markdown("## 요약할 기사의 주소를 입력하세요.")
    url = gr.Textbox(example_text, label="URL")
    abstract_btn = gr.Button("Abstract!")

    gr.Markdown("## 원문 기사 요약")
    abstract_box = gr.Textbox(label="Abstract", lines=7)
    abstract_btn.click(gpt_inferencer.summarize, inputs=[url], outputs=[abstract_box])


    gr.Markdown("## 요약문 한국어 번역")
    translate_btn = gr.Button("Translate!")
    translate_box = gr.Textbox(label="Translate", lines=7)
    translate_btn.click(gpt_inferencer.translate, inputs=[abstract_box], outputs=[translate_box])

app.launch(inline=False, share=True)
