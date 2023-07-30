import gradio as gr
import openai
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup

load_dotenv()


def crawl(url: str):
    rep = requests.get(url)

    soup = BeautifulSoup(rep.content, 'html.parser')

    # Get title
    # article = soup.find('h1').get_text()
    # Get main contents
    article = ""
    for paragraph in soup.find_all(['p', 'h2'], {"class":['paragraph', "subheader"]}):
        article += paragraph.text.strip()

    return article

# max 토큰 지정
# 날짜를 지정 -> 그 날의 헤드라인 5개 기사 뽑기 -> 요약 -> 번역
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
        print(response)
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
    url = gr.Textbox(label="URL")
    crawl_btn = gr.Button("Crawl!")

    gr.Markdown("## 원문 기사")
    original_box = gr.Textbox(label="Original article", lines=7)
    crawl_btn.click(crawl, inputs=[url], outputs=[original_box])
    summarize_btn = gr.Button("Summarize!")

    gr.Markdown("## 원문 기사 요약")
    abstract_box = gr.Textbox(label="Summarized article", lines=7)
    summarize_btn.click(gpt_inferencer.summarize, inputs=[original_box], outputs=[abstract_box])

    gr.Markdown("## 요약문 한국어 번역")
    translate_btn = gr.Button("Translate!")
    translate_box = gr.Textbox(label="Translated article", lines=7)
    translate_btn.click(gpt_inferencer.translate, inputs=[abstract_box], outputs=[translate_box])

app.launch(inline=False, share=True)
