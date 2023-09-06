"""Retriving News Chatbot with GPT & News API

- What is the latest issues?
- What are the popular issues for September 2023?
- What is Tesla up these days?

- [ ]: Argument 표시
- [ ]: 스크래핑 붙이기
"""

from typing import List, Dict, Any
import os
import requests
import json

import openai
from openai.openai_object import OpenAIObject
import gradio as gr
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

# 모델 - GPT 3.5 Turbo 선택
model = "gpt-3.5-turbo-0613"
openai.api_key = os.environ["OPENAI_API_KEY"]
news_api_key = os.environ["NEWS_API_KEY"]

MAX_NUM_ARTICLES = 5
TITLE_TO_URL = {}

def crawl(title: str):
    print(TITLE_TO_URL)
    url = TITLE_TO_URL[title]
    rep = requests.get(url)

    soup = BeautifulSoup(rep.content, "html.parser")

    # Get title
    article = soup.find("h1").get_text()
    # Get main contents
    article = ""
    for paragraph in soup.find_all(["p", "h2"], {"class": ["paragraph", "subheader"]}):
        article += paragraph.text.strip()

    summarized_article = gpt_inferencer.summarize(article)
    translated_article = gpt_inferencer.translate(summarized_article)

    return summarized_article, translated_article


def get_articles(query: str = None, from_date: str = None, to_date: str = None, sort_by: str = None):
    """Retrieve articles from newsapi.org (API key required)"""

    base_url = "https://newsapi.org/v2/everything"
    headers = {
        "x-api-key": news_api_key
    }
    params = {
        "sortBy": "publishedAt",
        "sources": "cnn",
        "language": "en",
    }

    if query is not None:
        params['q'] = query
    if from_date is not None:
        params['from'] = from_date
    if to_date is not None:
        params['to'] = to_date
    if sort_by is not None:
        params['sortBy'] = sort_by

    # Fetch from newsapi.org
    # reference: https://newsapi.org/docs/endpoints/top-headlines
    response = requests.get(base_url, params=params, headers=headers)
    data = response.json()

    if data['status'] == 'ok':
        print(
            f"Processing {data['totalResults']} articles from newsapi.org. "
            + f"Max number is {MAX_NUM_ARTICLES}."
        )
        return json.dumps(data['articles'][:min(MAX_NUM_ARTICLES, len(data['articles']))])
    else:
        print("Request failed with message:", data['message'])
        return 'No articles found'


func_get_articles = {
    "name": "get_articles",
    "description": "Get news articles",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Freeform keywords or a phrase to search for.",
            },
            "from_date": {
                "type": "string",
                "description": "A date and optional time for the oldest article allowed. This should be in ISO 8601 format",
            },
            "to_date": {
                "type": "string",
                "description": "A date and optional time for the newest article allowed. This should be in ISO 8601 format",
            },
            "sort_by": {
                "type": "string",
                "description": "The order to sort the articles in",
                "enum": ["relevancy","popularity","publishedAt"]
            }
        },
        "required": [],
    }
}

func_get_title_and_url = {
    "name": "get_title_and_url",
    "description": "Get title of article and url.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "array",
                "description": "title array of articles",
                "items": {
                    "type": "string",
                    "description": "title of article"
                }
            },
            "url": {
                "type": "array",
                "description": "url array of articles",
                "items": {
                    "type": "string",
                    "description": "url of article"
                }
            },
        },
        "required": ["title", "url"],
    }
}


class GPTInferencer:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "gpt-3.5-turbo-0613"

    def summarize(self, texts: str):
        prompt = f"""
            Summarize the sentences '---' below.
            ---
            {texts}
            """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # ChatGPT API 호출하기
        response = openai.ChatCompletion.create(model=self.model, messages=messages)

        input_tokens = response["usage"]["prompt_tokens"]
        output_tokens = response["usage"]["completion_tokens"]
        print(f"Input costs({input_tokens} tokens): ${input_tokens // 1000 * 0.0015}")
        print(f"Output costs({output_tokens} tokens): ${output_tokens // 1000 * 0.002}")
        return response["choices"][0]["message"]["content"]

    def translate(self, texts: str):
        prompt = f"""
            Translate the sentences '---' below to Korean.
            ---
            {texts}
            """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # ChatGPT API 호출하기
        response = openai.ChatCompletion.create(model=self.model, messages=messages)

        input_tokens = response["usage"]["prompt_tokens"]
        output_tokens = response["usage"]["completion_tokens"]
        print(f"Input costs({input_tokens} tokens): ${input_tokens // 1000 * 0.0015}")
        print(f"Output costs({output_tokens} tokens): ${output_tokens // 1000 * 0.002}")

        return response["choices"][0]["message"]["content"]

    def function_call(self, messages: List[Dict[str, str]], functions: List[Dict[str, Any]]) -> OpenAIObject:
        """
        If there is information for function in messages, get argument from messages.
        Otherwise get simple GPT response.
        """
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=functions,
            function_call="auto"
        )
        return response["choices"][0]["message"]


gpt_inferencer = GPTInferencer()

def respond(message: str, chat_history: list):
    global TITLE_TO_URL

    # Get args from prompt
    input_msg = [{"role": "user", "content": message}]
    args_resp = gpt_inferencer.function_call(input_msg, [func_get_articles])
    print("First response", type(args_resp), args_resp)

    # call functions requested by the model
    input_msg.append(args_resp)
    answer = args_resp["content"]
    title_list = []
    if args_resp.get("function_call"):
        function_name = args_resp["function_call"]["name"]
        get_articles_prompt = """
            You are an assistant that provides news and headlines to user requests.
            Always try to get the articles using the available function calls.
            Please output something like this:
            Number. [Title](Article Link)\n
                - Description: description\n
                - Publish Date: publish date\n
        """
        input_msg.append({"role": "system", "content": get_articles_prompt})

        # Run external function
        kwargs = json.loads(args_resp["function_call"]["arguments"])
        function_result = get_articles(**kwargs)
        input_msg.append({ "role": "function", "name": function_name, "content": function_result})

        # GPT inference include function result
        res = openai.ChatCompletion.create(
            model=model,
            messages=input_msg,
        )
        answer = res["choices"][0]["message"]["content"].strip()
        print("second response", answer)

        # Get titles and urls
        input_msg = [{ "role": "user", "content": answer }]
        args_resp = gpt_inferencer.function_call(input_msg, [func_get_title_and_url])
        args = json.loads(args_resp["function_call"]["arguments"])
        title_list, url_list = args.get("title"), args.get("url")
        TITLE_TO_URL = {title: url for title, url in zip(title_list, url_list)}
        print("Third response", args.get("title"), args.get("url"))

    chat_history.append((message, answer))

    # Update dropdown
    drop_down = None
    if title_list:
        drop_down = gr.update(choices=title_list, interactive=True)

    return "", chat_history, drop_down

def add_dropdown(x: str):
    return gr.update(choices=[x], interactive=True)


with gr.Blocks() as demo:
    gr.Markdown("# 뉴스 기사 탐색 챗봇")
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ## Chat
                얻고 싶은 정보에 대해 질문해보세요.
                """
            )
            chatbot = gr.Chatbot(label="Chat History")
            msg = gr.Textbox(label="Input prompt")
            clear = gr.ClearButton([msg, chatbot])

        with gr.Column():
            gr.Markdown(
                """
                ## Select News article
                원하는 기사를 선택하세요.
                """
            )
            article_list = gr.Dropdown(label="Article List", choices=None)
            abstract_box = gr.Textbox(label="Summarized article", lines=10, interactive=False)
            translate_box = gr.Textbox(label="Translated article", lines=10, interactive=False)
            crawl_btn = gr.Button("Get article!")
    msg.submit(respond, [msg, chatbot], [msg, chatbot, article_list])
    crawl_btn.click(crawl, inputs=[article_list], outputs=[abstract_box, translate_box])


if __name__ == "__main__":
    demo.launch()