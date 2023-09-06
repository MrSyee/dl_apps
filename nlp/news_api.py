"""Retriving News Chatbot with GPT & News API

- What is the latest issues?
- What are the popular issues for September 2023?
- What is Tesla up these days?

- [ ]: Argument 표시
- [ ]: 스크래핑 붙이기
"""

import os
import requests
import json

import openai
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# 모델 - GPT 3.5 Turbo 선택
model = "gpt-3.5-turbo-0613"
openai.api_key = os.environ["OPENAI_API_KEY"]
news_api_key = os.environ["NEWS_API_KEY"]

max_num_articles = 5

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
    print("query", query)
    print("from_date", from_date)
    print("to_date", to_date)
    print("sort_by", sort_by)
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
            + f"But, max number is {max_num_articles}."
        )
        return json.dumps(data['articles'][:min(max_num_articles, len(data['articles']))])
    else:
        print("Request failed with message:", data['message'])
        return 'No articles found'


functions = [
    {
        "name": "get_articles",
        "description": "Get top news headlines by country and/or category",
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
]


def respond(message: str, chat_history: list, function_call: str = "auto"):
    # 메시지 설정하기
    input_msg = [
        {"role": "user", "content": message},
    ]

    # Get args from prompt
    res = openai.ChatCompletion.create(
        model=model,
        messages=input_msg,
        functions=functions,
        function_call=function_call
    )
    response = res["choices"][0]["message"]
    print("First response", response)

    # Function call message
    input_msg = [
        {"role": "user", "content": message},
    ]
    input_msg.append(response)

    # call functions requested by the model
    if response.get("function_call"):
        function_name = response["function_call"]["name"]
        if function_name == "get_articles":
            llm_system_prompt = """
                You are an assistant that provides news and headlines to user requests.
                Always try to get the lastest breaking stories using the available function calls.
                Please output something like this:
                Number. [Title](Article Link)\n
                    - Description: description\n
                    - Publish Date: publish date\n
            """
            input_msg.append({"role": "system", "content": llm_system_prompt})

            args = json.loads(response["function_call"]["arguments"])
            headlines = get_articles(
                query=args.get("query"),
                from_date=args.get("from_date"),
                to_date=args.get("to_date"),
                sort_by=args.get("sort_by"),
            )
            print("headlines", headlines)
            input_msg.append({ "role": "function", "name": function_name, "content": headlines})

        print("input_msg", input_msg)
        res = openai.ChatCompletion.create(
            model=model,
            messages=input_msg,
        )

        print("res", res)
        response = res["choices"][0]["message"]["content"].strip()
        print("second response", response)
    else:
        response = response["content"]

    chat_history.append((message, response))
    # time.sleep(2)
    return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()