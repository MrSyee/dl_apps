import os
import time
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

def get_top_headlines(query: str = None, country: str = None, category: str = None):
    """Retrieve top headlines from newsapi.org (API key required)"""

    base_url = "https://newsapi.org/v2/top-headlines"
    headers = {
        "x-api-key": news_api_key
    }
    params = {
        # "sources": "cnn",
    }
    print("query", query)
    print("country", country)
    print("category", category)
    if query is not None:
        params['q'] = query
    if country is not None:
        params['country'] = country
    if category is not None:
        params['category'] = category

    # Fetch from newsapi.org
    # reference: https://newsapi.org/docs/endpoints/top-headlines
    response = requests.get(base_url, params=params, headers=headers)
    data = response.json()

    if data['status'] == 'ok':
        print(
            f"Processing {data['totalResults']} articles from newsapi.org. "
            + f"But, max number is {max_num_articles}"
        )
        return json.dumps(data['articles'][:min(max_num_articles, len(data['articles']))])
    else:
        print("Request failed with message:", data['message'])
        return 'No articles found'


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)



signature_get_top_headlines = {
    "name": "get_top_headlines",
    "description": "Get top news headlines by country and/or category",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Freeform keywords or a phrase to search for.",
            },
            "country": {
                "type": "string",
                "description": "The 2-letter ISO 3166-1 code of the country you want to get headlines for",
            },
            "category": {
                "type": "string",
                "description": "The category you want to get headlines for",
                "enum": ["business","entertainment","general","health","science","sports","technology"]
            }
        },
        "required": [],
    }
}

signature_get_current_weather = {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
}

def respond(message: str, chat_history: list, function_call: str = "auto"):
    # 메시지 설정하기
    input_msg = [
        {"role": "user", "content": message},
    ]

    # Get args from prompt
    res = openai.ChatCompletion.create(
        model=model,
        messages=input_msg,
        functions=[signature_get_top_headlines, signature_get_current_weather],
        function_call=function_call
    )
    response = res["choices"][0]["message"]
    print("First response", response)

    # Function call message
    llm_system_prompt = """
        You are an assistant that provides news and headlines to user requests.
        Always try to get the lastest breaking stories using the available function calls.
        Please output something like this:
        Number. [Title](Article Link)\n
            - Description: description\n
            - Publish Date: publish date\n
    """
    input_msg = [
        {"role": "user", "content": message},
    ]
    input_msg.append(response)

    # call functions requested by the model
    if response.get("function_call"):
        function_name = response["function_call"]["name"]
        if function_name == "get_top_headlines":
            input_msg.append({"role": "system", "content": llm_system_prompt})

            args = json.loads(response["function_call"]["arguments"])
            headlines = get_top_headlines(
                query=args.get("query"),
                country=args.get("country"),
                category=args.get("category")
            )
            print("headlines", len(headlines), headlines)
            input_msg.append({ "role": "function", "name": function_name, "content": headlines})

        elif function_name == "get_current_weather":
            function_args = json.loads(response["function_call"]["arguments"])
            function_response = get_current_weather(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            input_msg.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )

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