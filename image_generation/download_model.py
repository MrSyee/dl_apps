import os

import requests


ROOT_URL = "https://civitai.com"

def download_model(url: str):
    parsed = url.replace(f"{ROOT_URL}/", "").split("?modelVersionId=")
    model_id = parsed[0].replace("models/", "").split("/")[0]

    try:
        response = requests.get(
            f"{ROOT_URL}/api/v1/models/{model_id}", stream=True, timeout=600
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print(list(response.json().keys()))
    print(response.json()["name"])
    print(list(response.json()["modelVersions"][0].keys()))
    print(response.json()["modelVersions"][0]["name"])
    download_url = response.json()["modelVersions"][0]["downloadUrl"]
    filename = response.json()["modelVersions"][0]["files"][0]["name"]

    file_path = f"models/{filename}"
    if os.path.exists(file_path):
        print(f"[INFO] File already exists: {file_path}")
        return

    os.makedirs("models", exist_ok=True)
    os.system(f"wget -O '{file_path}' '{download_url}'")
    print(f"[INFO] File downloaded: {file_path}")

if __name__ == "__main__":
    url = "https://civitai.com/models/66/anything-v3"
    download_model(url)