import os

import requests
from tqdm import tqdm

ROOT_URL = "https://civitai.com"


def download_from_url(url: str, file_path: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(file_path, "wb") as file, tqdm(
        desc=file_path,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


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

    download_url = response.json()["modelVersions"][0]["downloadUrl"]
    filename = response.json()["modelVersions"][0]["files"][0]["name"]

    file_path = f"models/{filename}"
    if os.path.exists(file_path):
        print(f"[INFO] File already exists: {file_path}")
        return

    print(download_url)
    os.makedirs("models", exist_ok=True)
    print(f"[INFO] Download start!")
    download_from_url(download_url, file_path)
    print(f"[INFO] File downloaded: {file_path}")
    return file_path


if __name__ == "__main__":
    url = "https://civitai.com/models/66/anything-v3"
    download_model(url)
