import os

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

with open("examples/example.wav", "rb") as audio_file:
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

print(type(transcript))
print(transcript["text"])
