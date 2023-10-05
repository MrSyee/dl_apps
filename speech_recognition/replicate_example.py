import os

import replicate
from dotenv import load_dotenv

load_dotenv()

# audio_path = "examples/example.wav"
audio_path = "outputs/46s_example.webm"
output = replicate.run(
    "stayallive/whisper-subtitles:b97ba81004e7132181864c885a76cae0e56bc61caa4190a395f6d8ba45b7a969",
    input={
        "audio_path": open(audio_path, "rb"),
        "language": "ko",
    },
)
print(output)
