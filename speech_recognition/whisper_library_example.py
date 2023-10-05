import pandas as pd
import torch
import whisper
from whisper.utils import get_writer

audio_path = "examples/example2.wav"

writer = get_writer(output_format="srt", output_dir="outputs")
writer_args = {
    "highlight_words": False,
    "max_line_count": None,
    "max_line_width": None,
}

model = whisper.load_model("large")
print(model.device)
result = model.transcribe(audio_path)

print(result)
print(result["text"])

writer(result, audio_path, writer_args)
