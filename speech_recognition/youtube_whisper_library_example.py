import os

import whisper
from pytube import YouTube
from whisper.utils import get_writer

output_dir = "outputs"

# youtube_link = "https://youtu.be/dTn_EINLuwo?si=_iFFWPznSjkCxmxP"
youtube_link = "https://youtu.be/QfmYJW4Y0C4?si=peZ05Qka7gr2QTJL"  # chimchak
# youtube_link = "https://youtu.be/d14cQHBtZc4?si=zCLsCFXgU5uUezHu"
yt = YouTube(youtube_link)
audio_file_name = f"{yt.title}.webm"
youtube_audio_path = os.path.join(output_dir, audio_file_name)

print(yt.title)
print(youtube_audio_path)

audio_streams = yt.streams.filter(only_audio=True)
for audio_stream in audio_streams:
    if audio_stream.mime_type == "audio/webm" and audio_stream.abr == "160kbps":
        audio_streams[-1].download(output_path="outputs", filename=audio_file_name)


writer = get_writer(output_format="srt", output_dir="outputs")
writer_args = {
    "highlight_words": False,
    "max_line_count": None,
    "max_line_width": None,
}

model = whisper.load_model("large")
result = model.transcribe(youtube_audio_path, verbose=True)

print(result["text"])

writer(result, youtube_audio_path, writer_args)
