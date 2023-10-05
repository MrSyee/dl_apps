import os
from pathlib import Path

import openai
from dotenv import load_dotenv
from pytube import YouTube

load_dotenv()

output_path = Path("outputs")
audio_file_name = "audio_from_youtube.webm"
youtube_audio_path = Path(output_path / audio_file_name)

# youtube_link = "https://youtu.be/dTn_EINLuwo?si=_iFFWPznSjkCxmxP"
youtube_link = "https://youtu.be/QfmYJW4Y0C4?si=peZ05Qka7gr2QTJL"  # chimchak
# youtube_link = "https://youtu.be/d14cQHBtZc4?si=zCLsCFXgU5uUezHu"
yt = YouTube(youtube_link)

print(yt.title)

audio_streams = yt.streams.filter(only_audio=True)

for audio_stream in audio_streams:
    if audio_stream.mime_type == "audio/webm" and audio_stream.abr == "160kbps":
        audio_streams[-1].download(output_path="outputs", filename=audio_file_name)

print(len(audio_streams))
for a in audio_streams:
    print(a)


# Whisper API
openai.api_key = os.getenv("OPENAI_API_KEY")

audio_list = sorted(list(output_path.glob("*.mp4")))
print(audio_list)

# with open(youtube_audio_path, "rb") as audio_file:
#     transcript = openai.Audio.transcribe("whisper-1", audio_file)
#     print(transcript["text"])

# text = transcript["text"]
# with open("outputs/result.txt", "w") as f:
#     f.write(text)
