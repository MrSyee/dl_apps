import os
from pathlib import Path

import gradio as gr
import openai
from dotenv import load_dotenv
from pytube import YouTube

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# with open("examples/example.wav", "rb") as audio_file:
#     transcript = openai.Audio.transcribe("whisper-1", audio_file)

# print(type(transcript))
# print(transcript["text"])
def speech_to_text(audio_file_path):
    print(type(audio_file_path), audio_file_path)
    with open(audio_file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]


def transcribe(link: str):
    output_path = Path("outputs")
    audio_file_name = "audio_from_youtube.webm"
    youtube_audio_path = Path(output_path / audio_file_name)

    yt = YouTube(link)
    audio_streams = yt.streams.filter(only_audio=True)

    for audio_stream in audio_streams:
        if audio_stream.mime_type == "audio/webm" and audio_stream.abr == "160kbps":
            audio_streams[-1].download(output_path="outputs", filename=audio_file_name)

    return speech_to_text(youtube_audio_path)


# Set gradio app
with gr.Blocks() as demo:
    gr.Markdown("# Speech to Text")
    with gr.Tab("Mic"):
        record = gr.Audio(source="microphone", type="filepath", streaming=True)
        output = gr.Textbox(label="Output", lines=10)
        clear = gr.ClearButton([record, output])

        record.stop_recording(speech_to_text, [record], [output])

    with gr.Tab("Youtube"):
        link = gr.Textbox(label="Youtube Link")
        submit_btn = gr.Button("Submit")
        output = gr.Textbox(label="Output", lines=10)
        clear = gr.ClearButton([record, output])

        submit_btn.click(transcribe, [link], [output])


if __name__ == "__main__":
    demo.launch()
