import os

import gradio as gr
import whisper
from pytube import YouTube
from whisper.utils import get_writer

output_dir = "outputs"
whisper_writer = get_writer(output_format="srt", output_dir=output_dir)
writer_args = {
    "highlight_words": False,
    "max_line_count": None,
    "max_line_width": None,
}
model = whisper.load_model("large")
print("[INFO] Initialize model.")


def speech_to_text(audio_file_path: str, filename: str) -> str:
    transcript = model.transcribe(audio_file_path)
    whisper_writer(transcript, audio_file_path, writer_args)

    return os.path.join(output_dir, f"{filename}.srt")


def transcribe(link: str):
    video_file_name = "video_from_youtube.mp4"
    audio_file_name = "audio_from_youtube.webm"
    youtube_audio_path = os.path.join(output_dir, audio_file_name)

    yt = YouTube(link)
    _ = (
        yt.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
        .download(output_path=output_dir, filename=video_file_name)
    )
    audio_streams = yt.streams.filter(only_audio=True)
    for audio_stream in audio_streams:
        if audio_stream.mime_type == "audio/webm" and audio_stream.abr == "160kbps":
            audio_streams[-1].download(output_path=output_dir, filename=audio_file_name)

    transcript_file = speech_to_text(youtube_audio_path, yt.title)
    return transcript_file, [os.path.join(output_dir, video_file_name), transcript_file]


# Set gradio app
with gr.Blocks() as demo:
    gr.Markdown("# Speech to Text")

    with gr.Row():
        with gr.Column(scale=1):
            link = gr.Textbox(label="Youtube Link")
            subtile = gr.File(label="Subtitle", file_types=[".srt"])
            submit_btn = gr.Button(value="Transcibe!")

        with gr.Column(scale=4):
            output_video = gr.Video(label="Output", height=500)

    submit_btn.click(transcribe, [link], [subtile, output_video])


if __name__ == "__main__":
    demo.launch(inline=False, share=True)
