import whisper
from pytubefix import YouTube
import time

start_time = time.time()  

YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=_uk_6vfqwTA"
youtube = YouTube(YOUTUBE_VIDEO)
audio = youtube.streams.filter(only_audio=True).first()
file_path = audio.download()

whisper_model = whisper.load_model("base")
transcription = whisper_model.transcribe(file_path, fp16=True)["text"].strip()

with open("transcription2.txt", "w") as file:
    file.write(transcription)

end_time = time.time() 
execution_time = end_time - start_time
print(f"Waktu eksekusi: {execution_time:.2f} detik")