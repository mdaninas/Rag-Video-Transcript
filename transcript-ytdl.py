import whisper
import subprocess
import time

start_time = time.time()  
# Link video
video_url = "https://www.youtube.com/watch?v=_uk_6vfqwTA"

# Unduh audio menggunakan yt-dlp
subprocess.run([
    "yt-dlp",
    "-f", "bestaudio",
    "-o", "audio.%(ext)s",
    "--extract-audio",
    "--audio-format", "mp3",
    video_url
])

# Transkripsi dengan Whisper
model = whisper.load_model("base")
result = model.transcribe("audio.mp3", fp16=True) 

# Simpan ke file
with open("transcription.txt", "w", encoding="utf-8") as f:
    f.write(result["text"].strip())

end_time = time.time() 
execution_time = end_time - start_time
print(f"Waktu eksekusi: {execution_time:.2f} detik")