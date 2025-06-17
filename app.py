from fastapi import FastAPI
from pydantic import BaseModel
from whisper import load_model
import yt_dlp, os
from fuzzywuzzy import fuzz
from quran_uthmani import Quran

app = FastAPI()
quran = Quran()
model = load_model("medium")

class ProcessRequest(BaseModel):
    youtube_url: str

class AyahSegment(BaseModel):
    start: float
    end: float
    surah: int
    ayah: int
    ayah_text: str
    score: int

class ProcessResponse(BaseModel):
    audio_url: str
    ayah_segments: list[AyahSegment]

def download_audio(url):
    os.makedirs("static/audio", exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "static/audio/audio.%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio","preferredcodec": "wav"}],
        "noplaylist": True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)
    return "static/audio/audio.wav"

@app.post("/process", response_model=ProcessResponse)
async def process(req: ProcessRequest):
    audio_path = download_audio(req.youtube_url)
    transcription = model.transcribe(audio_path, language="ar")
    segments = transcription["segments"]
    ayah_segments = []

    for seg in segments:
        best = {"score": 0}
        for s in range(1, 115):
            for a in range(1, quran.surah_length(s) + 1):
                text = quran.get_ayah(s, a)
                score = fuzz.partial_ratio(seg["text"], text)
                if score > best["score"]:
                    best = {"surah": s, "ayah": a, "ayah_text": text, "score": score}
        if best["score"] > 85:
            ayah_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                **best
            })

    return {
        "audio_url": "http://localhost:8000/static/audio/audio.wav",
        "ayah_segments": ayah_segments
    }
