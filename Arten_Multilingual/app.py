import os
import time
import torch
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
import whisper
import librosa

# Load environment variables from .env file
load_dotenv('../API.env')

app = FastAPI()

# Determine the device to use
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Whisper model for English with appropriate device
english_model = whisper.load_model("small", device=device)

# Load Whisper model for Indonesian
indo_model = WhisperForConditionalGeneration.from_pretrained("tonimurfid/whisper-small-id")
indo_processor = WhisperProcessor.from_pretrained("tonimurfid/whisper-small-id")

# Set up the pipeline as in Gradio
indo_pipeline = pipeline(model="tonimurfid/whisper-small-id", device=0 if device == 'cuda' else -1)

# Retrieve the API key from environment variables
API_KEY = os.getenv('API_KEY')

def authenticate_api_key(x_api_key: str):
    if x_api_key and x_api_key == API_KEY:
        return True
    else:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/eng/transcribe")
async def transcribe_audio_eng(file: UploadFile = File(...), x_api_key: str = Header(...)):
    authenticate_api_key(x_api_key)

    if not file:
        raise HTTPException(status_code=400, detail="No file part")

    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    # Save the file temporarily
    filepath = os.path.join("/tmp", file.filename)
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    # Get the size of the file
    file_size = os.path.getsize(filepath)

    transcribe_start = time.time()
    # Process the file with Whisper English model and set language to English
    result = english_model.transcribe(filepath, language='en')
    transcribe_end = time.time()
    os.remove(filepath)  # Remove the file after processing

    response = {
        "transcription": result["text"],
        "stats": {
            "total_processing_time": transcribe_end - transcribe_start,
            "words_per_second": round(len(result["text"]) / (transcribe_end - transcribe_start), 2),
            "file_size_in_bytes": file_size
        },
        "filename": file.filename,
    }

    return JSONResponse(content=response)

@app.post("/id/transcribe")
async def transcribe_audio_indo(file: UploadFile = File(...), x_api_key: str = Header(...)):
    authenticate_api_key(x_api_key)

    if not file:
        raise HTTPException(status_code=400, detail="No file part")

    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    # Save the file temporarily
    filepath = os.path.join("/tmp", file.filename)
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    # Get the size of the file
    file_size = os.path.getsize(filepath)

    transcribe_start = time.time()

    # Load the audio file
    audio, sr = librosa.load(filepath, sr=16000)

    # Use the pipeline to transcribe the audio
    transcription = indo_pipeline(audio)["text"]
    
    transcribe_end = time.time()

    os.remove(filepath)  # Remove the file after processing

    if '<|startoftranscript|><|id|><|transcribe|><|notimestamps|>' in transcription:
        transcription = transcription.replace('<|startoftranscript|><|id|><|transcribe|><|notimestamps|>', '')
    response = {
        "transcription": transcription,
        "stats": {
            "total_processing_time": transcribe_end - transcribe_start,
            "words_per_second": round(len(transcription) / (transcribe_end - transcribe_start), 2),
            "file_size_in_bytes": file_size
        },
        "filename": file.filename,
    }

    return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
