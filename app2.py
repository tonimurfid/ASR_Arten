import whisper
import os
import time
import torch
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Load Whisper model with appropriate device
model = whisper.load_model("small", device='cuda')

# Retrieve the API key from environment variables
API_KEY = '00jhJ_YAsU90J3hpG-vNI1I9QBm_Voefj8NMcR-OEY8'

def authenticate_api_key(x_api_key: str):
    if x_api_key and x_api_key == API_KEY:
        return True
    else:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), x_api_key: str = Header(...)):
    authenticate_api_key(x_api_key)

    start = time.time()

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
    # Process the file with Whisper
    result = model.transcribe(filepath)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
