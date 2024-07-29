from flask import Flask, request, jsonify
import whisper
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Load Whisper model with CUDA
model = whisper.load_model("medium", device="cuda")

# Retrieve the API key from environment variables
API_KEY = '00jhJ_YAsU90J3hpG-vNI1I9QBm_Voefj8NMcR-OEY8'

def authenticate_api_key(request):
    api_key = request.headers.get('x-api-key')
    return api_key == API_KEY

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if not authenticate_api_key(request):
        return jsonify({"error": "Unauthorized"}), 401

    start = time.time()
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save the file temporarily
        filepath = os.path.join("/tmp", file.filename)
        file.save(filepath)

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

        return jsonify(response)

    return jsonify({"error": "Invalid request"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
