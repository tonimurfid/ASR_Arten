import os
import time
import torch
import grpc
import whisper
import librosa
from concurrent import futures
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from dotenv import load_dotenv

# Import the generated classes
import transcribe_pb2 as pb2
import transcribe_pb2_grpc as pb2_grpc

# Load environment variables from .env file
load_dotenv('API.env')

# Determine the device to use
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Whisper models
english_model = whisper.load_model("small", device=device)
indo_model = WhisperForConditionalGeneration.from_pretrained("tonimurfid/whisper-small-id")
indo_processor = WhisperProcessor.from_pretrained("tonimurfid/whisper-small-id")
indo_pipeline = pipeline(model="tonimurfid/whisper-small-id", device=device)

class EnglishTranscriptionService(pb2_grpc.EnglishTranscriptionServiceServicer):
    def TranscribeAudio(self, request, context):
        print('transaudio-eng')
        file_data = request.file_data

        filepath = "/tmp/audio_eng.wav"
        with open(filepath, "wb") as f:
            f.write(file_data)

        transcribe_start = time.time()
        result = english_model.transcribe(filepath, language='en')
        transcribe_end = time.time()

        os.remove(filepath)

        response = pb2.TranscriptionResponse(
            transcription_message=result["text"]
        )
        return response

class IndonesianTranscriptionService(pb2_grpc.IndonesianTranscriptionServiceServicer):
    def TranscribeAudio(self, request, context):
        print('transaudio-id')
        file_data = request.file_data

        filepath = "/tmp/audio_indo.wav"
        with open(filepath, "wb") as f:
            f.write(file_data)

        audio, sr = librosa.load(filepath, sr=16000)
        transcription = indo_pipeline(audio)["text"]

        if '<|startoftranscript|><|id|><|transcribe|><|notimestamps|>' in transcription:
            transcription = transcription.replace('<|startoftranscript|><|id|><|transcribe|><|notimestamps|>', '')

        os.remove(filepath)

        response = pb2.TranscriptionResponse(
            transcription_message=transcription
        )
        return response

def serve():
    print('serve')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_EnglishTranscriptionServiceServicer_to_server(EnglishTranscriptionService(), server)
    pb2_grpc.add_IndonesianTranscriptionServiceServicer_to_server(IndonesianTranscriptionService(), server)
    
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
