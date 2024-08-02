Sure, here's the updated README with the MIT license included:

# Whisper API Server

This project sets up a simple FastAPI server to provide an API for transcribing audio files using OpenAI's Whisper model, integrated with CUDA. It allows users to upload audio files through an API endpoint, which are then processed by Whisper to return transcriptions.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher and below 3.10 (recommended: Python 3.8)
- Pip (Python package manager)
- FFmpeg (for audio processing)

## Installation

To install the necessary dependencies for this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/tonimurfid/ASR_Arten.git
   cd Arten_Multilingual
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

   This will install FastAPI and Whisper.

3. Make sure `ffmpeg` is installed on your system:
   - For Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - For Fedora: `sudo dnf install ffmpeg`
   - For macOS (with Homebrew): `brew install ffmpeg`
   - For Windows, download and install from [FFmpeg's official site](https://ffmpeg.org/download.html).

## Running the Server

To run the server, execute:

```bash
python app.py
```

This will start the FastAPI server on `http://localhost:8000`.

## Usage

To transcribe an audio file, refer to the Jupyter notebook provided with the project:

```bash
test.ipynb
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any queries, please contact: fatonimurfids@gmail.com
