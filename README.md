# Whisper API Server with FastAPI and CUDA Integration

This project provides an API for transcribing audio files using OpenAI's Whisper model, integrated with CUDA for optimized performance. The server is built with FastAPI, offering a user-friendly and efficient way to process audio files via API endpoints.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [References](#references)

## Overview

The Whisper API Server allows users to upload audio files, which are then transcribed using OpenAI's Whisper model. This server is particularly suited for projects requiring scalable and fast audio transcription, taking advantage of CUDA for accelerated computation.

## Prerequisites

Before setting up the server, ensure you have the following installed:

- **Python 3.7 - 3.9** (Python 3.8 is recommended for optimal compatibility)
- **Pip** (Python package manager)
- **FFmpeg** (Required for audio processing)

## Installation

### Step 1: Clone the Repository

Begin by cloning the project repository to your local machine:

```bash
git clone https://github.com/tonimurfid/ASR_Arten.git
cd Arten_Multilingual
```

### Step 2: Install Python Dependencies

Install the necessary Python packages by executing:

```bash
pip install -r requirements.txt
```

This command will install all the required dependencies, including FastAPI and Whisper.

### Step 3: Install FFmpeg

Ensure `ffmpeg` is installed on your system, as it is essential for handling audio files:

- **Ubuntu/Debian:** 
  ```bash
  sudo apt-get install ffmpeg
  ```
- **Fedora:** 
  ```bash
  sudo dnf install ffmpeg
  ```
- **macOS (with Homebrew):** 
  ```bash
  brew install ffmpeg
  ```
- **Windows:** Download and install from [FFmpeg's official site](https://ffmpeg.org/download.html).

## Running the Server

To start the server, simply run:

```bash
python app.py
```

This command will launch the FastAPI server, which will be accessible at `http://localhost:8000`.

## Usage

To transcribe audio files using the Whisper API Server, you can either interact directly with the provided FastAPI endpoints or refer to the Jupyter notebook included in the repository for examples and usage:

- **Notebook:** 
  ```bash
  test.ipynb
  ```

## Contributing

Contributions are welcome! To contribute to this project, follow these steps:

1. **Fork the Repository.**
2. **Create a New Branch:** 
   ```bash
   git checkout -b feature-branch
   ```
3. **Make Your Changes and Commit:** 
   ```bash
   git commit -am 'Add some feature'
   ```
4. **Push to the Branch:** 
   ```bash
   git push origin feature-branch
   ```
5. **Create a New Pull Request.**

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or feedback, please reach out via email: [fatonimurfids@gmail.com](mailto:fatonimurfids@gmail.com).

## References

- This project has drawn inspiration from [Saurav Panda's Whisper Service](https://github.com/sauravpanda/whisper-service).
