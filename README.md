# Naan-mudhalvan-speech-recognition-
#Naanmudhalvan-speech-reconition-# 

accent_aware_virtual_assistant 1
# Accent-Aware Speech Recognition System for Virtual Assistant

This project implements a **real-time accent-aware speech recognition system** using deep learning and speaker adaptation techniques. It's designed to improve the performance of voice-based virtual assistants by adapting to different speaker accents and voices.

## Features

- Real-time speech-to-text transcription using Wav2Vec2
- Accent and speaker adaptation using x-vector embeddings from SpeechBrain
- RESTful API using Flask for easy integration
- Upload audio and get transcribed text and speaker embedding
- Supports `.wav` files sampled at 16kHz

---

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Torchaudio
- Librosa
- SpeechBrain
- Flask

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/accent-aware-virtual-assistant.git
cd accent-aware-virtual-assistant
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, manually install:
```bash
pip install torch torchaudio librosa transformers speechbrain flask
```

3. **Run the server:**

```bash
python app/accent_aware_virtual_assistant.py
```

---

## API Usage

### Endpoint:
`POST /transcribe`

### Payload:
Send a `.wav` file using form-data.

**Example using curl:**
```bash
curl -X POST -F "audio=@sample.wav" http://localhost:5000/transcribe
```

### Response:
```json
{
  "transcription": "Turn on the lights in the kitchen.",
  "speaker_embedding": [0.0412, -0.0133, ...]
}
```

---

## File Structure

```
accent-aware-virtual-assistant/
│
├── app/
│   └── accent_aware_virtual_assistant.py
│
├── uploads/
├── pretrained_models/
│   └── spkrec/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## License

This project is licensed under the MIT License.

-------------------------------------------------------------------------

Building a Speech-to-Text Transcription for healthcare 2

# Real-Time Speech-to-Text Transcription System for Healthcare with Noise Robustness

This Python project implements a real-time speech-to-text transcription system designed specifically for healthcare environments. It uses OpenAI’s Whisper model for transcription and includes noise robustness, medical term normalization, and critical keyword detection.

## Features

- **Real-time microphone recording**
- **Speech-to-text transcription** using Whisper (`base` model)
- **Noise robustness** with live audio stream handling
- **Normalization of medical terms** (e.g., "bp" → "blood pressure")
- **Keyword detection** for important healthcare alerts
- **Automatic logging** of transcriptions with timestamps

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/healthcare-stt-whisper.git
   cd healthcare-stt-whisper
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

## Usage

Run the main script:

```bash
python main_healthcare.py
```

### Output

The system will:
- Listen to audio in 5-second intervals
- Transcribe the speech
- Normalize common medical shorthand
- Detect and print alerts for keywords (e.g., "emergency", "blood pressure")
- Save all transcriptions in `transcription_log.txt`

### Sample Console Output

```
[INFO] Recording started. Speak now...
Transcription:
The patient's blood pressure is 130 over 85 and shows signs of diabetic condition.
------------------------------------------------------------
[ALERT] Detected medical terms: blood pressure, diabetic
```

## Example Medical Term Normalization

| Detected Term     | Normalized Term     |
|-------------------|---------------------|
| bp                | blood pressure      |
| sugar level       | glucose level       |
| ekg               | ECG                 |
| covid             | COVID-19            |
| diabetes type 1   | Type 1 Diabetes     |

## File Structure

```
healthcare-stt-whisper/
├── main_healthcare.py         # Main Python script
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
├── transcription_log.txt      # Auto-created log of transcriptions
└── .gitignore                 # Common ignored files
```

## Requirements

- Python 3.7+
- whisper
- numpy
- sounddevice
- soundfile
- torch

## License

This project is open-source and licensed under the [MIT License](LICENSE).

---

**Developed for use in healthcare transcription, telemedicine support, and voice-powered diagnostics.**

-------------------------------------------------------------------------------------------------------------------------------

Real-Time Speech-to-Text System 3

# Real-Time Speech-to-Text for Contact Center Automation

This project is a real-time speech-to-text transcription system using [OpenAI Whisper](https://github.com/openai/whisper). It's designed for customer support automation in contact centers, enabling real-time transcription from microphone input, even in noisy environments.

## Features

- Real-time microphone audio capture
- Robust speech-to-text transcription using Whisper
- Handles noisy environments effectively
- Easily extendable (CRM integration, sentiment analysis, speaker diarization)

## Requirements

Install dependencies using pip:

```bash
pip install openai-whisper sounddevice numpy soundfile
```

You also need to have `ffmpeg` installed. On Ubuntu/Debian:

```bash
sudo apt install ffmpeg
```

On MacOS with Homebrew:

```bash
brew install ffmpeg
```

## Usage

Save the following code in `main.py` and run it:

```python
import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import time
import tempfile
import soundfile as sf

model = whisper.load_model("base")
q = queue.Queue()
samplerate = 16000
blocksize = 16000 * 5  # 5 seconds of audio

def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

def record_audio():
    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", callback=audio_callback):
        print("Listening...")
        while True:
            time.sleep(0.1)

def transcribe_audio():
    while True:
        if not q.empty():
            audio_chunk = q.get()
            audio_chunk = np.squeeze(audio_chunk)
            audio_int16 = (audio_chunk * 32767).astype(np.int16)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_int16, samplerate)
                result = model.transcribe(f.name)
                print("Transcription:", result["text"])

threading.Thread(target=record_audio, daemon=True).start()
transcribe_audio()
```

Run the app with:

```bash
python main.py
```

## Future Extensions

- Speaker diarization (agent vs. customer)
- Sentiment analysis and intent recognition
- CRM integration (Salesforce, Zendesk, etc.)

## License

MIT License

-------------------------------------------------------------------------
