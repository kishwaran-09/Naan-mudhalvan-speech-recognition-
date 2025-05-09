#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install librosa soundfile wget')
get_ipython().system('apt-get install -y ffmpeg')


# In[ ]:


get_ipython().system('wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz')
get_ipython().system('mkdir -p speech_commands')
get_ipython().system('tar -xvzf speech_commands_v0.02.tar.gz -C speech_commands')


# In[ ]:


import wget

url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
wget.download(url, "ESC-50.zip")

get_ipython().system('unzip -q ESC-50.zip')
get_ipython().system('mkdir -p noise_data')
get_ipython().system('cp ESC-50-master/audio/*.wav noise_data/')


# In[ ]:


import os
import librosa
import soundfile as sf

def normalize_audio(input_path, output_path, target_sr=16000):
    try:
        audio, sr = librosa.load(input_path, sr=target_sr)
        audio = librosa.util.normalize(audio)
        sf.write(output_path, audio, target_sr)
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")


# In[ ]:


import glob

# Output folders
os.makedirs("clean_speech", exist_ok=True)
os.makedirs("clean_noise", exist_ok=True)

# Normalize speech files (e.g., first 100 files from "yes")
speech_files = glob.glob("speech_commands/yes/*.wav")[:100]
for path in speech_files:
    filename = os.path.basename(path)
    normalize_audio(path, f"clean_speech/{filename}")

# Normalize noise files (e.g., first 50 noise clips)
noise_files = glob.glob("noise_data/*.wav")[:50]
for path in noise_files:
    filename = os.path.basename(path)
    normalize_audio(path, f"clean_noise/{filename}")


# In[ ]:


def is_valid_audio(path):
    try:
        _ = librosa.load(path, sr=None)
        return True
    except:
        return False

# Remove bad speech files
for f in glob.glob("clean_speech/*.wav"):
    if not is_valid_audio(f):
        os.remove(f)
        print(f"Removed corrupted file: {f}")

# Remove bad noise files
for f in glob.glob("clean_noise/*.wav"):
    if not is_valid_audio(f):
        os.remove(f)
        print(f"Removed corrupted noise: {f}")


# In[ ]:


import os
import matplotlib.pyplot as plt

# Count samples per label (e.g., "yes", "no")
labels = [d for d in os.listdir("speech_commands") if os.path.isdir(f"speech_commands/{d}")]
label_counts = {label: len(os.listdir(f"speech_commands/{label}")) for label in labels}

# Plot
plt.figure(figsize=(12,6))
plt.bar(label_counts.keys(), label_counts.values())
plt.xticks(rotation=45)
plt.title("Number of Samples per Label")
plt.ylabel("Count")
plt.grid(True)
plt.show()


# In[ ]:


# Download Google Speech Commands dataset (v0.02)
get_ipython().system('wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz')
get_ipython().system('tar -xzf speech_commands_v0.02.tar.gz')


# In[ ]:


get_ipython().system('pip install git+https://github.com/openai/whisper.git')
import whisper
import os

model = whisper.load_model("base")

# Check if file exists
file_path = "speech_commands/yes/0a7c2a8d_nohash_0.wav"
if os.path.exists(file_path):
    result = model.transcribe(file_path)
    print(result["text"])
else:
    print(f"Error: File not found - {file_path}")


# In[ ]:


# Download (≈1 GB)
get_ipython().system('wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz')

# Extract
get_ipython().system('mkdir -p speech_commands')
get_ipython().system('tar -xzf speech_commands_v0.02.tar.gz -C speech_commands')


# In[ ]:


import os

sample_path = "speech_commands/yes/0a7c2a8d_nohash_0.wav"
print("File exists:", os.path.isfile(sample_path))


# In[ ]:


get_ipython().system('ls speech_commands/yes | head')


# In[ ]:


from vosk import Model, KaldiRecognizer
import wave
import json

# Path to model and audio file
model_path = "vosk-model-small-en-us-0.15"
audio_path = "speech_commands/yes/0a7c2a8d_nohash_0.wav"  # Replace with real path if needed

# Load model
model = Model(model_path)

# Transcribe
wf = wave.open(audio_path, "rb")
rec = KaldiRecognizer(model, wf.getframerate())

results = []
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        results.append(json.loads(rec.Result()))

# Output text
print(" ".join([r["text"] for r in results]))


# In[ ]:


import os
from collections import Counter

base_path = "speech_commands"
labels = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and not d.startswith('_')]

sample_counts = {label: len(os.listdir(os.path.join(base_path, label))) for label in labels}
print("Sample count per label:\n", sample_counts)


# In[ ]:


import numpy as np
import librosa
import soundfile as sf

def add_noise(input_path, output_path, noise_factor=0.02):
    audio, sr = librosa.load(input_path, sr=None)
    noise = np.random.randn(len(audio))
    audio_noisy = audio + noise_factor * noise
    sf.write(output_path, audio_noisy, sr)

# Example usage
add_noise("speech_commands/yes/0a7c2a8d_nohash_0.wav", "yes_noisy.wav")


# In[ ]:


from vosk import Model, KaldiRecognizer
import wave
import json

def transcribe_vosk(model_path, audio_path):
    wf = wave.open(audio_path, "rb")
    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    return " ".join([r.get("text", "") for r in results])

# Run transcription
clean_text = transcribe_vosk("vosk-model-small-en-us-0.15", "speech_commands/yes/0a7c2a8d_nohash_0.wav")
noisy_text = transcribe_vosk("vosk-model-small-en-us-0.15", "yes_noisy.wav")

print("Clean:", clean_text)
print("Noisy:", noisy_text)


# In[ ]:


import pandas as pd

# Example data
data = [
    {"file": "yes_clean.wav", "noise_level": "none", "accuracy": 1.0, "wer": 0.0, "homophone_error": False},
    {"file": "yes_noisy.wav", "noise_level": "low", "accuracy": 0.7, "wer": 0.3, "homophone_error": False},
    {"file": "write.wav", "noise_level": "medium", "accuracy": 0.6, "wer": 0.4, "homophone_error": True},
    {"file": "right.wav", "noise_level": "medium", "accuracy": 0.5, "wer": 0.5, "homophone_error": True},
]

df = pd.DataFrame(data)
df.to_csv("speech_metrics.csv", index=False)


# In[ ]:


df["timestamp"] = pd.date_range(start="2025-01-01", periods=len(df), freq="D")
df["accent"] = ["US", "US", "UK", "UK"]


# In[ ]:


import os
import pandas as pd
import numpy as np
from vosk import Model, KaldiRecognizer
import wave
import json

# -------------------------------
# Helper function: Vosk Transcription
# -------------------------------
def transcribe_vosk(model_path, audio_path):
    wf = wave.open(audio_path, "rb")
    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    return " ".join([r.get("text", "") for r in results])

# -------------------------------
# Simulated Test Set
# -------------------------------
# You can replace these with real paths and labels
test_files = [
    {"file": "speech_commands/yes/0a7c2a8d_nohash_0.wav", "label": "yes", "noise": "none"},
    {"file": "yes_noisy.wav", "label": "yes", "noise": "low"},
    {"file": "write.wav", "label": "write", "noise": "medium"},
    {"file": "right.wav", "label": "right", "noise": "medium"},
]

# -------------------------------
# Run Evaluation
# -------------------------------
model_path = "vosk-model-small-en-us-0.15"
results = []

for sample in test_files:
    file_path = sample["file"]
    true_label = sample["label"]
    noise = sample["noise"]

    if not os.path.exists(file_path):
        print(f"Skipping missing file: {file_path}")
        continue

    prediction = transcribe_vosk(model_path, file_path)
    wer = np.random.uniform(0.1, 0.5)  # Simulated WER (you can compute it properly if needed)
    accuracy = 1.0 if true_label in prediction else 0.0
    is_homophone = true_label in ["write", "right"]  # You can build a real homophone set later

    results.append({
        "file": os.path.basename(file_path),
        "noise_level": noise,
        "true_label": true_label,
        "predicted": prediction,
        "accuracy": accuracy,
        "wer": round(wer, 2),
        "homophone_error": is_homophone and accuracy == 0.0
    })

# -------------------------------
# Save to CSV
# -------------------------------
df = pd.DataFrame(results)
df.to_csv("speech_metrics.csv", index=False)
print("✅ Saved: speech_metrics.csv")
df.head()


# In[ ]:


from google.colab import files
files.download('speech_metrics.csv')


# In[ ]:


get_ipython().system('pip install torchaudio librosa')

import os
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchaudio.transforms as T
import librosa
import numpy as np


# In[ ]:


from torchaudio.datasets import SPEECHCOMMANDS

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__(".", download=True)
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as f:
                return [os.path.join(self._path, line.strip()) for line in f]
        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

train_set = SubsetSC("training")
test_set = SubsetSC("testing")


# In[ ]:


def label_to_tensor(label):
    return torch.tensor(label_to_index[label])

mfcc_transform = T.MFCC(sample_rate=16000, n_mfcc=40)

def preprocess(sample):
    waveform, sample_rate, label, *_ = sample
    mfcc = mfcc_transform(waveform).squeeze(0).transpose(0, 1)  # (time, mfcc)
    return mfcc, label_to_tensor(label)


# In[ ]:


import pandas as pd

# Example data (replace with your actual metrics)
data = {
    "Condition": ["Clean", "Noisy"],
    "Accuracy (%)": [92.5, 78.4],
    "WER (%)": [7.5, 21.6],
    "Latency (ms)": [120, 130]
}

# Create a DataFrame
df_metrics = pd.DataFrame(data)

# Save to CSV
df_metrics.to_csv("speech_kpis.csv", index=False)


# In[ ]:


import pandas as pd

data = {
    "Condition": ["Clean", "Noisy - Light", "Noisy - Medium", "Noisy - Heavy"],
    "Accuracy (%)": [93.8, 88.5, 82.3, 76.0],
    "WER (%)": [6.2, 11.5, 17.7, 24.0],
    "Latency (ms)": [115, 125, 132, 145]
}

df = pd.DataFrame(data)
df.to_csv("speech_kpis.csv", index=False)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Make sure pandas is imported

# Assuming your data is in 'speech_kpis.csv'
df = pd.read_csv("speech_kpis.csv") # Read data from CSV to DataFrame

plt.figure(figsize=(8, 5))
sns.barplot(x='Condition', y='WER (%)', data=df) # Use df here instead of dataset
plt.title('Word Error Rate (WER) under Different Conditions')
plt.ylabel('WER (%)')
plt.xlabel('Audio Condition')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Make sure pandas is imported

# Assuming your data is in 'speech_kpis.csv'
df = pd.read_csv("speech_kpis.csv") # Read data from CSV to DataFrame

plt.figure(figsize=(8, 5))
sns.lineplot(x='Condition', y='Accuracy (%)', marker='o', data=df) # Use df here instead of dataset
plt.title('Accuracy Across Audio Conditions')
plt.ylabel('Accuracy (%)')
plt.xlabel('Condition')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # Make sure pandas is imported

# Assuming your data is in 'speech_kpis.csv'
df = pd.read_csv("speech_kpis.csv")  # Read data from CSV to DataFrame

plt.figure(figsize=(8, 5))
sns.barplot(x='Condition', y='Latency (ms)', data=df)  # Use df instead of dataset
plt.title('Latency by Audio Condition')
plt.ylabel('Latency (ms)')
plt.xlabel('Condition')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample accuracy data
data = {
    'Accent': ['American', 'British', 'Indian', 'Australian'],
    'Clean': [94, 91, 88, 90],
    'Light Noise': [90, 87, 83, 85],
    'Heavy Noise': [78, 75, 70, 72]
}

df = pd.DataFrame(data).set_index('Accent')
plt.figure(figsize=(8, 5))
sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".1f")
plt.title("Accuracy Heatmap: Accents vs. Noise Levels")
plt.ylabel("Accent")
plt.xlabel("Noise Level")
plt.tight_layout()
plt.show()


# In[ ]:


# Bar chart of error types
error_types = ['Homophones', 'Accents', 'Noise']
counts = [120, 80, 150]

plt.figure(figsize=(6,4))
sns.barplot(x=error_types, y=counts, palette='pastel')
plt.title("Error Distribution by Type")
plt.ylabel("Number of Errors")
plt.show()


# In[ ]:


# Simulated WER over 10 epochs
epochs = list(range(1, 11))
wer = [25.0, 21.4, 18.3, 15.9, 14.2, 12.5, 11.7, 10.8, 10.1, 9.6]

plt.figure(figsize=(8,5))
plt.plot(epochs, wer, marker='o', linestyle='-', color='teal')
plt.title("WER Over Training Iterations")
plt.xlabel("Epoch")
plt.ylabel("Word Error Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Simulated labels
y_true = ['yes', 'no', 'go', 'stop', 'yes', 'no', 'go', 'stop']
y_pred = ['yes', 'no', 'stop', 'stop', 'yes', 'go', 'go', 'stop']

labels = ['yes', 'no', 'go', 'stop']
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Word Predictions")
plt.tight_layout()
plt.show()


# In[ ]:


dataset_path = "/content/your_audio_folder"  # If your files are stored in this directory in Colab


# In[ ]:


from google.colab import files
uploaded = files.upload()  # Upload files manually

# Now use the correct folder path
dataset_path = "/content"


# In[ ]:


import os
os.listdir("/content")


# In[ ]:


import nltk
from nltk.corpus import wordnet as wn

# Download WordNet if needed
nltk.download('wordnet')

# List of homophones (this is an example, expand it)
homophones = [
    ("sea", "see"),
    ("flower", "flour"),
    ("break", "brake"),
]

# Check for homophones in a sentence
def check_homophones(sentence, homophones):
    words = sentence.split()
    homophone_pairs = []
    for word1, word2 in homophones:
        if word1 in words and word2 in words:
            homophone_pairs.append((word1, word2))
    return homophone_pairs

# Example transcription (replace with actual transcriptions)
transcription = "I see a flower near the sea."

# Check homophones in the transcription
pairs = check_homophones(transcription, homophones)
print("Detected homophones in transcription:", pairs)


# In[ ]:


get_ipython().system('pip install jiwer')


# In[ ]:


from jiwer import wer, cer

# Example data
ground_truth = ["yes", "no", "maybe"]
predictions_clean = ["yes", "no", "maybe"]
predictions_noisy = ["yes", "no", "may be"]

# Evaluate
print("WER (Clean):", wer(ground_truth, predictions_clean))
print("WER (Noisy):", wer(ground_truth, predictions_noisy))


# In[ ]:


from jiwer import wer, cer

# Ground truth and predictions
ground_truth = ["yes", "no", "maybe"]
predictions_clean = ["yes", "no", "maybe"]
predictions_noisy = ["yes", "no", "may be"]

# WER and CER
print("WER (Clean):", wer(ground_truth, predictions_clean))
print("WER (Noisy):", wer(ground_truth, predictions_noisy))
print("CER (Clean):", cer(ground_truth, predictions_clean))
print("CER (Noisy):", cer(ground_truth, predictions_noisy))


# In[ ]:


get_ipython().system('pip install vosk')


# In[ ]:


get_ipython().system('wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip')
get_ipython().system('unzip -o vosk-model-small-en-us-0.15.zip')


# In[1]:


from google.colab import files
uploaded = files.upload()


# In[2]:


get_ipython().system('pip install jiwer')

from jiwer import wer, cer
from sklearn.metrics import precision_score, recall_score, f1_score
import time

# Example ground truth and predictions
ground_truth = ["yes", "no", "maybe"]
predicted_clean = ["yes", "no", "maybe"]
predicted_noisy = ["yes", "now", "baby"]

# Calculate WER
print("WER (clean):", wer(ground_truth, predicted_clean))
print("WER (noisy):", wer(ground_truth, predicted_noisy))

# Accuracy
def word_accuracy(gt, pred):
    correct = sum([g == p for g, p in zip(gt, pred)])
    return correct / len(gt)

print("Accuracy (clean):", word_accuracy(ground_truth, predicted_clean))
print("Accuracy (noisy):", word_accuracy(ground_truth, predicted_noisy))

# Precision, Recall, F1 (for multi-class words)
all_labels = list(set(ground_truth + predicted_noisy))
y_true = [all_labels.index(w) for w in ground_truth]
y_pred = [all_labels.index(w) for w in predicted_noisy]

print("Precision:", precision_score(y_true, y_pred, average='macro'))
print("Recall:", recall_score(y_true, y_pred, average='macro'))
print("F1 Score:", f1_score(y_true, y_pred, average='macro'))

# Latency measurement (example)
start_time = time.time()
# Simulated model prediction
_ = [w for w in predicted_noisy]
end_time = time.time()

print("Latency (in seconds):", round(end_time - start_time, 4))


# In[8]:


get_ipython().system('pip install Whisper')


# In[9]:


get_ipython().system('pip install gradio')


# In[11]:


get_ipython().system('pip install --upgrade --no-cache-dir git+https://github.com/openai/whisper.git')


# In[4]:


import whisper
import gradio as gr

model = whisper.load_model("small")

def transcribe(audio):

    #time.sleep(3)
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    return result.text



gr.Interface(
    title = 'OpenAI Whisper ASR Gradio Web UI',
    fn=transcribe,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath") # Changed from gr.inputs.Audio to gr.Audio
    ],
    outputs=[
        "textbox"
    ],
    live=True).launch()

