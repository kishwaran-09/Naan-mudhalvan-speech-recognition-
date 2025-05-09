#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Install dependencies for audio processing and deep learning
get_ipython().system('pip install librosa soundfile tensorflow scikit-learn')
import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.preprocessing import LabelEncoder

# Create a directory for the dataset
DATASET_PATH = '/content/telecom_data'
os.makedirs(DATASET_PATH, exist_ok=True)

# Function to create synthetic audio (just for demonstration)
def create_synthetic_audio(filename, label):
    sr = 16000  # Sample rate
    duration = 1  # 1 second of audio
    t = np.linspace(0, duration, int(sr * duration))
    # Sine wave tone for each label as a basic example
    signal = np.sin(2 * np.pi * 440 * t)  # 440Hz tone (A4)
    sf.write(os.path.join(DATASET_PATH, f"{label}_{filename}.wav"), signal, sr)

# Create synthetic files for "hello" and "goodbye"
create_synthetic_audio('01', 'hello')
create_synthetic_audio('02', 'goodbye')
create_synthetic_audio('03', 'hello')
create_synthetic_audio('04', 'goodbye')

# Check the files in the dataset folder
print(os.listdir(DATASET_PATH))
import librosa
import numpy as np

# Function to extract MFCC features from a .wav file
def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=16000)  # Load audio file with 16kHz sample rate
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)  # Extract 13 MFCC features
    return np.mean(mfcc.T, axis=0)  # Return mean of MFCCs across time frames

# List files in the dataset folder
files = os.listdir(DATASET_PATH)
features = []
labels = []

# Extract features from each file
for file in files:
    if file.endswith('.wav'):
        label = file.split('_')[0]
         # Extract label from filename (before the first "_")
        file_path = os.path.join(DATASET_PATH, file)
        mfcc = extract_features(file_path)
        features.append(mfcc)
        labels.append(label)

X = np.array(features)  # Features for training
y = np.array(labels)  # Corresponding labels
# Encode labels (e.g., "hello" -> 0, "goodbye" -> 1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build a simple feed-forward neural network (dense layers)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y_encoded)), activation='softmax')  # Output layer (softmax for classification)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
# Train the model on the training data
model.fit(X_train, y_train, epochs=30, batch_size=4, validation_data=(X_test, y_test))
# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

