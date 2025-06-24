import os
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import joblib
import requests

# Paths to the trained model and scalers
MODEL_PATH = r"D:\mini_project\venv1\scream_detection_model.h5"
SCALER_PATH = r"D:\mini_project\venv1\scaler.pkl"
ENCODER_PATH = r"D:\mini_project\venv1\label_encoder.pkl"

# Telegram Bot Credentials
TELEGRAM_BOT_TOKEN = "7649768778:AAGdFh87kx8sIFVhsN9hltN-rHUZWQEg3os"
TELEGRAM_CHAT_ID = "6186493810"

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load the scaler and label encoder
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

# Audio recording settings
SR = 22050  # Sample rate
DURATION = 2  # Record for 2 seconds

def send_telegram_alert():
    """Sends a Telegram alert when a scream is detected."""
    message = "🚨 ALERT! Scream detected! 🚨"
    url = f"https://api.telegram.org/bot{7649768778}/sendMessage"
    payload = {"chat_id": 6186493810, "text": message}
    requests.post(url, data=payload)
    print("🚀 Alert sent to Telegram!")

def extract_features(audio):
    """Extracts MFCC features from recorded audio."""
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def classify_audio(audio):
    """Processes the recorded audio and classifies it."""
    features = extract_features(audio)
    features_scaled = scaler.transform([features])
    
    # Predict scream (1) or non-scream (0)
    prediction = model.predict(features_scaled)
    return "scream" if prediction[0] > 0.5 else "non_scream"

def record_audio_and_detect():
    """Records real-time audio and checks for screams."""
    print("🎤 Listening for screams... (Press Ctrl+C to stop)")
    
    while True:
        audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='float32')
        sd.wait()  # Wait for recording to finish
        audio = np.squeeze(audio)  # Convert to 1D array
        
        result = classify_audio(audio)
        print(f"🔊 Detected: {result}")
        
        if result == "scream":
            send_telegram_alert()

# Start real-time scream detection
record_audio_and_detect()
