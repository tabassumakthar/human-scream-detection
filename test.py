import numpy as np
import librosa
import tensorflow as tf
import joblib
import requests

# Paths to the trained model and scalers
MODEL_PATH = r"D:\mini_project\venv1\scream_detection_model.h5"
SCALER_PATH = r"D:\mini_project\venv1\scaler.pkl"
ENCODER_PATH = r"D:\mini_project\venv1\label_encoder.pkl"

# Telegram Bot Credentials
TELEGRAM_BOT_TOKEN = "7649768778:AAGdFh87kx8sIFVhsN9hltN-rHUZWQEg3os"  # Replace with your correct token
TELEGRAM_CHAT_ID = "6186493810"  # Replace with the correct chat ID from @userinfobot

# Load the trained model and other components
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

def send_telegram_alert():
    """Sends a Telegram alert when a scream is detected."""
    message = "🚨 ALERT! Scream detected in audio file! 🚨"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    response = requests.post(url, data=payload)
    
    if response.status_code == 200:
        print("🚀 Alert sent to Telegram!")
    else:
        print(f"❌ Failed to send Telegram alert. Status code: {response.status_code}")
        print(response.text)

def extract_features(audio, sr):
    """Extracts MFCC features from audio signal."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def classify_audio(audio, sr):
    """Processes the audio signal and classifies it."""
    features = extract_features(audio, sr)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return "scream" if prediction[0] > 0.5 else "non_scream"

def test_audio_file(file_path):
    """Loads audio file and tests if it's a scream."""
    audio, sr = librosa.load(file_path, sr=22050)
    result = classify_audio(audio, sr)
    print(f"🔊 File: {file_path} | Detected: {result}")

    if result == "scream":
        send_telegram_alert()

# 🔍 Set your test WAV file path here
test_audio_file(r"D:\mini_project\venv1\sounds\sounds\1....220285__gabrielaupf__screaming-male-effort (6).wav")
