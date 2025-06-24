from flask import Flask, render_template, jsonify
import threading
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import joblib
import requests
import time

app = Flask(__name__)

# Paths
MODEL_PATH = r"D:\mini_project\venv1\scream_web_app\scream_detection_model.h5"
SCALER_PATH = r"D:\mini_project\venv1\scream_web_app\scaler.pkl"
ENCODER_PATH = r"D:\mini_project\venv1\label_encoder.pkl"

# Telegram
TELEGRAM_BOT_TOKEN = "7649768778:AAGdFh87kx8sIFVhsN9hltN-rHUZWQEg3os"
TELEGRAM_CHAT_ID = "6186493810"

# Audio config
SR = 22050
DURATION = 1  # Faster response time

# Load model and preprocessing
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

# Shared variables
detection_result = "Waiting..."
alert_sent = False

def send_telegram_alert():
    message = "🚨 ALERT! Scream detected! 🚨"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=payload)
        print("🚀 Alert sent!" if response.ok else "⚠️ Failed to send alert.")
    except Exception as e:
        print(f"Telegram Error: {e}")

def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def classify_audio(audio):
    features = extract_features(audio)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    score = prediction[0][0]
    print(f"🎯 Prediction score: {score:.4f}")
    return "scream" if score > 0.5 else "non_scream"

def detection_loop():
    global detection_result, alert_sent
    while True:
        print("🎙️ Listening...")
        audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='float32')
        sd.wait()
        audio = np.squeeze(audio)

        result = classify_audio(audio)
        detection_result = result
        print(f"🔍 Detected: {result}")

        if result == "scream" and not alert_sent:
            send_telegram_alert()
            alert_sent = True
        elif result == "non_scream":
            alert_sent = False

        time.sleep(0.2)  # Smaller pause for faster loop

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/start_detection')
def start_detection():
    threading.Thread(target=detection_loop, daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/get_result')
def get_result():
    return jsonify({"result": detection_result})

if __name__ == '__main__':
    app.run(debug=True)
