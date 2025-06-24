import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Paths
AUDIO_DIR = r"D:\mini_project\venv1\sounds\sounds"
CSV_FILE = r"D:\mini_project\venv1\sound_classification.csv"
MODEL_SAVE_PATH = r"D:\mini_project\venv1\scream_detection_model.h5"
SCALER_SAVE_PATH = r"D:\mini_project\venv1\scaler.pkl"
ENCODER_SAVE_PATH = r"D:\mini_project\venv1\label_encoder.pkl"

# Load CSV
df = pd.read_csv(CSV_FILE)

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Extract features for each file
features, labels = [], []

for index, row in df.iterrows():
    filename = row['filename']
    label = row['label']
    
    file_path = os.path.join(AUDIO_DIR, filename)
    
    if os.path.exists(file_path):
        feature = extract_features(file_path)
        if feature is not None:
            features.append(feature)
            labels.append(label)
    else:
        print(f"File not found: {file_path}")

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save the label encoder
joblib.dump(encoder, ENCODER_SAVE_PATH)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, SCALER_SAVE_PATH)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Build a simple neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification: scream (1) / non-scream (0)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Save the model
model.save(MODEL_SAVE_PATH)

print(f"Model trained and saved at {MODEL_SAVE_PATH}")
