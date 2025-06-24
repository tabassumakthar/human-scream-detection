# human-scream-detection
Human Scream Detection is a real-time, CPU-optimized scream detection and safety alerting system. Designed to run continuously on a PC, it not only detects screams from live microphone input but also performs threat analysis to assess potential human safety concerns. When a valid threat is detected, it automatically sends a notification via Telegram.

🔍 Features
🎤 Real-Time Scream Detection
Detects human screams using a custom-trained audio classification model optimized for CPU.

📬 Telegram Alert System
Sends alerts only if a real threat is confirmed by analyzing the video content.

🌐 Web Interface
Integrates a Flask-based web UI to display real-time detection results and logs.

🧰 Tech Stack
Python (NumPy, OpenCV, librosa, etc.)
PyTorch/TensorFlow (for custom scream detection model)
Flask for web UI
Telegram Bot API for notifications
🧪 Dataset & Model
Trained using ESC-50 dataset
Only the human_scream category was extracted
Other sounds were merged into a non_scream class
Environmental noise augmentation applied (e.g., vehicle sounds, crowd cheers)
🚧 Future Improvements
Mobile app integration
Support for multilingual alerting
Further model compression for edge deployment
1. Clone the Repository
