from flask import Flask, request, jsonify
import librosa
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
import subprocess  # 🔧 NEW: Needed to run ffmpeg for file conversion
from generate_mfcc import preprocess_audio, run_tflite_model

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "Flask server is running."

# 📦 Load your TFLite model only once at startup
model = tf.lite.Interpreter(model_path="model.tflite")
model.allocate_tensors()

# 🔁 NEW FUNCTION: Convert uploaded audio (e.g., .3gp) to .wav using ffmpeg
def convert_to_wav(input_path, output_path):
    subprocess.run([
        'ffmpeg', '-y',              # Overwrite if file exists
        '-i', input_path,           # Input file (e.g., .3gp or .m4a)
        '-ar', '16000',             # Set sample rate to 16kHz
        '-ac', '1',                 # Set to mono audio
        output_path                 # Output .wav file path
    ], check=True)                  # Raise error if ffmpeg fails

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # 📥 STEP 1: Save the uploaded file temporarily
    file.save(filepath)

    try:
        # 🔄 STEP 2: Convert uploaded file to WAV format
        # Resulting file will have the same name but with .wav extension
        wav_path = os.path.splitext(filepath)[0] + ".wav"
        convert_to_wav(filepath, wav_path)

        # 🎧 STEP 3: Preprocess the .wav file (e.g., generate MFCC or spectrogram)
        input_data = preprocess_audio(wav_path)

        # 🔍 STEP 4: Run inference using your TFLite model
        label, confidence = run_tflite_model(input_data)

        # ✅ STEP 5: Return prediction and confidence
        return jsonify({"prediction": label, "confidence": confidence})
    
    except Exception as e:
        # ❌ STEP 6: Error handling
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 🚀 Run the Flask server on all available interfaces (important for Railway)
    app.run(host="0.0.0.0", port=5000)
