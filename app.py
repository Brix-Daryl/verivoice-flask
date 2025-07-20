from flask import Flask, request, jsonify
import librosa
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
from generate_mfcc import preprocess_audio, run_tflite_model

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "Flask server is running."

model = tf.lite.Interpreter(model_path="model.tflite")
model.allocate_tensors()

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        input_data = preprocess_audio(filepath)
        label, confidence = run_tflite_model(input_data)

        return jsonify({"prediction": label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
