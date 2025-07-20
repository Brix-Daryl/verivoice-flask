from flask import Flask, request, jsonify
import librosa
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
model = tf.lite.Interpreter(model_path="model.tflite")
model.allocate_tensors()

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = librosa.util.fix_length(mfcc, 224, axis=1)
    mfcc = mfcc[:13, :224]  # Resize if necessary
    img = np.stack([mfcc]*3, axis=-1)  # Shape: (13, 224, 3)
    img = np.resize(img, (1, 224, 224, 3)).astype(np.float32)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio uploaded'}), 400

    file = request.files['audio']
    filename = secure_filename(file.filename)
    file_path = os.path.join("uploads", filename)
    file.save(file_path)

    input_data = preprocess_audio(file_path)
    input_index = model.get_input_details()[0]['index']
    output_index = model.get_output_details()[0]['index']
    model.set_tensor(input_index, input_data)
    model.invoke()
    result = model.get_tensor(output_index)[0][0]
    os.remove(file_path)

    return jsonify({'confidence': float(result), 'label': 'fake' if result > 0.5 else 'real'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
