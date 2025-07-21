import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import tensorflow as tf
from PIL import Image
from io import BytesIO

# Load the TFLite model once
interpreter = tf.lite.Interpreter(model_path="model.tflite")  # adjust if different name
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def audio_to_spectrogram_image(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.24, 2.24), frameon=False)
    plt.axis('off')
    librosa.display.specshow(S_DB, sr=sr, cmap='magma')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)

    img = Image.open(buf).convert('RGB')
    return img.resize((224, 224))

def preprocess_audio(audio_path):
    image = audio_to_spectrogram_image(audio_path)
    arr = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def run_tflite_model(input_array):
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0][0]
    label = "Genuine" if output >= 0.5 else "AI-Generated"
    return label, float(output)
