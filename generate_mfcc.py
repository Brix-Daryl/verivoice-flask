import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def extract_mfcc(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=22050)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Normalize
    mfccs = librosa.util.normalize(mfccs)

    # Create a temporary figure
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)  # 224x224 pixels
    plt.axis('off')
    plt.imshow(mfccs, aspect='auto', origin='lower', cmap='viridis')
    
    # Save to buffer
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # Resize to (224, 224, 3) just in case
    image = Image.fromarray(img)
    image = image.resize((224, 224))
    array = np.array(image)

    return array
