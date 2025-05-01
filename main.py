from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import numpy as np
import librosa
import cv2
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load Models
audio_model = tf.keras.models.load_model('weights/audio.h5')
image_model = tf.keras.models.load_model('weights/image.h5')
text_model = tf.keras.models.load_model('weights/text.h5')

# Load Label Encoder & Tokenizer
audio_label_encoder = pickle.load(open('weights/audio.pkl', 'rb'))
with open("weights/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Upload folder setup
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class names for image classification
image_classes = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

# --- AUDIO FEATURE EXTRACTION ---
def extract_audio_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0).reshape(1, 40, 1, 1)

# --- AUDIO PREDICTION ---
def predict_audio(file_path):
    feature = extract_audio_features(file_path)
    prediction = audio_model.predict(feature)
    return audio_label_encoder.inverse_transform([np.argmax(prediction)])[0]

# --- IMAGE PROCESSING ---
def preprocess_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(image, 'RGB').resize((128, 128))
    input_data = np.expand_dims(image, axis=0) / 255.0
    return input_data

# --- IMAGE PREDICTION ---
def predict_image(file_path):
    input_data = preprocess_image(file_path)
    if input_data is None:
        return None
    pred = image_model.predict(input_data)[0]
    confidence = np.max(pred)
    return image_classes[np.argmax(pred)] if confidence >= 0.6 else "Uncertain"

# --- TEXT PREDICTION ---
def predict_text(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=50)
    prediction = text_model.predict(pad)[0][0]
    return "Potential Suicide Post" if prediction > 0.5 else "Non-Suicide Post"

# --- ROUTES ---
@app.route("/")
def index():
    return render_template("main.html")

@app.route("/predict_audio", methods=["POST"])
def predict_audio_route():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    file = request.files['audio']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    sentiment = predict_audio(file_path)
    return jsonify({"sentiment": sentiment, "file_path": file_path})

@app.route("/predict_image", methods=["POST"])
def predict_image_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400
    file = request.files['image']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    emotion = predict_image(file_path)
    return jsonify({"emotion": emotion, "file_path": file_path})

@app.route("/predict_text", methods=["POST"])
def predict_text_route():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = predict_text(text)
    return jsonify({"result": result})

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
