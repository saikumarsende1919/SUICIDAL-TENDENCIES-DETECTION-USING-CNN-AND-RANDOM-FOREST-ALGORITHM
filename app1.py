from flask import Flask, render_template, request, jsonify
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
import whisper

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

# --- AUDIO TRANSCRIPTION ---
def transcribe_audio_with_whisper(audio_path, model_name="base"):
    try:
        model = whisper.load_model(model_name)
        audio, sr = librosa.load(audio_path, sr=16000)  # Convert to 16kHz for Whisper
        result = model.transcribe(audio=audio)
        return result["text"]
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return None

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
def predict_image(file_path, threshold=0.6):
    input_data = preprocess_image(file_path)
    if input_data is None:
        return None

    pred = image_model.predict(input_data)[0]
    predicted_index = np.argmax(pred)
    predicted_confidence = np.max(pred)

    if predicted_confidence >= threshold:
        predicted_class = image_classes[predicted_index]
    elif predicted_confidence >= 0.5:
        predicted_class = image_classes[predicted_index]
    elif predicted_confidence >= 0.4:
        predicted_class = image_classes[predicted_index]
    return predicted_class

# --- TEXT PREDICTION ---
def predict_text(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=50)
    prediction = text_model.predict(pad)[0][0]
    return "Suicide" if prediction > 0.5 else "Non-Suicide"

# --- COMBINED PREDICTION ---
def determine_suicide_risk(audio_result, image_result, text_result):
    positive_image_classes = ["Anger", "Sadness", "Fear"]
    
    image_positive = image_result in positive_image_classes
    text_positive = text_result == "Suicide"
    audio_positive = audio_result in ["Sadness", "Positive"] 
    
    positive_count = sum([image_positive, text_positive, audio_positive])
    
    risk_levels = {
        3: ("Suicide Post", 100),
        2: ("Potential Suicide Post", 66),
        1: ("Trying to Suicide Post", 33),
        0: ("Not a Suicide Post", 0)
    }
    
    return risk_levels[positive_count]

# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_combined", methods=["POST"])
def predict_combined():
    if 'audio' not in request.files or 'image' not in request.files or 'text' not in request.form:
        return jsonify({"error": "Missing input data"}), 400

    # Process audio
    audio_file = request.files['audio']
    audio_filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_filename)
    audio_file.save(audio_path)

    # Transcribe audio using Whisper
    transcribed_text = transcribe_audio_with_whisper(audio_path)

    # Predict sentiment from transcribed text
    audio_result = predict_audio(audio_path)

    # Process image
    image_file = request.files['image']
    image_filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
    image_file.save(image_path)
    image_result = predict_image(image_path)

    # Process text (use transcribed text if original text is empty)
    text_input = request.form['text'].strip()
    if not text_input:
        text_input = transcribed_text  # Use Whisper transcription if no text was provided

    text_result = predict_text(text_input)

    # Determine final prediction
    final_prediction, risk_percentage = determine_suicide_risk(audio_result, image_result, text_result)

    return jsonify({
        "audio_result": audio_result,
        "image_result": image_result,
        "text_result": text_result,
        "final_prediction": final_prediction,
        "risk_percentage": risk_percentage,
        "transcribed_text": transcribed_text  # Send Whisper transcription to frontend
    })

if __name__ == "__main__":
    app.run(debug=True)
