# from flask import Flask, render_template, request, send_from_directory
# import numpy as np
# import librosa
# from werkzeug.utils import secure_filename
# import os
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder
# import pickle

# # Initialize the Flask application
# app = Flask(__name__)

# model = tf.keras.models.load_model('weights/audio.h5')
# le = pickle.load(open('weights/audio.pkl','rb'))

# # Audio upload path
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Ensure upload folder exists
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Function to extract features from audio
# def extract_features(file_name):
#     audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
#     mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
#     return np.mean(mfccs.T, axis=0)

# # Function to predict sentiment
# def predict_sentiment(audio_path):
#     feature = extract_features(audio_path) 
#     feature = feature.reshape(1, 40, 1, 1)  
#     prediction = model.predict(feature)     
#     predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
#     return predicted_label

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         # Get the uploaded file
#         file = request.files["audio"]
#         if file:
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#             file.save(file_path)

#             # Predict sentiment
#             predicted_sentiment = predict_sentiment(file_path)

#             # Return the result to the frontend
#             return render_template("index.html",
#                                    sentiment=predicted_sentiment,
#                                    audio_path=file_path)

#     return render_template("index.html", sentiment=None, audio_path=None)

# # Route to serve uploaded files
# @app.route("/uploads/<filename>")
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



# if __name__ == "__main__":
#     app.run(debug=True)

import numpy as np
import librosa
import tensorflow as tf
import pickle
import whisper

# Load the trained sentiment analysis model and label encoder
model = tf.keras.models.load_model('weights/audio.h5')
le = pickle.load(open('weights/audio.pkl', 'rb'))

# Function to extract features from audio
def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Function to predict sentiment
def predict_sentiment(audio_path):
    feature = extract_features(audio_path) 
    feature = feature.reshape(1, 40, 1, 1)  
    prediction = model.predict(feature)     
    predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
    return predicted_label

# Function to transcribe audio using Whisper (without ffmpeg)
def transcribe_audio_with_whisper_in_english(audio_path, model_name="base"):
    try:
        print("Loading Whisper model...")
        model = whisper.load_model(model_name)

        print("Loading audio using librosa (bypassing ffmpeg)...")
        audio, sr = librosa.load(audio_path, sr=16000)  # Convert to 16kHz for Whisper
        
        print("Transcribing audio with translation to English...")
        result = model.transcribe(audio=audio)  # Use raw waveform instead of file path
        
        transcription = result["text"]
        print("\nTranscription (Translated to English):")
        print(transcription)
        return transcription
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return None

# Example usage
if __name__ == "__main__":
    audio_file = "input_data/audio/112.wav"  # Replace with actual file path

    # Transcribe the audio (without using ffmpeg)
    transcription = transcribe_audio_with_whisper_in_english(audio_file)

    # Predict sentiment
    sentiment = predict_sentiment(audio_file)

    print(f"\nTranscription: {transcription}")
    print(f"Predicted Sentiment: {sentiment}")
