import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model (.h5)
model = load_model("weights/text.h5")

# Load tokenizer (Ensure you have saved it in a .pkl file)
with open("weights/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Example text input
twt = ["Yeah, there's all sorts of differences. I'm gonna go along in this building. It's here in my Craigslist, so that's why. Oh, yeah, there's a couple groups of these stakes. ... (truncated)"]

# Convert text to sequence
twt_seq = tokenizer.texts_to_sequences(twt)
twt_pad = pad_sequences(twt_seq, maxlen=50)  # Ensure maxlen matches training

# Make prediction
prediction = model.predict(twt_pad)[0][0]

# Print results
print(f"Prediction Score: {prediction}")

if prediction > 0.5:
    print("Potential Suicide Post")
else:
    print("Non-Suicide Post")
