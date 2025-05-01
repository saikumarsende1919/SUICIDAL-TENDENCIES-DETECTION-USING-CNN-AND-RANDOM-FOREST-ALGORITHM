import keras
import numpy as np
import cv2
from PIL import Image

# Load model
model_path = "weights/image.h5"
loaded_model = keras.models.load_model(model_path)

# Define the correct emotion class names
class_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

# Read image as grayscale
image_path = "C:\\Users\\infyz\\.cache\\kagglehub\\datasets\\msambare\\fer2013\\versions\\1\\train\\happy\\Training_50580.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image was loaded successfully
if image is None:
    raise FileNotFoundError(f"Error: Unable to read the image file at {image_path}. Check if the file exists and is accessible.")

# Convert grayscale to RGB
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# Resize image
image_fromarray = Image.fromarray(image, 'RGB')
resize_image = image_fromarray.resize((128, 128))

# Normalize and prepare input
expand_input = np.expand_dims(resize_image, axis=0)
input_data = np.array(expand_input) / 255.0

# Predict
pred = loaded_model.predict(input_data)
pred_probabilities = pred[0]  # Get prediction probabilities
predicted_index = pred_probabilities.argmax()  # Get class index
predicted_confidence = pred_probabilities[predicted_index]  # Get confidence score

# Set threshold (adjust as needed)
threshold = 0.6

# Determine final class output
if predicted_confidence >= threshold:
    predicted_class = class_names[predicted_index]
else:
    predicted_class = "Uncertain"

# Output result
print(f"Predicted Class: {predicted_class} (Confidence: {predicted_confidence:.2f})")
