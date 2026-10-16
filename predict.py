import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("model/brain_model.keras")  # Ensure this path matches where your model is saved

# Class names (must match your dataset folder names)
class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index] * 100

    result = class_names[predicted_index]

    return result.upper(), round(confidence, 2)