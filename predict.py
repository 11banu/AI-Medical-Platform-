import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Load trained model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "model.h5"
)

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        result = "PNEUMONIA"
        confidence = prediction[0][0] * 100
    else:
        result = "NORMAL"
        confidence = (1 - prediction[0][0]) * 100

    return result, round(confidence, 2)