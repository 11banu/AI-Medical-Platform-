import tensorflow as tf

try:
    model = tf.keras.models.load_model("model/brain_model.keras")
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print("❌ Error:", e)