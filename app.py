from flask import Flask, render_template, request
import os
from predict import predict_image

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded."

    file = request.files["image"]

    if file.filename == "":
        return "No file selected."

    # Save uploaded image
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    print("Saving file to:", filepath)

    file.save(filepath)

    # Predict
    result, confidence = predict_image(filepath)

    # Display result
    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        image=file.filename
    )


if __name__ == "__main__":
    app.run(debug=True)