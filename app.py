from flask import Flask, render_template, request
import joblib
import os

from heart_disease_prediction.predict import predict_heart_disease
from pneumonia_detection.predict import predict_image


app = Flask(__name__)


# =========================================================
# DIABETES MODEL
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "diabetes_prediction",
    "model",
    "diabetes_model.pkl"
)

model = joblib.load(MODEL_PATH)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# =========================================================
# DIABETES
# =========================================================

@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():

    result = None

    if request.method == "POST":

        pregnancies = float(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        blood_pressure = float(request.form["blood_pressure"])
        skin_thickness = float(request.form["skin_thickness"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        diabetes_pedigree = float(request.form["diabetes_pedigree"])
        age = float(request.form["age"])

        data = [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]]

        prediction = model.predict(data)[0]

        if prediction == 1:
            result = "Higher risk of diabetes"
        else:
            result = "Lower risk of diabetes"

    return render_template(
        "diabetes.html",
        result=result
    )


# =========================================================
# HEART DISEASE
# =========================================================

@app.route("/heart", methods=["GET", "POST"])
def heart():

    prediction = None
    confidence = None

    if request.method == "POST":

        age = float(request.form["age"])
        sex = float(request.form["sex"])
        cp = float(request.form["cp"])
        trestbps = float(request.form["trestbps"])
        chol = float(request.form["chol"])
        fbs = float(request.form["fbs"])
        restecg = float(request.form["restecg"])
        thalach = float(request.form["thalach"])
        exang = float(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = float(request.form["slope"])
        ca = float(request.form["ca"])
        thal = float(request.form["thal"])

        features = [
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]

        prediction, confidence = predict_heart_disease(features)

    return render_template(
        "heart.html",
        prediction=prediction,
        confidence=confidence
    )


# =========================================================
# PNEUMONIA
# =========================================================

@app.route("/pneumonia", methods=["GET", "POST"])
def pneumonia():

    prediction = None
    confidence = None
    image_name = None

    if request.method == "POST":

        if "image" not in request.files:
            return "No image uploaded."

        file = request.files["image"]

        if file.filename == "":
            return "No file selected."

        upload_folder = os.path.join(
            os.path.dirname(__file__),
            "pneumonia_detection",
            "static",
            "uploads"
        )

        os.makedirs(upload_folder, exist_ok=True)

        filepath = os.path.join(
            upload_folder,
            file.filename
        )

        file.save(filepath)

        prediction, confidence = predict_image(filepath)

        image_name = file.filename

    return render_template(
        "pneumonia.html",
        prediction=prediction,
        confidence=confidence,
        image=image_name
    )


# =========================================================
# RUN APPLICATION
# =========================================================
# BRAIN TUMOR
# =========================================================

from brain_tumor_detection.predict import predict_image


@app.route("/brain", methods=["GET", "POST"])
def brain():

    prediction = None
    confidence = None
    image_name = None

    if request.method == "POST":

        if "image" not in request.files:
            return "No image uploaded."

        file = request.files["image"]

        if file.filename == "":
            return "No file selected."

        upload_folder = os.path.join(
            os.path.dirname(__file__),
            "brain_tumor_detection",
            "static",
            "uploads"
        )

        os.makedirs(upload_folder, exist_ok=True)

        filepath = os.path.join(
            upload_folder,
            file.filename
        )

        file.save(filepath)

        prediction, confidence = predict_image(filepath)

        image_name = file.filename

    return render_template(
        "brain.html",
        prediction=prediction,
        confidence=confidence,
        image=image_name
    )
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)