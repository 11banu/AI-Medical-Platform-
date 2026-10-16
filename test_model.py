from predict import predict_heart_disease

sample_patient = [
    63,   # age
    1,    # sex
    3,    # cp
    145,  # trestbps
    233,  # chol
    1,    # fbs
    0,    # restecg
    150,  # thalach
    0,    # exang
    2.3,  # oldpeak
    0,    # slope
    0,    # ca
    1     # thal
]

result, confidence = predict_heart_disease(sample_patient)

print("\n❤️ AI Heart Disease Analysis")
print("----------------------------")
print("Prediction :", result)
print("Confidence :", confidence, "%")