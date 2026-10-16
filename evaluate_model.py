import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
model_path = os.path.join(BASE_DIR, "model", "model.h5")
model = tf.keras.models.load_model(model_path)

# Test dataset
test_path = os.path.join(BASE_DIR, "dataset", "test")

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Predictions
probabilities = model.predict(test_data)
y_prob = probabilities.ravel()

y_pred = (y_prob >= 0.5).astype(int)
y_true = test_data.classes

# Classification report
print("=" * 50)
print("PNEUMONIA CLASSIFICATION REPORT")
print("=" * 50)

print(classification_report(
    y_true,
    y_pred,
    target_names=["NORMAL", "PNEUMONIA"]
))

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# ROC-AUC
roc_auc = roc_auc_score(y_true, y_prob)
print(f"ROC-AUC: {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Save confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm)

plt.title("Pneumonia Prediction - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xticks([0, 1], ["NORMAL", "PNEUMONIA"])
plt.yticks([0, 1], ["NORMAL", "PNEUMONIA"])

plt.tight_layout()

plt.savefig(
    os.path.join(BASE_DIR, "confusion_matrix.png"),
    dpi=300
)

plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.3f}"
)

plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Pneumonia Prediction - ROC Curve")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(BASE_DIR, "roc_curve.png"),
    dpi=300
)

plt.show()