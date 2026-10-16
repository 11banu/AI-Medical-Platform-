# 🫁 Pneumonia Detection using Deep Learning

An AI-powered web application that detects **Pneumonia** from **Chest X-ray images** using a Convolutional Neural Network (CNN). The application provides an intuitive web interface where users can upload an X-ray image and receive an AI-generated prediction with a confidence score.

---

## 📌 Features

- Upload Chest X-ray images
- Detect **Normal** or **Pneumonia**
- Display prediction confidence
- Simple and responsive web interface
- Built using Flask and TensorFlow
- Easy to extend with additional disease modules

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Backend | Flask |
| Frontend | HTML5, CSS3 |
| Image Processing | OpenCV, NumPy |
| Model | Convolutional Neural Network (CNN) |

---

## 📂 Project Structure

```text
pneumonia_detection/
│
├── dataset/
│   ├── train/
│   ├── test/
│   └── val/
│
├── model/
│   └── model.h5
│
├── static/
│   ├── uploads/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── predict.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## 🧠 Model Overview

The model is trained using a **Convolutional Neural Network (CNN)** for binary image classification.

### Classes

- Normal
- Pneumonia

### Image Size

- **224 × 224 pixels**

### Output

- Predicted Class
- Confidence Score

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/pneumonia-detection.git
```

### 2. Navigate to the project directory

```bash
cd pneumonia_detection
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Flask server:

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📸 Application Workflow

1. Upload a Chest X-ray image.
2. The trained CNN processes the image.
3. The model predicts:
   - Normal
   - Pneumonia
4. The prediction and confidence score are displayed.

---

## 📊 Future Enhancements

- Tuberculosis Detection
- COVID-19 Detection
- Multi-Disease Classification
- Patient Login System
- Doctor Dashboard
- PDF Medical Report Generation
- Cloud Deployment

---

## 📖 Learning Objectives

This project demonstrates:

- Deep Learning using CNN
- Medical Image Classification
- Flask Web Development
- Image Preprocessing
- Model Deployment

---

## 👩‍💻 Author

**Bhanushree R**

Final Year Information Science Engineering Student

---

## 📄 License

This project is developed for educational and research purposes.