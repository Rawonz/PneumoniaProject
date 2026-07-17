# 🩺 Pneumonia Detection System

An AI-powered web application for detecting pneumonia from chest X-ray images using deep learning. The project uses a VGG19 transfer learning model trained on chest X-ray images and provides predictions through a FastAPI backend with a modern React frontend.

---

## ✨ Features

- 🧠 VGG19 Transfer Learning
- 📷 Chest X-ray image upload
- ⚡ FastAPI REST API
- 🎨 Modern React + Vite frontend
- 📊 Training metrics visualization
- 📈 Confusion matrix generation
- 🎯 Confidence score prediction
- 🔍 Binary classification (NORMAL / PNEUMONIA)

---

## 🖼️ Demo

Upload a chest X-ray image and receive:

- Prediction (NORMAL / PNEUMONIA)
- Confidence score
- Class probabilities
- Inference time

---

## 📊 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **88%** |
| Precision | **89%** |
| Recall | **93%** |
| F1 Score | **0.9089** |

---

## 🛠️ Tech Stack

### Machine Learning

- TensorFlow
- Keras
- VGG19
- NumPy
- Pillow

### Backend

- FastAPI
- Uvicorn
- Python

### Frontend

- React
- TypeScript
- Vite

---

## 📁 Project Structure

```
PneumoniaProject
│
├── backend/
│   ├── main.py
│   └── requirements.txt
│
├── ml/
│   ├── train.py
│   ├── evaluate.py
│   ├── eda.py
│   └── pneumonia_vgg19.ipynb
│
├── src/
├── public/
├── README.md
└── package.json
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/Rawonz/PneumoniaProject.git
cd PneumoniaProject
```

### Backend

```bash
cd backend

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python -m uvicorn main:app --reload
```

Backend will run on:

```
http://127.0.0.1:8000
```

---

### Frontend

```bash
npm install

npm run dev
```

Frontend will run on:

```
http://localhost:5173
```

---

## 📷 API Endpoint

### POST /predict

Upload a chest X-ray image.

Example response:

```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.9821,
  "probabilities": {
    "NORMAL": 0.0179,
    "PNEUMONIA": 0.9821
  },
  "inference_ms": 42.8
}
```

---

## 📈 Training

The model was trained using transfer learning with **VGG19** and binary classification.

Training outputs include:

- Training Curves
- Confusion Matrix
- Metrics JSON
- Class Indices

---

## ⚠️ Note

The trained model (.h5) and dataset are not included in this repository because of GitHub file size limitations.

---

## 📌 Future Improvements

- Docker support
- User authentication
- Cloud deployment
- Model optimization
- Explainable AI (Grad-CAM)
- Multi-class disease detection

---

## 👨‍💻 Author

**Uğur Aleskerli**

GitHub:
https://github.com/Rawonz

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
