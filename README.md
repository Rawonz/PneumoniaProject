# 🩺 Pneumonia Detection System

An AI-powered web application for detecting pneumonia from chest X-ray images using a VGG19 deep learning model. The project combines a React frontend, FastAPI backend, and TensorFlow/Keras model to provide fast and accurate predictions.

> **Disclaimer**
>
> This project is intended for **research and educational purposes only**. It is **not a certified medical device** and must not be used for clinical diagnosis.

---

# ✨ Features

- 🧠 AI-powered chest X-ray analysis
- 📷 Upload JPEG or PNG X-ray images
- ⚡ FastAPI REST API
- 🎨 Modern React + TypeScript interface
- 📊 Confidence score for every prediction
- 📈 Class probability visualization
- 🔍 Binary classification (NORMAL / PNEUMONIA)
- 📱 Responsive design

---

# 📊 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **88%** |
| Precision | **89%** |
| Recall | **93%** |
| F1 Score | **0.9089** |

---

# 🛠 Tech Stack

## Machine Learning

- TensorFlow
- Keras
- VGG19 Transfer Learning
- NumPy
- Pillow

## Backend

- FastAPI
- Uvicorn
- Python

## Frontend

- React
- TypeScript
- Vite
- Tailwind CSS

---

# 📁 Project Structure

```
PneumoniaProject/
│
├── backend/
│   ├── main.py
│   └── requirements.txt
│
├── ml/
│   ├── train.py
│   ├── evaluate.py
│   ├── eda.py
│   ├── pneumonia_vgg19.ipynb
│   └── outputs/
│
├── src/
├── public/
├── package.json
├── README.md
└── .gitignore
```

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/Rawonz/PneumoniaProject.git

cd PneumoniaProject
```

---

## 2. Backend Setup

Create a virtual environment

```bash
python -m venv venv
```

Activate it

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r backend/requirements.txt
```

Run the API

```bash
cd backend

python -m uvicorn main:app --reload
```

Backend:

```
http://127.0.0.1:8000
```

Swagger Documentation:

```
http://127.0.0.1:8000/docs
```

---

## 3. Frontend Setup

Install packages

```bash
npm install
```

Start development server

```bash
npm run dev
```

Frontend:

```
http://localhost:5173
```

---

# 📖 Usage

## Step 1

Run the backend.

```bash
python -m uvicorn main:app --reload
```

---

## Step 2

Run the frontend.

```bash
npm run dev
```

---

## Step 3

Open

```
http://localhost:5173
```

---

## Step 4

Upload a chest X-ray image.

Supported formats

- JPEG
- JPG
- PNG

Maximum file size

```
10 MB
```

---

## Step 5

Wait for the prediction.

The application returns

- Prediction
- Confidence Score
- Class Probabilities
- Inference Time

---

# 🌐 REST API

## POST /predict

Upload a chest X-ray image.

Example response

```json
{
  "prediction": "NORMAL",
  "confidence": 0.94,
  "probabilities": {
    "NORMAL": 0.94,
    "PNEUMONIA": 0.06
  },
  "inference_ms": 289.6
}
```

---

## GET /health

Returns API health status.

Example

```json
{
  "status": "ok",
  "model_ready": true
}
```

---

# 🧠 Model

The application uses **VGG19 Transfer Learning** for binary image classification.

Classes

```
NORMAL
PNEUMONIA
```

Training includes

- Data preprocessing
- Data augmentation
- Transfer Learning
- Fine-tuning
- Binary classification

---

# 📈 Training Outputs

After training, the following files are generated

```
training_curves.png
confusion_matrix.png
metrics.json
training_log.csv
class_indices.json
```

---

# ⚠️ Model Files

Large model files are **not included** in this repository because they exceed GitHub's file size limit.

To enable inference, place the following files inside

```
ml/outputs/
```

Required files

```
pneumonia_vgg19_final.h5
class_indices.json
```

---

# 📌 Future Improvements

- Docker support
- JWT Authentication
- User accounts
- Cloud deployment
- Grad-CAM visualization
- Multi-class disease detection
- Model optimization
- Better UI animations

---

# 👨‍💻 Author

**RawonzExE**

GitHub

https://github.com/Rawonz

---

# 📄 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

Feedback and contributions are always welcome.
