# PneumoScan — AI Chest X-Ray Pneumonia Detector

> **Research prototype only — not for clinical use.**  
> A full-stack application that classifies chest X-rays as **NORMAL** or **PNEUMONIA** using a VGG19 convolutional neural network trained on the Kaggle Chest X-Ray Images dataset.

---

## Project Structure

```
├── ml/                         # Machine-learning code
│   ├── eda.py                  # Exploratory data analysis
│   ├── train.py                # Full training script (CLI)
│   ├── evaluate.py             # Post-training evaluation script
│   ├── pneumonia_vgg19.ipynb   # Jupyter notebook (end-to-end walkthrough)
│   ├── requirements.txt        # ML Python dependencies
│   └── outputs/                # Created at runtime
│       ├── best_model.h5       # Best checkpoint (val_accuracy)
│       ├── pneumonia_vgg19_final.h5  # Final exported model
│       ├── class_indices.json  # Class-name → index mapping
│       ├── metrics.json        # Test-set metrics
│       ├── training_log.csv    # Per-epoch metrics
│       ├── training_curves.png
│       └── confusion_matrix.png
│
├── backend/                    # FastAPI REST service
│   ├── main.py                 # Application entry point
│   └── requirements.txt        # Backend Python dependencies
│
├── src/                        # React + Vite frontend
│   ├── App.tsx                 # Main component (upload + results)
│   └── …                      # shadcn/ui components, CSS
│
├── .env.example                # Copy to .env to configure API URL
├── package.json
└── README.md                   # This file
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Node.js | ≥ 18 |
| Python | ≥ 3.10 |
| pip | ≥ 23 |
| GPU (optional) | CUDA-compatible for faster training |

---

## 1 · Download the Dataset

1. Create a free [Kaggle](https://www.kaggle.com) account.
2. Install the Kaggle CLI: `pip install kaggle`
3. Place your API token at `~/.kaggle/kaggle.json`.
4. Download and extract:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data
# Expected: data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}
```

---

## 2 · Train the Model

### Using the Python script (recommended for full runs)

```bash
cd ml
pip install -r requirements.txt

# Exploratory data analysis (generates charts to outputs/eda/)
python eda.py --data_dir ../data/chest_xray --output_dir outputs/eda

# Train (frozen VGG19 base, 20 epochs)
python train.py \
  --data_dir  ../data/chest_xray \
  --output_dir outputs \
  --epochs 20 \
  --batch_size 32

# Optional: fine-tune the last VGG19 block
python train.py --data_dir ../data/chest_xray --fine_tune

# Standalone evaluation on test set
python evaluate.py \
  --model_path outputs/pneumonia_vgg19_final.h5 \
  --data_dir   ../data/chest_xray
```

### Using the Jupyter Notebook

```bash
pip install jupyter
jupyter notebook ml/pneumonia_vgg19.ipynb
```

The notebook walks through EDA → augmentation → VGG19 architecture → training → evaluation with all metrics and plots inline.

### Training details

| Hyper-parameter | Value |
|-----------------|-------|
| Architecture | VGG19 (ImageNet weights, frozen base) |
| Input size | 224 × 224 RGB |
| Head | GAP → BN → Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Sigmoid |
| Optimizer | Adam, lr=1e-4 |
| Loss | Binary cross-entropy |
| Class weights | Computed from train-set frequencies |
| Augmentation | Rotation ±15°, shifts ±10%, shear, zoom, H-flip |
| Callbacks | ModelCheckpoint, EarlyStopping (patience=5), ReduceLROnPlateau |
| Fine-tune phase | Last 4 VGG19 layers, lr=1e-5 (optional flag `--fine_tune`) |

Expected test-set performance (≈5,800 images):

| Metric | Typical value |
|--------|--------------|
| Accuracy | ≥ 90% |
| Precision | ≥ 88% |
| Recall | ≥ 93% |
| F1-Score | ≥ 90% |

---

## 3 · Run the Backend API

```bash
cd backend
pip install -r requirements.txt

# Set model path (defaults to ../ml/outputs/pneumonia_vgg19_final.h5)
export MODEL_PATH=../ml/outputs/pneumonia_vgg19_final.h5
export CLASS_INDICES_PATH=../ml/outputs/class_indices.json

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Returns `{ status, model_ready }` |
| POST | `/predict` | Upload image → prediction JSON |

#### Example `POST /predict` response

```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.9421,
  "probabilities": {
    "NORMAL": 0.0579,
    "PNEUMONIA": 0.9421
  },
  "inference_ms": 142.3
}
```

#### cURL example

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@chest_xray.jpg" | jq .
```

Interactive docs are available at: `http://localhost:8000/docs`

---

## 4 · Run the Frontend

```bash
# From the project root
cp .env.example .env          # Edit VITE_API_URL if the API is not on port 8000
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

### Building for production

```bash
npm run build
# Serve the dist/ folder with any static file server
```

---

## 5 · Running Everything Together

```bash
# Terminal 1 — API
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend dev server
npm run dev
```

---

## Evaluation Artefacts

After training, `ml/outputs/` contains:

| File | Description |
|------|-------------|
| `pneumonia_vgg19_final.h5` | Keras model weights (final epoch) |
| `best_model.h5` | Best checkpoint by val_accuracy |
| `class_indices.json` | `{"NORMAL":0,"PNEUMONIA":1}` |
| `metrics.json` | Accuracy, Precision, Recall, F1 |
| `training_log.csv` | Per-epoch loss / accuracy |
| `training_curves.png` | Accuracy & Loss vs. Epoch |
| `confusion_matrix.png` | Test-set confusion matrix heatmap |

Running `evaluate.py` additionally produces:
- `roc_curve.png` — ROC curve with AUC
- `confusion_matrix_eval.png` — Re-generated confusion matrix
- `eval_metrics.json` — Including ROC-AUC

---

## Reproducibility

```bash
# Pin exact Python versions
pip freeze > requirements_frozen.txt

# Set seeds (already done in train.py / notebook)
PYTHONHASHSEED=42
```

The training script sets `random`, `numpy`, and `tensorflow` seeds to `42`. GPU non-determinism may cause minor run-to-run variation.

---

## Ethical Notice

This system is trained on a relatively small, curated dataset. It should **not** be used as a substitute for professional radiological diagnosis. False negatives (missed pneumonia) carry serious clinical risk. Always involve a qualified physician.

---

## License

MIT — see `LICENSE` for details.
