# ECG Disease Classification API

A FastAPI-based inference service for multi-label ECG disease classification using an EfficientNet-B0 model trained with PyTorch.

The API supports multiple ECG input formats and returns disease predictions with probabilities.

---

## Features

- FastAPI REST API
- EfficientNet-B0 PyTorch model
- Multi-label ECG classification
- Supports:
  - DICOM (.dcm)
  - MATLAB (.mat)
  - WFDB (.hea + .dat)
- Returns:
  - Detected diseases
  - Top 10 predictions
  - Prediction probabilities

---

## Project Structure

```
.
├── app.py
├── inference.py
├── requirements.txt
├── scp_statements.csv
├── saved_models/
│   └── model.pth
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/<username>/<repository>.git
cd <repository>
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the API

```bash
uvicorn app:app --host 0.0.0.0 --port 5055 --reload
```

Swagger UI

```
http://localhost:5055/docs
```

---

## API Endpoint

### POST /predict

Supported inputs

- DICOM (.dcm)
- MATLAB (.mat)
- WFDB (.hea + .dat)

---

## Example Response

```json
{
  "threshold": 0.5,
  "detected_labels": [
    {
      "code": "LVH",
      "statement_category": "Ventricular Hypertrophy",
      "probability": 0.8796
    }
  ],
  "top_10_predictions": [
    {
      "code": "LVH",
      "statement_category": "Ventricular Hypertrophy",
      "probability": 0.8796
    }
  ]
}
```

---

## Model

- Architecture: EfficientNet-B0
- Framework: PyTorch
- Task: Multi-label ECG classification

---

## Technologies Used

- Python
- FastAPI
- PyTorch
- timm
- NumPy
- Pandas
- OpenCV
- SciPy
- pydicom
- WFDB

---

## Notes

This repository contains the inference API only.

Model training is not included.
