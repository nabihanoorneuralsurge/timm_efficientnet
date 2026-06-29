from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import shutil
import pandas as pd 

from inference import (
    load_timm_model_notebook,
    predict_timm_notebook,
    MODEL_PATH,
    THRESHOLD,
    TOP_K
)

app = FastAPI(
    title="ECG Prediction API",
    version="1.0"
)

# Load model once when the server starts
model, label_cols, signal_length, image_size = load_timm_model_notebook(MODEL_PATH)
# Load the CSV
mapping_df = pd.read_csv("scp_statements.csv")
# Rename the first column
mapping_df.rename(columns={"Unnamed: 0": "code"}, inplace=True)

# Build mapping:

label_map = dict(
    zip(
        mapping_df["code"],
        mapping_df["description"]
    )
)

@app.get("/")
def home():
    return {
        "message": "ECG Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(
    dcm: UploadFile = File(None),
    mat: UploadFile = File(None),
    hea: UploadFile = File(None),
    dat: UploadFile = File(None)
):

    os.makedirs("temp", exist_ok=True)

    files_to_delete = []

    try:

        # -----------------------------
        # DICOM
        # -----------------------------
        if dcm is not None:

            temp_path = os.path.join("temp", dcm.filename)

            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(dcm.file, buffer)

            input_path = temp_path
            files_to_delete.append(temp_path)

        # -----------------------------
        # MATLAB
        # -----------------------------
        elif mat is not None:

            temp_path = os.path.join("temp", mat.filename)

            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(mat.file, buffer)

            input_path = temp_path
            files_to_delete.append(temp_path)

        # -----------------------------
        # WFDB (.hea + .dat)
        # -----------------------------
        elif hea is not None and dat is not None:

            hea_path = os.path.join("temp", hea.filename)
            dat_path = os.path.join("temp", dat.filename)

            with open(hea_path, "wb") as buffer:
                shutil.copyfileobj(hea.file, buffer)

            with open(dat_path, "wb") as buffer:
                shutil.copyfileobj(dat.file, buffer)

            input_path = hea_path

            files_to_delete.extend([hea_path, dat_path])

        else:

            raise HTTPException(
                status_code=400,
                detail="Upload either a .dcm file, a .mat file, or BOTH .hea and .dat files."
            )

        # -----------------------------
        # Prediction
        # -----------------------------
        results = predict_timm_notebook(
            model=model,
            label_cols=label_cols,
            signal_length=signal_length,
            image_size=image_size,
            input_path=input_path,
            threshold=THRESHOLD,
            top_k=TOP_K
        )

        detected = [
            {
                "code": r["label"],
                "statement_category": label_map.get(r["label"], "Unknown"),
                "probability": round(r["probability"], 4)
            }
            for r in results
            if r["detected"]
        ]

        top10 = [
            {
                "code": r["label"],
                "statement_category": label_map.get(r["label"], "Unknown"),
                "probability": round(r["probability"], 4)
            }
            for r in results[:10]
        ]
        return {
            "threshold": THRESHOLD,
            "detected_labels": detected,
            "top_10_predictions": top10
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:

        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)