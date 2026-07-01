import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import inference

# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the model and load the CSV mapping on startup
    try:
        inference.init_model()
    except Exception as e:
        print(f"Error during model initialization: {e}")
        # Keep loading so server doesn't crash immediately, but health endpoint will reflect failure
    yield


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="ECG Multi-label Disease Prediction API",
    description="FastAPI endpoint for timm EfficientNet-B0 ECG HR 500Hz model.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "message": "ECG Prediction API is running",
        "model_path": inference.MODEL_PATH,
        "device": inference.DEVICE,
        "endpoints": {
            "health": "/health",
            "predict_upload": "/predict/upload",
            "predict_path": "/predict/path"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": inference.model is not None,
        "model_name": inference.model_name,
        "device": inference.DEVICE,
        "num_labels": len(inference.label_cols) if inference.label_cols is not None else 0,
        "signal_length": inference.signal_length,
        "image_size": inference.image_size
    }


@app.post("/predict/path")
def predict_from_path(
    file_path: str = Form(...)
):
    try:
        path = Path(file_path)

        # For WFDB base path, file may not exist directly,
        # but .hea/.dat exists.
        suffix = path.suffix.lower()

        if suffix in [".dcm", ".mat"]:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

        if suffix == ".hea":
            if not path.exists():
                raise FileNotFoundError(f"HEA file not found: {file_path}")

        if suffix == ".dat":
            if not path.exists():
                raise FileNotFoundError(f"DAT file not found: {file_path}")

        if suffix == "":
            hea_path = Path(str(path) + ".hea")
            dat_path = Path(str(path) + ".dat")

            if not hea_path.exists() and not dat_path.exists():
                raise FileNotFoundError(
                    f"WFDB base path not found. Expected {hea_path} and {dat_path}"
                )

        result = inference.run_prediction(path)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/upload")
async def predict_from_upload(
    file: UploadFile = File(...)
):
    saved_path = None
    try:
        suffix = Path(file.filename).suffix.lower()

        if suffix not in [".dcm", ".mat", ".hea", ".dat"]:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Use .dcm, .mat, .hea, or .dat"
            )

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            saved_path = Path(temp_file.name)

        result = inference.run_prediction(saved_path)
        return result

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if saved_path and saved_path.exists():
            try:
                saved_path.unlink()
            except Exception as e:
                print(f"Error deleting temporary file {saved_path}: {e}")


if __name__ == "__main__":
    import uvicorn
    # Defaults to port 8089 as requested
    uvicorn.run("main:app", host="0.0.0.0", port=8089, reload=True)
