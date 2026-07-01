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
        return {"labels": result["labels"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/upload")
async def predict_from_upload(
    dcm: UploadFile = File(None),
    mat: UploadFile = File(None),
    hea: UploadFile = File(None),
    dat: UploadFile = File(None)
):
    # Ensure at least one file is uploaded
    if dcm is None and mat is None and hea is None and dat is None:
        raise HTTPException(
            status_code=400,
            detail="No files uploaded. Please upload a .dcm, .mat, or .hea + .dat files."
        )

    try:
        # Create a temporary directory that will be deleted automatically at the end of the context
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Save all files that were uploaded to the temporary directory
            if dcm is not None:
                with (temp_dir_path / dcm.filename).open("wb") as buffer:
                    shutil.copyfileobj(dcm.file, buffer)
            if mat is not None:
                with (temp_dir_path / mat.filename).open("wb") as buffer:
                    shutil.copyfileobj(mat.file, buffer)
            if hea is not None:
                with (temp_dir_path / hea.filename).open("wb") as buffer:
                    shutil.copyfileobj(hea.file, buffer)
            if dat is not None:
                with (temp_dir_path / dat.filename).open("wb") as buffer:
                    shutil.copyfileobj(dat.file, buffer)

            # Determine the target entry point file for prediction
            target_file = None

            if hea is not None:
                # For both .mat + .hea and .dat + .hea, the .hea file is the primary path.
                # In inference.py, load_any_ecg will automatically check if .mat or .dat exists
                # in the same folder and load it accordingly.
                target_file = temp_dir_path / hea.filename
            elif mat is not None:
                # If only .mat is uploaded
                target_file = temp_dir_path / mat.filename
            elif dcm is not None:
                # If only .dcm is uploaded
                target_file = temp_dir_path / dcm.filename
            else:
                # Only dat was uploaded without hea
                raise HTTPException(
                    status_code=400,
                    detail="For WFDB records (.dat), the header file (.hea) must also be uploaded."
                )

            result = inference.run_prediction(target_file)
            return {"labels": result["labels"]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Defaults to port 8089 as requested
    uvicorn.run("main:app", host="0.0.0.0", port=8089, reload=True)
