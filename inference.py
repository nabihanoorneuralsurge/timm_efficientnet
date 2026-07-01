from pathlib import Path
import csv

import numpy as np

import torch
import torch.nn.functional as F

import timm
import wfdb
import pydicom
from scipy.io import loadmat
from scipy.signal import resample


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "timm_efficientnet_b0_imagenet_ecg_multilabel_hr_500hz_best.pth"

# Forced to CPU as requested
DEVICE = "cpu"

STANDARD_LEADS = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

print("Device:", DEVICE)


# ============================================================
# MODEL GLOBALS
# ============================================================

model = None
label_cols = None
signal_length = None
image_size = None
model_name = None
label_mapping = {}


# ============================================================
# LOAD MODEL & LABELS MAPPING
# ============================================================

def load_label_mapping(csv_path="scp_statements.csv"):
    mapping = {}
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Warning: Label mapping file {csv_path} not found.")
        return mapping

    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                label = row[0].strip()
                description = row[1].strip()
                mapping[label] = description
    except Exception as e:
        print(f"Error loading label mapping from CSV: {e}")
    return mapping


def load_timm_model(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    labels = checkpoint["label_cols"]
    num_classes = len(labels)

    loaded_model_name = checkpoint.get("model_name", None)

    if loaded_model_name is None:
        loaded_model_name = checkpoint.get("config", {}).get("model", "tf_efficientnet_b0")

    loaded_signal_length = int(
        checkpoint.get(
            "signal_length",
            checkpoint.get("config", {}).get("signal_length", 5000)
        )
    )

    loaded_image_size = int(
        checkpoint.get(
            "image_size",
            checkpoint.get("config", {}).get("image_size", 224)
        )
    )

    loaded_model = timm.create_model(
        loaded_model_name,
        pretrained=False,
        num_classes=num_classes
    )

    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.to(DEVICE)
    loaded_model.eval()

    print("\nModel loaded successfully")
    print("Checkpoint:", checkpoint_path)
    print("Model name:", loaded_model_name)
    print("Labels:", num_classes)
    print("Signal length:", loaded_signal_length)
    print("Image size:", loaded_image_size)

    return loaded_model, labels, loaded_signal_length, loaded_image_size, loaded_model_name


def init_model(checkpoint_path=MODEL_PATH, csv_path="scp_statements.csv"):
    global model, label_cols, signal_length, image_size, model_name, label_mapping
    model, label_cols, signal_length, image_size, model_name = load_timm_model(checkpoint_path)
    label_mapping = load_label_mapping(csv_path)


# ============================================================
# COMMON SIGNAL PROCESSING
# ============================================================

def ensure_samples_by_leads(signal):
    """
    Converts ECG to shape: (samples, 12)
    """
    signal = np.asarray(signal)

    if signal.ndim != 2:
        raise ValueError(f"Expected 2D ECG signal, got shape: {signal.shape}")

    # If shape is (12, samples), convert to (samples, 12)
    if signal.shape[0] == 12 and signal.shape[1] != 12:
        signal = signal.T

    # If more than 12 channels, keep first 12
    if signal.shape[1] > 12:
        signal = signal[:, :12]

    # If fewer than 12 channels, pad zeros
    if signal.shape[1] < 12:
        pad = 12 - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad)), mode="constant")

    return signal.astype(np.float32)


def preprocess_signal_for_timm(signal, target_signal_length=5000, target_image_size=224):
    """
    Raw ECG -> model input.

    Input:
        ECG shape: (samples, 12) or (12, samples)

    Output:
        Tensor shape: (1, 3, 224, 224)
    """
    signal = ensure_samples_by_leads(signal)

    signal = np.nan_to_num(
        signal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    original_samples = int(signal.shape[0])
    original_leads = int(signal.shape[1])

    was_resampled = False

    # For HR 500Hz model, every ECG becomes 5000 samples
    if signal.shape[0] != target_signal_length:
        signal = resample(signal, target_signal_length, axis=0)
        was_resampled = True

    # Same normalization as training
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)

    std[std < 1e-8] = 1.0

    signal = (signal - mean) / std

    signal = np.nan_to_num(
        signal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # (5000, 12) -> (12, 5000)
    signal = signal.T.astype(np.float32)

    x = torch.tensor(signal, dtype=torch.float32)

    # (12, 5000) -> (1, 1, 12, 5000)
    x = x.unsqueeze(0).unsqueeze(0)

    # (1, 1, 12, 5000) -> (1, 1, 224, 224)
    x = F.interpolate(
        x,
        size=(target_image_size, target_image_size),
        mode="bilinear",
        align_corners=False
    )

    # (1, 1, 224, 224) -> (1, 3, 224, 224)
    x = x.repeat(1, 3, 1, 1)

    x = torch.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    preprocess_info = {
        "original_samples": original_samples,
        "original_leads": original_leads,
        "target_samples": int(target_signal_length),
        "image_size": int(target_image_size),
        "was_resampled": was_resampled,
        "final_input_shape": list(x.shape)
    }

    return x, preprocess_info


# ============================================================
# WFDB .DAT + .HEA LOADER
# ============================================================

def load_wfdb_record(path):
    """
    Supports:
    records500/00000/00001_hr
    records500/00000/00001_hr.hea
    records500/00000/00001_hr.dat
    records100/00000/00001_lr
    """
    path = Path(path)

    if path.suffix.lower() in [".hea", ".dat"]:
        record_base = path.with_suffix("")
    else:
        record_base = path

    signal, meta = wfdb.rdsamp(str(record_base))

    loader_info = {
        "loader": "wfdb",
        "record_base": str(record_base),
        "shape": list(signal.shape),
        "sampling_rate": meta.get("fs", None),
        "leads": meta.get("sig_name", None)
    }

    return signal, loader_info


# ============================================================
# MAT + HEA LOADER
# ============================================================

def load_mat_record(path):
    """
    Supports .mat file with key 'val'.
    Also supports .hea if matching .mat exists.
    """
    path = Path(path)

    if path.suffix.lower() == ".hea":
        mat_path = path.with_suffix(".mat")
    elif path.suffix.lower() == ".mat":
        mat_path = path
    else:
        mat_path = path.with_suffix(".mat")

    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    mat = loadmat(str(mat_path))

    if "val" not in mat:
        raise ValueError(f"MAT file does not contain key 'val': {mat_path}")

    signal = mat["val"]
    signal = ensure_samples_by_leads(signal)

    loader_info = {
        "loader": "mat",
        "mat_path": str(mat_path),
        "shape": list(signal.shape)
    }

    return signal, loader_info


# ============================================================
# DICOM ECG WAVEFORM LOADER
# ============================================================

def clean_lead_name(name):
    name = str(name).strip()
    lower = name.lower()

    if lower in ["i", "lead i", "lead i (einthoven)"]:
        return "I"
    if lower in ["ii", "lead ii"]:
        return "II"
    if lower in ["iii", "lead iii"]:
        return "III"

    # Important: III before II before I
    if "lead iii" in lower:
        return "III"
    if "lead ii" in lower:
        return "II"
    if "lead i" in lower:
        return "I"

    if "avr" in lower:
        return "aVR"
    if "avl" in lower:
        return "aVL"
    if "avf" in lower:
        return "aVF"

    for lead in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        if lead.lower() in lower:
            return lead

    return name


def get_dicom_lead_names(waveform_item):
    lead_names = []

    if not hasattr(waveform_item, "ChannelDefinitionSequence"):
        return lead_names

    for ch in waveform_item.ChannelDefinitionSequence:
        lead_name = None

        if hasattr(ch, "ChannelSourceSequence"):
            src = ch.ChannelSourceSequence[0]
            lead_name = getattr(src, "CodeMeaning", None)

        if lead_name is None:
            lead_name = f"CH{len(lead_names) + 1}"

        lead_names.append(clean_lead_name(lead_name))

    return lead_names


def choose_dicom_waveform_item(ds):
    if not hasattr(ds, "WaveformSequence"):
        raise ValueError("DICOM has no WaveformSequence. This may be image-only DICOM.")

    candidates = []

    for item in ds.WaveformSequence:
        channels = int(getattr(item, "NumberOfWaveformChannels", 0))
        samples = int(getattr(item, "NumberOfWaveformSamples", 0))
        originality = str(getattr(item, "WaveformOriginality", "")).upper()
        label = str(getattr(item, "MultiplexGroupLabel", "")).upper()

        score = 0

        if channels == 12:
            score += 10

        if originality == "ORIGINAL":
            score += 10

        if "RHYTHM" in label:
            score += 5

        score += samples / 10000.0

        candidates.append((score, item))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

    return candidates[0][1]


def load_dicom_waveform(path):
    path = Path(path)

    ds = pydicom.dcmread(str(path))

    item = choose_dicom_waveform_item(ds)

    num_channels = int(item.NumberOfWaveformChannels)
    num_samples = int(item.NumberOfWaveformSamples)
    fs = float(getattr(item, "SamplingFrequency", 0))

    bits_allocated = int(getattr(item, "WaveformBitsAllocated", 16))
    interpretation = str(getattr(item, "WaveformSampleInterpretation", "SS")).upper()

    raw = item.WaveformData

    if bits_allocated == 16:
        dtype = np.uint16 if interpretation == "US" else np.int16
    elif bits_allocated == 8:
        dtype = np.uint8 if interpretation == "UB" else np.int8
    else:
        raise ValueError(f"Unsupported DICOM waveform bits allocated: {bits_allocated}")

    signal = np.frombuffer(raw, dtype=dtype)

    expected = num_samples * num_channels

    if signal.size != expected:
        signal = signal[:expected]

    signal = signal.reshape(num_samples, num_channels).astype(np.float32)

    # Apply DICOM channel scaling
    if hasattr(item, "ChannelDefinitionSequence"):
        for ch_idx, ch_def in enumerate(item.ChannelDefinitionSequence):
            sensitivity = float(getattr(ch_def, "ChannelSensitivity", 1.0))
            correction = float(getattr(ch_def, "ChannelSensitivityCorrectionFactor", 1.0))
            baseline = float(getattr(ch_def, "ChannelBaseline", 0.0))

            signal[:, ch_idx] = (signal[:, ch_idx] - baseline) * sensitivity * correction

    lead_names = get_dicom_lead_names(item)

    missing_leads = []

    # Reorder to standard lead order
    if len(lead_names) == signal.shape[1]:
        lead_map = {lead: idx for idx, lead in enumerate(lead_names)}

        ordered = []

        for lead in STANDARD_LEADS:
            if lead in lead_map:
                ordered.append(signal[:, lead_map[lead]])
            else:
                missing_leads.append(lead)
                ordered.append(np.zeros(signal.shape[0], dtype=np.float32))

        signal = np.stack(ordered, axis=1)

    signal = ensure_samples_by_leads(signal)

    loader_info = {
        "loader": "dicom",
        "path": str(path),
        "modality": getattr(ds, "Modality", "unknown"),
        "channels": num_channels,
        "samples": num_samples,
        "sampling_frequency": fs,
        "originality": getattr(item, "WaveformOriginality", "unknown"),
        "group_label": getattr(item, "MultiplexGroupLabel", "unknown"),
        "bits_allocated": bits_allocated,
        "interpretation": interpretation,
        "dicom_leads": lead_names,
        "missing_standard_leads": missing_leads,
        "converted_shape": list(signal.shape)
    }

    return signal, loader_info


# ============================================================
# AUTO ECG LOADER
# ============================================================

def load_any_ecg(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".dcm":
        return load_dicom_waveform(path)

    if suffix == ".mat":
        return load_mat_record(path)

    if suffix == ".hea":
        mat_path = path.with_suffix(".mat")
        dat_path = path.with_suffix(".dat")

        if mat_path.exists():
            return load_mat_record(path)

        if dat_path.exists():
            return load_wfdb_record(path)

        raise FileNotFoundError(f"No matching .mat or .dat file found for: {path}")

    if suffix == ".dat":
        return load_wfdb_record(path)

    # No suffix means WFDB record base path
    return load_wfdb_record(path)


# ============================================================
# PREDICTION
# ============================================================

def get_confidence(prob: float) -> str:
    if prob >= 0.8:
        return "high"
    elif prob >= 0.6:
        return "medium"
    else:
        return "low"


@torch.no_grad()
def run_prediction(file_path):
    global model, label_cols, signal_length, image_size, model_name, label_mapping
    import os
    from pathlib import Path

    # Dynamically load .env file if it exists in the workspace
    env_path = Path("main.py").parent / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip()
        except Exception as e:
            print(f"Error reading .env: {e}")

    try:
        threshold = float(os.getenv("THRESHOLD", 0.5))
    except ValueError:
        threshold = 0.5

    raw_signal, loader_info = load_any_ecg(file_path)

    x, preprocess_info = preprocess_signal_for_timm(
        signal=raw_signal,
        target_signal_length=signal_length,
        target_image_size=image_size
    )

    x = x.to(DEVICE)

    logits = model(x)

    logits = torch.nan_to_num(
        logits,
        nan=0.0,
        posinf=20.0,
        neginf=-20.0
    )

    probs = torch.sigmoid(logits).cpu().numpy()[0]

    probs = np.nan_to_num(
        probs,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    detected_labels = []

    for label, prob in zip(label_cols, probs):
        prob = float(prob)

        if prob >= threshold:
            lbl_str = str(label)
            desc = label_mapping.get(lbl_str, lbl_str)  # fallback to label if no mapping found
            detected_labels.append({
                "label": lbl_str,
                "description": desc,
                "probability": round(prob, 4),
                "confidence": get_confidence(prob)
            })

    # Sort top predictions by probability descending
    detected_labels = sorted(
        detected_labels,
        key=lambda x: x["probability"],
        reverse=True
    )

    return {"labels": detected_labels}