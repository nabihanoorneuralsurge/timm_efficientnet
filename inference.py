from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import timm
import wfdb
import pydicom
from scipy.io import loadmat
from scipy.signal import resample


# ============================================================
# CHANGE THESE PATHS ONLY
# ============================================================

MODEL_PATH = r"saved_models\timm_efficientnet_b0_imagenet_ecg_multilabel_hr_500hz_best.pth"

INPUT_PATH = r"ECG\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\records500\00000\00001_hr"
# Examples:
# INPUT_PATH = r"JS00001.mat"
# INPUT_PATH = r"JS00001.hea"
# INPUT_PATH = r"your_ecg_file.dcm"
# INPUT_PATH = r"ECG\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\records500\00000\00001_hr"
# INPUT_PATH = r"ECG\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\records500\00000\00001_hr.hea"

THRESHOLD = 0.5
TOP_K = 10


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STANDARD_LEADS = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

print("Device:", DEVICE)


# ============================================================
# LOAD TIMM MODEL
# ============================================================

def load_timm_model_notebook(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    label_cols = checkpoint["label_cols"]
    num_classes = len(label_cols)

    model_name = checkpoint.get("model_name", None)

    if model_name is None:
        config = checkpoint.get("config", {})
        model_name = config.get("model", "tf_efficientnet_b0")

    signal_length = int(
        checkpoint.get(
            "signal_length",
            checkpoint.get("config", {}).get("signal_length", 5000)
        )
    )

    image_size = int(
        checkpoint.get(
            "image_size",
            checkpoint.get("config", {}).get("image_size", 224)
        )
    )

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Loaded model:", checkpoint_path)
    print("Model name:", model_name)
    print("Number of labels:", num_classes)
    print("Signal length from checkpoint:", signal_length)
    print("Image size from checkpoint:", image_size)

    return model, label_cols, signal_length, image_size


# ============================================================
# COMMON SIGNAL PROCESSING
# ============================================================

def ensure_samples_by_leads(signal):
    """
    Output shape: (samples, 12)
    """
    signal = np.asarray(signal)

    if signal.ndim != 2:
        raise ValueError(f"Expected 2D ECG signal, got shape: {signal.shape}")

    # If shape is (12, samples), convert to (samples, 12)
    if signal.shape[0] == 12 and signal.shape[1] != 12:
        signal = signal.T

    # Keep first 12 channels if more than 12
    if signal.shape[1] > 12:
        signal = signal[:, :12]

    # Pad if fewer than 12 channels
    if signal.shape[1] < 12:
        pad = 12 - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad)), mode="constant")

    return signal.astype(np.float32)


def preprocess_signal_for_timm(signal, signal_length=5000, image_size=224):
    """
    Raw ECG -> model input

    Input raw signal:
        (samples, 12) or (12, samples)

    Output tensor:
        (1, 3, 224, 224)
    """
    signal = ensure_samples_by_leads(signal)

    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    print("Raw standardized signal shape:", signal.shape)

    # For your HR model, every input is resampled to 5000 samples
    if signal.shape[0] != signal_length:
        print(f"Resampling from {signal.shape[0]} samples to {signal_length} samples")
        signal = resample(signal, signal_length, axis=0)

    # Same normalization as training
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True) + 1e-8
    signal = (signal - mean) / std

    # (5000, 12) -> (12, 5000)
    signal = signal.T.astype(np.float32)

    # (12, 5000) -> tensor
    x = torch.tensor(signal, dtype=torch.float32)

    # (12, 5000) -> (1, 1, 12, 5000)
    x = x.unsqueeze(0).unsqueeze(0)

    # (1, 1, 12, 5000) -> (1, 1, 224, 224)
    x = F.interpolate(
        x,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False
    )

    # (1, 1, 224, 224) -> (1, 3, 224, 224)
    x = x.repeat(1, 3, 1, 1)

    return x


# ============================================================
# WFDB .DAT + .HEA LOADER
# ============================================================

def load_wfdb_record(path):
    path = Path(path)

    if path.suffix.lower() in [".hea", ".dat"]:
        record_base = path.with_suffix("")
    else:
        record_base = path

    signal, meta = wfdb.rdsamp(str(record_base))

    print("\nLoaded WFDB record:", record_base)
    print("Shape:", signal.shape)
    print("Sampling rate:", meta.get("fs", "unknown"))
    print("Leads:", meta.get("sig_name", "unknown"))

    return signal


# ============================================================
# MAT + HEA LOADER
# ============================================================

def load_mat_record(path):
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
        raise ValueError("MAT file does not contain key 'val'.")

    signal = mat["val"]

    print("\nLoaded MAT file:", mat_path)
    print("Raw MAT shape:", signal.shape)

    signal = ensure_samples_by_leads(signal)

    print("Converted MAT shape:", signal.shape)

    return signal


# ============================================================
# DICOM ECG WAVEFORM LOADER
# ============================================================

def clean_lead_name(name):
    name = str(name).strip()
    lower = name.lower()

    # Exact/simple first
    if lower in ["i", "lead i", "lead i (einthoven)"]:
        return "I"
    if lower in ["ii", "lead ii"]:
        return "II"
    if lower in ["iii", "lead iii"]:
        return "III"

    # Important: check III before II before I
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
    """
    Prefer ORIGINAL 12-channel rhythm waveform.
    """
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

    print("\nLoaded DICOM:", path)
    print("Modality:", getattr(ds, "Modality", "unknown"))
    print("Has WaveformSequence:", hasattr(ds, "WaveformSequence"))

    item = choose_dicom_waveform_item(ds)

    num_channels = int(item.NumberOfWaveformChannels)
    num_samples = int(item.NumberOfWaveformSamples)
    fs = float(getattr(item, "SamplingFrequency", 0))

    bits_allocated = int(getattr(item, "WaveformBitsAllocated", 16))
    interpretation = str(getattr(item, "WaveformSampleInterpretation", "SS")).upper()

    print("Selected waveform:")
    print("Channels:", num_channels)
    print("Samples:", num_samples)
    print("Sampling frequency:", fs)
    print("Originality:", getattr(item, "WaveformOriginality", "unknown"))
    print("Group label:", getattr(item, "MultiplexGroupLabel", "unknown"))

    raw = item.WaveformData

    if bits_allocated == 16:
        dtype = np.uint16 if interpretation == "US" else np.int16
    elif bits_allocated == 8:
        dtype = np.uint8 if interpretation == "UB" else np.int8
    else:
        raise ValueError(f"Unsupported waveform bits allocated: {bits_allocated}")

    signal = np.frombuffer(raw, dtype=dtype)

    expected = num_samples * num_channels

    if signal.size != expected:
        print(f"Warning: signal size {signal.size} != expected {expected}")
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
    print("DICOM leads:", lead_names)

    # Reorder to standard order if possible
    if len(lead_names) == signal.shape[1]:
        lead_map = {lead: idx for idx, lead in enumerate(lead_names)}

        ordered = []

        for lead in STANDARD_LEADS:
            if lead in lead_map:
                ordered.append(signal[:, lead_map[lead]])
            else:
                ordered.append(np.zeros(signal.shape[0], dtype=np.float32))

        signal = np.stack(ordered, axis=1)

    signal = ensure_samples_by_leads(signal)

    print("DICOM converted shape:", signal.shape)

    return signal


# ============================================================
# AUTO LOADER
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

@torch.no_grad()
def predict_timm_notebook(model, label_cols, signal_length, image_size, input_path, threshold=0.5, top_k=10):
    raw_signal = load_any_ecg(input_path)

    x = preprocess_signal_for_timm(
        signal=raw_signal,
        signal_length=signal_length,
        image_size=image_size
    )

    print("\nFinal model input shape:", tuple(x.shape))

    x = x.to(DEVICE)

    logits = model(x)
    probs = torch.sigmoid(logits).cpu().numpy()[0]

    results = []

    for label, prob in zip(label_cols, probs):
        results.append({
            "label": label,
            "probability": float(prob),
            "detected": bool(prob >= threshold)
        })

    results = sorted(results, key=lambda r: r["probability"], reverse=True)
    detected = [r for r in results if r["detected"]]

    print("\n================ ECG PREDICTION ================")
    print("Input:", input_path)
    print("Threshold:", threshold)

    if detected:
        print("\nDetected labels:")
        for r in detected:
            print(f"{r['label']}: {r['probability']:.4f}")
    else:
        print("\nNo label crossed threshold.")

    print(f"\nTop {top_k} probabilities:")
    for r in results[:top_k]:
        print(f"{r['label']}: {r['probability']:.4f}")

    print("================================================")

    return results


# ============================================================
# RUN INFERENCE
# ============================================================

if __name__ == "__main__":

    model, label_cols, signal_length, image_size = load_timm_model_notebook(MODEL_PATH)

    results = predict_timm_notebook(
        model=model,
        label_cols=label_cols,
        signal_length=signal_length,
        image_size=image_size,
        input_path=INPUT_PATH,
        threshold=THRESHOLD,
        top_k=TOP_K
    )