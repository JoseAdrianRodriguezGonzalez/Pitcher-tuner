import os
import numpy as np
import pandas as pd
import librosa

AUDIO_DIR = {
    "train": "data/nsynth-train/audio",
    "val":   "data/nsynth-valid/audio",
    "test":  "data/nsynth-test/audio",
}

SPLITS = {
    "train": "train_df.csv",
    "val":   "val_df.csv",
    "test":  "test_df.csv",
}

def load_audio(note_str, split):
    path = os.path.join(AUDIO_DIR[split], f"{note_str}.wav")
    y, sr = librosa.load(path, sr=None)
    return y, sr

def extract_spectral(y, sr):
    """Extrae características espectrales del audio y regresa un diccionario con su media."""
    # Calcular cada descriptor con librosa.feature.*
    # Cada función regresa una matriz (1, frames), tomar la media sobre el eje del tiempo
    centroid  = None  # librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = None  # librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr       = None  # librosa.feature.zero_crossing_rate(y)
    rolloff   = None  # librosa.feature.spectral_rolloff(y=y, sr=sr)

    return {
        "centroid":  centroid,
        "bandwidth": bandwidth,
        "zcr":       zcr,
        "rolloff":   rolloff,
    }

def extract_mfcc(y, sr, n_mfcc=13):
    """Extrae los coeficientes MFCC y regresa un diccionario con la media de cada uno."""
    # Calcular los MFCC con librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # El resultado tiene forma (n_mfcc, frames), tomar la media por coeficiente
    # Convertir a columnas separadas: {"mfcc_0": ..., "mfcc_1": ..., ..., "mfcc_12": ...}
    mfccs = None

    return {}

def extract_cqt(y, sr):
    """Extrae las energías del CQT y regresa un diccionario con la media por bin."""
    # Calcular el CQT con librosa.cqt(y, sr=sr)
    # El resultado es complejo, usar np.abs para obtener la magnitud
    # Tomar la media sobre el tiempo por cada bin de frecuencia
    # Convertir a columnas separadas: {"cqt_0": ..., "cqt_1": ..., ...}
    cqt = None

    return {}

def process_file(note_str, pitch, instrument, split, mode):
    """Carga un archivo de audio y extrae sus características según el modo indicado."""
    y, sr = load_audio(note_str, split)

    if mode == "spectral":
        features = extract_spectral(y, sr)
    elif mode == "mfcc":
        features = extract_mfcc(y, sr)
    elif mode == "cqt":
        features = extract_cqt(y, sr)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    features["note_str"]              = note_str
    features["pitch"]                 = pitch
    features["instrument_family_str"] = instrument
    return features

def build_dataset(split, mode):
    """Itera sobre el split indicado y extrae características de cada audio."""
    df = pd.read_csv(SPLITS[split])
    rows = []
    for _, row in df.iterrows():
        record = process_file(
            row["note_str"],
            row["pitch"],
            row["instrument_family_str"],
            split,
            mode,
        )
        rows.append(record)
    return pd.DataFrame(rows)

def pipeline():
    # Genera 9 CSVs en processed/: una combinación de modo x split
    # Ejemplo: processed/spectral_train.csv, processed/mfcc_val.csv, etc.
    os.makedirs("processed", exist_ok=True)
    modes  = ["spectral", "mfcc", "cqt"]
    splits = ["train", "val", "test"]

    for mode in modes:
        for split in splits:
            print(f"Extracting {mode} — {split}...")
            result_df = build_dataset(split, mode)
            out_path  = f"processed/{mode}_{split}.csv"
            result_df.to_csv(out_path, index=False)
            print(f"  Saved {len(result_df)} rows -> {out_path}")
