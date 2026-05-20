"""
Feature Extraction para NSynth

Genera 9 CSVs en processed/:
    spectral_train.csv / spectral_val.csv / spectral_test.csv
    mfcc_train.csv     / mfcc_val.csv     / mfcc_test.csv
    cqt_train.csv      / cqt_val.csv      / cqt_test.csv
"""
import os
import numpy as np
import pandas as pd
import librosa

#Rutas
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

#FUNCIONES
def load_audio(note_str: str, split: str):
    """Carga el .wav correspondiente a note_str en el split indicado."""
    path = os.path.join(AUDIO_DIR[split], f"{note_str}.wav")
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr

def extract_spectral(y: np.ndarray, sr: int) -> dict: #características espectrales
    """
    Cada función devuelve (1, n_frames):

        centroid  — centro de masa del espectro (Hz); indica grave/agudo
        bandwidth — dispersión alrededor del centroide; sonido limpio vs ruidoso
        zcr — tasa de cruces por cero; relacionada con agudeza / ruido
        rolloff   — frecuencia bajo la cual cae el 85 % de la energía
    """
    centroid  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    zcr       = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))

    return {
        "centroid":  centroid,
        "bandwidth": bandwidth,
        "zcr":       zcr,
        "rolloff":   rolloff,
    }

def extract_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13) -> dict:
    """
    Calcula n_mfcc coeficientes Mel-Frequency Cepstral. Retorna un dict con claves mfcc_0 … mfcc_{n_mfcc-1}.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc, frames)
    mfcc_means = np.mean(mfccs, axis=1) # (n_mfcc,)
    return {f"mfcc_{i}": float(mfcc_means[i]) for i in range(n_mfcc)}

def extract_cqt(y: np.ndarray, sr: int) -> dict:
    """
    mapea la energía del audio a bins de frecuencia espaciados logarítmicamente

    Retorna un dict con claves cqt_0 … cqt_{n_bins-1}.
    """
    C = librosa.cqt(y, sr=sr)          # (n_bins, n_frames), complejo
    C_mag = np.abs(C)                  # magnitud
    cqt_means = np.mean(C_mag, axis=1) # (n_bins,)
    return {f"cqt_{i}": float(cqt_means[i]) for i in range(len(cqt_means))}

def process_file(note_str: str, pitch: int, instrument: str,
                 split: str, mode: str) -> dict:
    """
    Carga el audio y extrae características según `mode`.

    Args:
        note_str— nombre del archivo sin extensión (e.g. 'guitar_acoustic_001-082-050')
        pitch— valor MIDI de la nota (0-127)
        instrument — familia del instrumento (e.g. 'guitar')
        split — 'train', 'val' o 'test'
        mode — 'spectral', 'mfcc' o 'cqt'
    """
    y, sr = load_audio(note_str, split)

    if mode == "spectral":
        features = extract_spectral(y, sr)
    elif mode == "mfcc":
        features = extract_mfcc(y, sr)
    elif mode == "cqt":
        features = extract_cqt(y, sr)
    else:
        raise ValueError(f"Modo desconocido: {mode}. Usa 'spectral', 'mfcc' o 'cqt'.")

    # Añadir metadatos al registro
    features["note_str"]              = note_str
    features["pitch"]                 = pitch
    features["instrument_family_str"] = instrument
    return features

def build_dataset(split: str, mode: str,
                  max_rows: int = None,
                  verbose: bool = True) -> pd.DataFrame:
    """
    Itera sobre todos los audios de un split y extrae sus características:

        split — 'train', 'val' o 'test'
        mode — 'spectral', 'mfcc' o 'cqt'
        max_rows — limita el número de filas procesadas; útil para pruebas rápidas
        verbose— imprime progreso cada 500 filas
    """
    csv_path = SPLITS[split]
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No se encontró el CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.head(max_rows)

    rows = []
    total = len(df)

    for idx, row in df.iterrows():
        if verbose and idx % 500 == 0:
            print(f" [{split}/{mode}] {idx}/{total}")
        try:
            record = process_file(
                row["note_str"],
                int(row["pitch"]),
                row["instrument_family_str"],
                split,
                mode,
            )
            rows.append(record)
        except Exception as e:
            # Si un archivo de audio no existe o está corrupto, se omite
            print(f"se omitió '{row['note_str']}' — {e}")

    return pd.DataFrame(rows)


def pipeline(max_rows: int = None):
    """
    Genera los 9 CSVs de características en processed/.
    """
    os.makedirs("processed", exist_ok=True)

    modes  = ["spectral", "mfcc", "cqt"]
    splits = ["train", "val", "test"]

    for mode in modes:
        for split in splits:
            out_path = f"processed/{mode}_{split}.csv"
            if os.path.isfile(out_path):
                print(f"Ya existe, omitiendo: {out_path}")
                continue

            print(f"\nExtrayendo [{mode}] — [{split}]")
            result_df = build_dataset(split, mode, max_rows=max_rows)
            result_df.to_csv(out_path, index=False)
            print(f"Guardado {len(result_df)} filas - {out_path}")

    print("\nPipeline completado")

#main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extractor de características de audio para NSynth."
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limita el número de audios procesados por split/modo (útil para pruebas). "
             "Omitir para procesar el dataset completo.",
    )
    parser.add_argument(
        "--mode",
        choices=["spectral", "mfcc", "cqt", "all"],
        default="all",
        help="Modo de extracción a ejecutar (default: all).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Split a procesar (default: all).",
    )
    args = parser.parse_args()

    if args.mode == "all" and args.split == "all":
        pipeline(max_rows=args.max_rows)
    else:
        os.makedirs("processed", exist_ok=True)
        modes  = ["spectral", "mfcc", "cqt"] if args.mode == "all" else [args.mode]
        splits = ["train", "val", "test"]    if args.split == "all" else [args.split]
        for mode in modes:
            for split in splits:
                print(f"\nExtrayendo [{mode}] — [{split}]")
                df = build_dataset(split, mode, max_rows=args.max_rows)
                out = f"processed/{mode}_{split}.csv"
                df.to_csv(out, index=False)
                print(f"Guardado {len(df)} filas - {out}")
        print("\nListo")
