# src/preprocessing/make_Xy_1h.py
import json, numpy as np, pandas as pd
from pathlib import Path

DERIVED = Path("data/derived/BTCUSDT/features_1h.csv")
PREP    = Path("data/preprocessed/BTCUSDT")
META    = Path("model/model_meta.json")
DT_COL  = "datetime"

def load_feat_cols():
    meta = json.loads(Path(META).read_text(encoding="utf-8"))
    return meta["feat_cols"]

def get_labels(feats: pd.DataFrame) -> np.ndarray:
    # 1) target_color ∈ {-1,1}
    if "target_color" in feats.columns:
        y = feats["target_color"].astype(int).values
        y = np.where(y == 1, 1, -1)
        return y
    # 2) target_up ∈ {0,1} → {-1,1}
    if "target_up" in feats.columns:
        y = feats["target_up"].astype(int).values
        return np.where(y == 1, 1, -1)
    # 3) fallback: construir label desde close.shift(-1)
    if "close" in feats.columns:
        nxt = feats["close"].shift(-1)
        y = (nxt > feats["close"]).astype(int).values
        return np.where(y == 1, 1, -1)
    raise ValueError("No encuentro columnas de target (target_color/target_up) ni 'close' para derivarlo.")

def main():
    PREP.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(DERIVED, parse_dates=[DT_COL])
    feats[DT_COL] = pd.to_datetime(feats[DT_COL], utc=True)

    feat_cols = load_feat_cols()
    missing = [c for c in feat_cols if c not in feats.columns]
    if missing:
        raise ValueError(f"Faltan columnas en features: {missing}")

    # X e Y
    X = feats[feat_cols].copy()
    y = get_labels(feats)

    # guardar
    X.to_csv(PREP/"X_1h.csv", index=False)
    pd.DataFrame({"timestamp": feats[DT_COL], "label_color": y}).to_csv(PREP/"y_color_1h.csv", index=False)
    print("✅ Guardados: preprocessed/BTCUSDT/X_1h.csv y y_color_1h.csv")

if __name__ == "__main__":
    main()
