from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
META_PATH   = ROOT / "model" / "model_meta.json"
FEATS_CSV   = ROOT / "data" / "derived" / "BTCUSDT" / "features_1h.csv"
DEFAULT_PKL = ROOT / "model" / "models" / "best_model_xgb.joblib"

def resolve_model_path() -> Path:
    candidates = []
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        p = meta.get("model_pkl")
        if p:
            p = Path(p)
            candidates += [p, ROOT / p, ROOT / "model" / "models" / p.name]
    candidates.append(DEFAULT_PKL)
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]

def pick_safe_last_row(feats: pd.DataFrame, feat_cols, lookback: int = 5) -> pd.DataFrame:
    tail = feats.tail(lookback)
    for idx in reversed(tail.index.tolist()):
        row = tail.loc[[idx], feat_cols]
        if not row.isna().any().any() and not np.isinf(row.values).any():
            return row
    return feats.tail(1)[feat_cols]

def get_base_estimator(calibrated):
    """Devuelve el XGB base dentro del CalibratedClassifierCV (robusto a versiones)."""
    # scikit 1.2/1.3
    if hasattr(calibrated, "base_estimator"):
        return calibrated.base_estimator
    # scikit 1.4+
    if hasattr(calibrated, "estimator"):
        return calibrated.estimator
    # fallback al primer calibrator
    if hasattr(calibrated, "calibrated_classifiers_"):
        cc = calibrated.calibrated_classifiers_[0]
        for attr in ("classifier", "estimator", "base_estimator"):
            if hasattr(cc, attr):
                return getattr(cc, attr)
    return None

def main():
    print("📂 ROOT:", ROOT)
    print("📄 META:", META_PATH)
    print("📄 FEATS:", FEATS_CSV)

    model_path = resolve_model_path()
    print("🔎 Modelo esperado en:", model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"No encuentro el modelo en: {model_path}")

    if not FEATS_CSV.exists():
        raise FileNotFoundError(f"No encuentro features en {FEATS_CSV}. Corré: python -m src.feature.make_features_1h")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    feat_cols = meta.get("feat_cols")
    if not feat_cols:
        raise KeyError("Falta 'feat_cols' en model/model_meta.json.")

    model = joblib.load(model_path)
    feats = pd.read_csv(FEATS_CSV, parse_dates=["datetime"])

    missing = [c for c in feat_cols if c not in feats.columns]
    if missing:
        raise ValueError(f"Faltan columnas en features: {missing}")

    # Seleccionar última fila sana
    X_last = pick_safe_last_row(feats, feat_cols, lookback=5)
    idx = X_last.index[0]
    ts = feats.loc[idx, "datetime"]

    # --- DEBUG: proba base (sin calibración) vs calibrada ---
    base = get_base_estimator(model)
    p_up_base = None
    if base is not None and hasattr(base, "predict_proba"):
        try:
            p_up_base = float(base.predict_proba(X_last)[:, 1][0])
        except Exception:
            p_up_base = None

    # Proba calibrada
    p_up_cal = float(model.predict_proba(X_last)[:, 1][0])

    # Selección robusta: si la calibrada está saturada, usar la base
    eps = 1e-3
    if (p_up_cal <= 0.01 or p_up_cal >= 0.99) and (p_up_base is not None):
        p_up = float(np.clip(p_up_base, eps, 1 - eps))
        src = "BASE"
    else:
        p_up = float(np.clip(p_up_cal,  eps, 1 - eps))
        src = "CAL"

    p_dn = 1.0 - p_up

    print("⏱️ Última vela:", ts)
    if p_up_base is not None:
        print(f"Score base (sin calib): {p_up_base:.3f} | Calibrada: {p_up_cal:.3f} | Usada({src}): {p_up:.3f}")
    else:
        print(f"Prob SUBA (clip): {p_up:.3f} | Prob BAJA: {p_dn:.3f}")

    print("✅ Inferencia OK")

    # --- Debug de NaN/Inf fila usada ---
    row_dbg = feats.loc[[idx], feat_cols]
    print("NaNs en fila usada:", int(row_dbg.isna().sum().sum()), "| Infs:", int(np.isinf(row_dbg.values).sum()))

    # --- Outliers de features (z-score simple en cola) ---
    tailN = min(500, len(feats))
    tail = feats.tail(tailN)[feat_cols]
    mu, sd = tail.mean(), tail.std().replace(0, np.nan)
    z = ((X_last.iloc[0] - mu) / sd).abs().sort_values(ascending=False)
    print("Top outliers (|z|) de la última fila:")
    print(z.head(8).round(2).to_string())

    # Distribución de probas recientes (calibradas)
    X_tail = feats.tail(200)[feat_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(X_tail) >= 10:
        probas_200 = model.predict_proba(X_tail)[:, 1]
        print(f"Últimas {len(X_tail)} → min:{float(probas_200.min()):.3f} "
              f"max:{float(probas_200.max()):.3f} mean:{float(probas_200.mean()):.3f}")
    else:
        print("Pocas filas sanas para estadísticas recientes.")

   

if __name__ == "__main__":
    main()
