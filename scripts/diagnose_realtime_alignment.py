# scripts/diagnose_realtime_alignment.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
FEATS = ROOT / "data" / "derived" / "BTCUSDT" / "features_1h.csv"
META  = ROOT / "model" / "model_meta.json"
PKL   = ROOT / "model" / "models" / "best_model_xgb.joblib"

K = 48  # horas recientes a chequear

def main():
    # --- meta seguro (json.load)
    with open(META, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feat_cols = meta["feat_cols"]
    thr_up    = float(meta.get("policy_thr_up",   0.53))
    thr_dn    = float(meta.get("policy_thr_down", 0.53))

    # --- features
    feats = pd.read_csv(FEATS, parse_dates=["datetime"])
    if feats["datetime"].dt.tz is None:
        feats["datetime"] = feats["datetime"].dt.tz_localize("UTC")
    feats = feats.sort_values("datetime").reset_index(drop=True)

    # target: movimiento de la SIGUIENTE hora
    feats["close_next"] = feats["close"].shift(-1)
    feats["actual_next_hour"] = (feats["close_next"] > feats["close"]).astype("Int64")

    # evitamos la última (no tiene next)
    eval_df = feats.iloc[-(K+1):-1].copy()

    # --- modelo y probabilidades calibradas
    model = joblib.load(PKL)
    X = eval_df[feat_cols].astype(float)
    p_up = model.predict_proba(X)[:, 1]
    eval_df["p_up"] = p_up
    eval_df["p_down"] = 1 - p_up

    # señal con thresholds actuales (sin EV/Kelly para simplificar diagnóstico)
    eval_df["signal"] = np.where(
        eval_df["p_up"] >= thr_up, 1,
        np.where(eval_df["p_down"] >= thr_dn, -1, 0)
    )
    eval_df["actual"] = eval_df["actual_next_hour"].astype(int)
    eval_df["win"] = ((eval_df["signal"] == 1) & (eval_df["actual"] == 1)) | \
                     ((eval_df["signal"] == -1) & (eval_df["actual"] == 0))

    # métricas
    covered = float((eval_df["signal"] != 0).mean())
    hit = float(eval_df.loc[eval_df["signal"] != 0, "win"].mean()) if (eval_df["signal"] != 0).any() else float("nan")
    mix = eval_df.loc[eval_df["signal"] != 0, "signal"].value_counts()

    print("\n=== Diagnóstico últimas", K, "horas ===")
    print("Ventana:", eval_df["datetime"].iloc[0], "→", eval_df["datetime"].iloc[-1])
    print(f"p_up  (min/med/max): {eval_df['p_up'].min():.3f} / {eval_df['p_up'].median():.3f} / {eval_df['p_up'].max():.3f}")
    print(f"Cobertura (señales != 0): {covered:.2%}")
    print("Mix señales (1=UP, -1=DOWN):")
    if not mix.empty:
        print(mix.to_string())
    else:
        print("(sin señales con los thresholds actuales)")
    print(f"Hit-rate sobre señales: {hit:.2%}" if hit == hit else "Hit-rate sobre señales: n/a")

    # chequeo de alineación temporal (t -> t+1)
    tail = eval_df.tail(5)[["datetime","close","close_next","p_up","p_down","signal","actual"]]
    print("\n--- Últimos 5 registros (t -> t+1) ---")
    print(tail.to_string(index=False))

if __name__ == "__main__":
    main()
