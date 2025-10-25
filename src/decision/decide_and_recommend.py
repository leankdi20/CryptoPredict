import pandas as pd
import joblib
from pathlib import Path
from datetime import timedelta
from model.utils import load_model_meta, apply_policy, _et_window_from_now, _fmt_hour



ROOT = Path(__file__).resolve().parents[2]
FEATS_CSV = ROOT / "data" / "derived" / "BTCUSDT" / "features_1h.csv"
MODEL_META_JSON = ROOT / "model" / "model_meta.json"
MODEL_FILE = ROOT / "model" / "models" / "best_model_xgb.joblib"
LOG_CSV = ROOT / "archivosextras" / "decisions_log.csv"

def main():
    feats = pd.read_csv(FEATS_CSV, parse_dates=["datetime"])

# Eliminar columnas no usadas por el modelo (target y OHLC crudos)
    cols_to_drop = ['target_up', 'target_ret', 'open', 'high', 'low', 'volume']
    feats = feats.drop(columns=[c for c in cols_to_drop if c in feats.columns])
    print("❓ Columns que tiene feats:")
    print(feats.columns.tolist())

    feats["datetime"] = feats["datetime"].dt.tz_convert("UTC")

    X = feats.drop(columns=["datetime", "close"], errors="ignore")
    y_close = feats["close"].copy()
    ts_candle = feats["datetime"].iloc[-1]  # vela que acaba de cerrar
    close = y_close.iloc[-1]

    model, calibrator = load_model_meta(MODEL_META_JSON, MODEL_FILE)
    proba = model.predict_proba(X.iloc[[-1]])[0][1]
    p_up = calibrator.predict_proba([[proba]])[0][1] if calibrator else proba
    p_down = 1 - p_up

    be_up, be_dn = 0.530, 0.470
    thr_up, thr_down = 0.527, 0.527
    min_ev, min_kelly = 0.020, 0.020
    src = "CAL"

    EV_up = (p_up - be_up) / be_up
    EV_dn = (p_down - be_dn) / be_dn
    k_up = (p_up - be_up) / (1 - be_up)
    k_dn = (p_down - be_dn) / (1 - be_dn)

    if p_up >= thr_up and EV_up >= min_ev and k_up >= min_kelly:
        decision = "BET_UP"
        p_used, k_raw = p_up, k_up
    elif p_down >= thr_down and EV_dn >= min_ev and k_dn >= min_kelly:
        decision = "BET_DOWN"
        p_used, k_raw = p_down, k_dn
    else:
        decision = "NO_BET"
        p_used, k_raw = None, None

    k_suggested = round(0.25 * k_raw, 3) if k_raw else None
    ts_entry = ts_candle + timedelta(hours=1)
    ts_settle = ts_entry + timedelta(hours=1)



    # --- Prints
    print(f"Prob SUBA ({src}): {p_up:.3f} | Prob BAJA: {p_down:.3f}")
    print("────────────────────────────────────────────────────────")
    print(f"🧱 Vela cerrada usada: {ts_candle} | Close: {close:,.2f} USDT")
    print(f"🪙 Apuesta:            {ts_entry} → liquida {ts_settle}")
    print(f"Break-even   → UP={be_up:.3f} | DOWN={be_dn:.3f}")
    print(f"Política     → UP≥{thr_up:.3f} | DOWN≥{thr_down:.3f} | min_EV≥{min_ev:.3f} | min_Kelly≥{min_kelly:.3f}")
    print(f"EV           → UP={EV_up:.4f} | DOWN={EV_dn:.4f}")
    print(f"👉 RECOMENDACIÓN: {decision}")
    if decision != "NO_BET":
        print(f"   • Prob usada: {p_used:.3f} | Kelly raw: {k_raw:.3f} | Kelly 25%: {k_suggested:.3f}")
    print(f"📝 Log: {LOG_CSV}")
    print("────────────────────────────────────────────────────────")
    start_et, end_et, end_utc, end_cr = _et_window_from_now()
    print("────────────────────────────────────────────────────────")
    print(f"🕒 Polymarket (ET): {_fmt_hour(start_et)}–{_fmt_hour(end_et)} ET | "
        f"cierra {end_et.strftime('%Y-%m-%d %H:%M:%S %Z')} "
        f"(UTC {end_utc.strftime('%H:%M')}, CR {end_cr.strftime('%H:%M')})")
    print("────────────────────────────────────────────────────────")

    # === Guardar logs ===
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    NO_BET_CSV = ROOT / "archivosextras" / "no_bet_log.csv"

    # Si es NO_BET → guardar en archivo especial y salir
    if decision == "NO_BET":
        no_bet_row = {
            "timestamp": ts_candle,
            "reason": "NO_BET",
            "p_up": round(p_up, 6),
            "p_down": round(p_down, 6),
            "src": src
        }
        # --- Evitar duplicados ---
        if NO_BET_CSV.exists():
            df_nb = pd.read_csv(NO_BET_CSV)
            # Convertir timestamp a string para comparar sin errores de tipo
            if str(ts_candle) in df_nb["timestamp"].astype(str).values:
                print(f"⚠️ NO_BET ya registrado para {ts_candle}, se omite guardado.")
            else:
                pd.DataFrame([no_bet_row]).to_csv(NO_BET_CSV, mode="a", header=False, index=False)
                print(f"✅ NO_BET registrado en {NO_BET_CSV}")
        else:
            pd.DataFrame([no_bet_row]).to_csv(NO_BET_CSV, index=False)
            print(f"✅ NO_BET registrado en {NO_BET_CSV} (nuevo archivo)")

        return  # sale después de guardar el no_bet


    # === Si hay apuesta (BET_UP o BET_DOWN) ===
    row = {
        "ts_entry": ts_entry,
        "ts_settle": ts_settle,
        "signal": decision.replace("BET_", ""),
        "price_entry": close,
        "model": src,
        "p_up": round(p_up, 3),
        "p_down": round(p_down, 3),
        "EV_up": round(EV_up, 4),
        "EV_down": round(EV_dn, 4),
        "kelly_raw": round(k_raw, 3),
        "kelly_25pct": k_suggested,
        "src": src,
        "close": close,
    }

    try:
        df_log = pd.read_csv(LOG_CSV, parse_dates=["ts_entry", "ts_settle"])
    except FileNotFoundError:
        df_log = pd.DataFrame()

    exists = (
        not df_log.empty and
        ((df_log["ts_entry"] == ts_entry) & (df_log["ts_settle"] == ts_settle)).any()
    )

    if exists:
        print(f"⚠️ Ya existe una apuesta para {ts_entry} → {ts_settle}, no se duplicará.")
    else:
        df_log = pd.concat([df_log, pd.DataFrame([row])], ignore_index=True)
        df_log.sort_values("ts_entry", inplace=True)
        df_log.to_csv(LOG_CSV, index=False)
        print(f"✅ Apuesta guardada en log ({ts_entry} → {ts_settle}).")


if __name__ == "__main__":
    main()
