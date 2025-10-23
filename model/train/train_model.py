# model/train/train_model.py
# Entrenamiento 1H (walk-forward semanal) + calibración y export de predicciones de TEST por iteración

import json, math, numpy as np, pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
import joblib

# ============================ Paths ============================
BASE        = Path(".")
DERIVED     = BASE/"data/derived/BTCUSDT/features_1h.csv"  # (referencia/opcional)
PREP_X      = BASE/"data/preprocessed/BTCUSDT/X_1h.csv"
PREP_Y      = BASE/"data/preprocessed/BTCUSDT/y_color_1h.csv"
OUT_MODELS  = BASE/"model/BTC/models"
OUT_PREDS   = BASE/"model/BTC/preds"              # carpeta de predicciones por iter
OUT_BEST    = BASE/"model/models/best_model_xgb.joblib"
META_PATH   = BASE/"model/model_meta.json"

# ======================== Configuración ========================
# Modo de entrenamiento:
# - "FIXED_WINDOWS": usa TRAIN_WINDOWS + TEST_WINDOW
# - "WEEKLY_WF": walk-forward semanal (train→calib→test) 2022→hoy
MODE = "WEEKLY_WF"

# Ventanas fijas (si usas FIXED_WINDOWS)
TRAIN_WINDOWS = [
    ("2022-01-01", "2022-07-01"),
    ("2022-07-01", "2023-01-01"),
    ("2023-01-01", "2023-07-01"),
    ("2023-07-01", "2024-01-01"),
    ("2024-01-01", "2024-07-01"),
    ("2024-07-01", "2025-09-01"),
]
TEST_WINDOW = ("2025-01-01", "2026-01-01")

# Ventanas dinámicas (si usas WEEKLY_WF)
TEST_DAYS         = 7            # bloque de TEST = 1 semana (168 velas)
CAL_DAYS          = 30           # bloque de calibración previo al TEST
HALF_LIFE_DAYS    = 15           # pesos exponenciales (más peso a lo reciente)
MAX_TRAIN_MONTHS  = 9           # histórico máximo usado para entrenar (rolling)
N_WEEKS = 8  
# Modelo base XGB
XGB_PARAMS = dict(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0,
    objective="binary:logistic", random_state=42, n_jobs=4
)

# ===================== Utilidades de tiempo ====================
def _to_ts(x):
    return pd.Timestamp(x, tz="UTC")

def _mask_between(dts: pd.Series, a, b):
    return (dts >= _to_ts(a)) & (dts <= _to_ts(b))

def _half_life_weights(dts: pd.Series, half_life_days=30):
    """w = exp(-(Tmax - t)/tau) con tau = half_life/ln(2)"""
    tau = half_life_days / math.log(2)
    tmax = dts.max()
    age_days = (tmax - dts).dt.total_seconds() / 86400.0
    return np.exp(-age_days / tau).astype(float)

def _weekly_cutoffs(dts: pd.Series, start="2022-01-01"):
    """Fechas domingo (W-SUN) dentro del rango, dejando ~1 mes inicial para no arrancar ‘vacío’."""
    start = max(_to_ts(start), dts.min())
    end   = dts.max()
    weeks = pd.date_range(start=start.normalize(), end=end.normalize(), freq="W-SUN", tz="UTC")
    return [w for w in weeks if w >= start + pd.Timedelta(days=30)]

# ============================ Carga ============================
def _load_feat_cols():
    meta = json.loads(Path(META_PATH).read_text(encoding="utf-8"))
    return meta["feat_cols"], meta

def _load_Xy():
    # X: features alineadas 1:1 con y_color_1h.csv
    X = pd.read_csv(PREP_X)
    y_df = pd.read_csv(PREP_Y, parse_dates=["timestamp"])
    y = y_df["label_color"].astype(int).values        # {-1, 1}
    dts = (y_df["timestamp"].dt.tz_convert("UTC")
           if y_df["timestamp"].dt.tz is not None
           else y_df["timestamp"].dt.tz_localize("UTC"))
    assert len(X) == len(y) == len(dts), "X, y y timestamps deben alinear 1:1"
    return X, y, dts

# ========================= Entrenadores ========================
def train_fixed_windows(X, y, dts, feat_cols, meta):
    """Entrena por bloques fijos y elige el mejor por logloss en VALID."""
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_PREDS.mkdir(parents=True, exist_ok=True)

    best, best_ll = None, 1e9

    # TEST fijo, VALID = 20% final del TRAIN de cada bloque
    test_mask = _mask_between(dts, TEST_WINDOW[0], TEST_WINDOW[1])
    X_te, y_te, d_te = X[test_mask], y[test_mask], dts[test_mask]

    iter_idx = 0
    for (a, b) in TRAIN_WINDOWS:
        tr_mask = _mask_between(dts, a, b)
        X_tr_all, y_tr_all, d_tr_all = X[tr_mask], y[tr_mask], dts[tr_mask]
        if len(X_tr_all) < 500:
            continue

        cut = int(len(X_tr_all)*0.8)
        X_tr, y_tr, d_tr = X_tr_all.iloc[:cut], y_tr_all[:cut], d_tr_all.iloc[:cut]
        X_va, y_va, d_va = X_tr_all.iloc[cut:], y_tr_all[cut:], d_tr_all.iloc[cut:]

        w_tr = _half_life_weights(d_tr, HALF_LIFE_DAYS)
        w_va = _half_life_weights(d_va, HALF_LIFE_DAYS)

        # balanceo
        pos = int(np.sum(y_tr == 1)); neg = int(np.sum(y_tr == -1))
        spw = (neg / max(pos, 1)) if pos > 0 else 1.0

        base = XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": spw})
        base.fit(X_tr, (y_tr==1).astype(int), sample_weight=w_tr, verbose=False)
        cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
        cal.fit(X_va, (y_va==1).astype(int), sample_weight=w_va)

        # métrica y snapshot
        p_va = cal.predict_proba(X_va)[:,1]
        ll_va = log_loss((y_va==1).astype(int), p_va, labels=[0,1])

        p_te = cal.predict_proba(X_te)[:,1]
        acc_te = accuracy_score((y_te==1).astype(int), (p_te>=0.5).astype(int))

        print(f"[FIX] train {a}..{b} | val ll={ll_va:.4f} | test acc={acc_te:.3f}")
        snap = OUT_MODELS/f"xgb_fix_{a}_{b}_ll{ll_va:.4f}.joblib"
        joblib.dump(cal, snap)

        # thresholds de política desde meta (fallback 0.5)
        thr_up   = float(meta.get("policy_thr_up",   0.5))
        thr_down = float(meta.get("policy_thr_down", 0.5))

        # export de TEST de esta iter en el formato requerido
        prob_up   = p_te
        prob_down = 1.0 - prob_up
        pred_cls  = np.where(prob_up >= 0.5, 1, -1).astype(int)
        signal    = np.where(prob_up >= thr_up, 1, np.where(prob_down >= thr_down, -1, 0)).astype(int)

        df_pred = pd.DataFrame({
            "timestamp": d_te.astype("datetime64[ns, UTC]").astype(str),
            "prob_down": prob_down.astype(float),
            "prob_up":   prob_up.astype(float),
            "signal":    signal,
            "pred_cls":  pred_cls,
            "actual":    np.where(y_te==1, 1, -1).astype(int)
        })
        df_pred.to_csv(OUT_PREDS/f"pred_iter_fix_{iter_idx:02d}.csv", index=False)

        iter_idx += 1
        if ll_va < best_ll:
            best_ll, best = ll_va, cal

    if best is None:
        raise RuntimeError("No hubo ventana fija válida.")
    _persist_best(best)
    return best

def train_weekly_walkforward(X, y, dts, feat_cols, meta):
    """
    Walk-forward semanal:
      TEST  = semana que termina en week_end  (7d → 168 velas)
      CAL   = últimos CAL_DAYS previos a TEST (p/ calibrar probas)
      TRAIN = hasta CAL-1h, limitado por MAX_TRAIN_MONTHS
    Exporta por iteración: model/BTC/preds/pred_iterNN.csv
    """
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_PREDS.mkdir(parents=True, exist_ok=True)

    best, best_ll = None, 1e9
    cutoffs = _weekly_cutoffs(dts, start="2022-01-01")

    N_WEEKS = 8  # o 12 si querés más historia
    if len(cutoffs) > N_WEEKS:
        cutoffs = cutoffs[-N_WEEKS:]

    print(f"📆 Usando las últimas {len(cutoffs)} semanas (hasta {cutoffs[-1].date()})")

    # thresholds de política desde meta (fallback 0.5)
    thr_up   = float(meta.get("policy_thr_up",   0.6))
    thr_down = float(meta.get("policy_thr_down", 0.6))

    iter_idx = 0
    for week_end in cutoffs:
        # TEST = semana que termina en week_end
        test_start = week_end - pd.Timedelta(days=TEST_DAYS) + pd.Timedelta(hours=1)
        test_mask = (dts >= test_start) & (dts <= week_end)
        if test_mask.sum() < 24:
            continue

        # CAL = CAL_DAYS previos al inicio de TEST
        cal_end = test_start - pd.Timedelta(hours=1)
        cal_start = cal_end - pd.Timedelta(days=CAL_DAYS) + pd.Timedelta(hours=1)
        cal_mask = (dts >= cal_start) & (dts <= cal_end)
        if cal_mask.sum() < 24:
            continue

        # TRAIN = hasta cal_start-1h (máx MAX_TRAIN_MONTHS)
        tr_end = cal_start - pd.Timedelta(hours=1)
        tr_start = tr_end - pd.DateOffset(months=MAX_TRAIN_MONTHS)
        tr_mask = (dts >= tr_start) & (dts <= tr_end)
        if tr_mask.sum() < 200:
            continue

        # slices
        X_tr, y_tr, d_tr = X[tr_mask], y[tr_mask], dts[tr_mask]
        X_cal, y_cal, d_cal = X[cal_mask], y[cal_mask], dts[cal_mask]
        X_te,  y_te,  d_te  = X[test_mask], y[test_mask], dts[test_mask]

        # pesos half-life
        w_tr  = _half_life_weights(d_tr, HALF_LIFE_DAYS)
        w_cal = _half_life_weights(d_cal, HALF_LIFE_DAYS)

        # balanceo en TRAIN
        pos = int(np.sum(y_tr == 1)); neg = int(np.sum(y_tr == -1))
        spw = (neg / max(pos, 1)) if pos > 0 else 1.0

        # entrenar base
        base = XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": spw})
        base.fit(X_tr, (y_tr==1).astype(int), sample_weight=w_tr, verbose=False)

        # calibración sobre CAL (sigmoid estable)
        cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
        cal.fit(X_cal, (y_cal==1).astype(int), sample_weight=w_cal)

        # métricas
        p_cal = cal.predict_proba(X_cal)[:,1]
        p_te  = cal.predict_proba(X_te )[:,1]
        ll_cal   = log_loss((y_cal==1).astype(int), p_cal, labels=[0,1])
        acc_cal  = accuracy_score((y_cal==1).astype(int), (p_cal>=0.5).astype(int))
        acc_test = accuracy_score((y_te ==1).astype(int), (p_te >=0.5).astype(int))

        print(
            f"[WF] train {d_tr.min().date()}..{d_tr.max().date()} | "
            f"cal {d_cal.min().date()}..{d_cal.max().date()} | "
            f"test {d_te.min().date()}..{d_te.max().date()} | "
            f"cal ll={ll_cal:.4f} acc={acc_cal:.3f} | test acc={acc_test:.3f}"
        )

        # —— export CSV de la semana de TEST (formato requerido) ——
        prob_up   = p_te
        prob_down = 1.0 - prob_up

        # Clase pura del modelo (corte 0.5): 1=UP, -1=DOWN
        pred_cls = np.where(prob_up >= 0.5, 1, -1).astype(int)

        # Señal de trading con política: +1 si prob_up≥thr_up, -1 si prob_down≥thr_down, 0 si no apuesta
        signal = np.where(
            prob_up >= thr_up, 1,
            np.where(prob_down >= thr_down, -1, 0)
        ).astype(int)

        df_pred = pd.DataFrame({
            "timestamp": d_te.astype("datetime64[ns, UTC]").astype(str),
            "prob_down": prob_down.astype(float),
            "prob_up":   prob_up.astype(float),
            "signal":    signal,
            "pred_cls":  pred_cls,
            "actual":    y_te.astype(int)
        })
        out_csv = OUT_PREDS / f"pred_iter{iter_idx:02d}.csv"
        df_pred.to_csv(out_csv, index=False)

        # (opcional) reporte corto de TEST
        y_true_bin = (y_te==1).astype(int)
        y_hat_bin  = (prob_up>=0.5).astype(int)
        rep = classification_report(y_true_bin, y_hat_bin,
                                    target_names=["down(-1)", "up(+1)"], digits=3)
        (OUT_PREDS / f"classif_report_iter{iter_idx:02d}.txt").write_text(rep, encoding="utf-8")
        print(f"Predicciones TEST guardadas → {out_csv}\n{rep}\n")

        # snapshot de modelo de esta iteración
        snap = OUT_MODELS / f"xgb_wf_{d_tr.max().date()}_ll{ll_cal:.4f}.joblib"
        joblib.dump(cal, snap)

        # ‘mejor’ por logloss en CAL
        if ll_cal < best_ll:
            best_ll, best = ll_cal, cal

        iter_idx += 1

    if best is None:
        raise RuntimeError("No se pudo entrenar ningún fold semanal válido.")
    _persist_best(best)
    return best

# ===================== Persistencia / Meta =====================
def _persist_best(best):
    OUT_BEST.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, OUT_BEST)
    print(f"✅ Modelo ‘best’ guardado en: {OUT_BEST}")

def _update_meta(extra: dict):
    meta_path = Path(META_PATH)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update(extra)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"📝 Meta actualizada: {META_PATH}")

# ============================== Main ==========================
def main():
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_PREDS.mkdir(parents=True, exist_ok=True)

    feat_cols, meta = _load_feat_cols()
    X, y, dts = _load_Xy()

    # sanity de columnas
    for c in feat_cols:
        if c not in X.columns:
            raise ValueError(f"Falta feature en X: {c}")

    if MODE == "FIXED_WINDOWS":
        model = train_fixed_windows(X, y, dts, feat_cols, meta)
        meta_trainer = dict(
            mode=MODE, half_life_days=HALF_LIFE_DAYS,
            xgb_params=XGB_PARAMS, train_windows=TRAIN_WINDOWS,
            test_window=TEST_WINDOW
        )
    elif MODE == "WEEKLY_WF":
        model = train_weekly_walkforward(X, y, dts, feat_cols, meta)
        meta_trainer = dict(
            mode=MODE, half_life_days=HALF_LIFE_DAYS,
            cal_days=CAL_DAYS, test_days=TEST_DAYS,
            max_train_months=MAX_TRAIN_MONTHS,
            xgb_params=XGB_PARAMS
        )
    else:
        raise ValueError("MODE debe ser 'FIXED_WINDOWS' o 'WEEKLY_WF'.")

    _update_meta({"trainer": meta_trainer})

if __name__ == "__main__":
    main()
