# model/update_thresholds_meta.py
from pathlib import Path
import json

META_PATH = Path("model/model_meta.json")
BEST_SUMMARY = Path("model/BTC/analysis/best_threshold_summary.json")

# backup
backup = META_PATH.with_suffix(".backup.json")
backup.write_text(META_PATH.read_text(encoding="utf-8"), encoding="utf-8")

meta = json.loads(META_PATH.read_text(encoding="utf-8"))

if BEST_SUMMARY.exists():
    best = json.loads(BEST_SUMMARY.read_text(encoding="utf-8"))
    th = best["best_threshold"]
    summ = best.get("summary", {})
    # thresholds desde el análisis
    meta["policy_thr_up"] = float(th["th_up"])
    meta["policy_thr_down"] = float(th["th_dn"])
    # filtros prudentes
    meta["policy_min_ev"] = 0.02
    meta["policy_min_kelly"] = 0.05
    # trazabilidad
    meta["threshold_search"] = {
        "from": "best_threshold_summary.json",
        "production_thr_up": float(th["th_up"]),
        "production_thr_down": float(th["th_dn"]),
        "cv_bets_ref": float(summ.get("cv_bets", 0.0)),
        "cv_net_ref": float(summ.get("cv_net", 0.0)),
        "dd_max_ref": float(summ.get("dd_max", 0.0)),
        "ret_pct_ref": float(summ.get("ret_pct", 0.0)),
    }
else:
    # fallback a 0.53/0.53 si no existe el resumen
    meta["policy_thr_up"] = 0.53
    meta["policy_thr_down"] = 0.53
    meta["policy_min_ev"] = 0.02
    meta["policy_min_kelly"] = 0.05

META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("✔ meta actualizado desde best_threshold_summary.json (o fallback 0.53/0.53)")
print(f"🗂️ backup: {backup}")
