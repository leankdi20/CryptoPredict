# model/BTC/analyse/find_bad_periods.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ========= Config =========
PREDS_DIR   = Path("model/BTC/preds")
OUT_DIR     = Path("model/BTC/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Política/Simulación (podés ajustar)
BANK0         = 100.0
STAKE_DYNAMIC = False
STAKE_PCT     = 0.02     # usado solo si STAKE_DYNAMIC=True
STAKE_FIX     = 1.0
PROFIT_F      = 0.885
FEE_M         = 0.0

# Umbrales a usar (si existe best_threshold_summary.json, se usan esos)
BEST_JSON     = OUT_DIR / "best_threshold_summary.json"
TH_UP         = 0.53
TH_DN         = 0.53

# Reglas para sugerir bloqueos
MIN_BETS_PER_BUCKET = 200     # mínimo de trades en bucket para evaluarlo
PCT_HIT_FLOOR       = 0.50    # si el hit rate < 50% y pnl<0 ⇒ candidato a bloqueo

def _read_preds():
    files = sorted(PREDS_DIR.glob("pred_iter*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay pred_iter*.csv en {PREDS_DIR}")
    dfs=[]
    for f in files:
        df = pd.read_csv(f)
        rn = {c:c.strip().lower() for c in df.columns}
        df.rename(columns=rn, inplace=True)
        for c in ("timestamp","prob_up","prob_down","actual"):
            if c not in df.columns: raise ValueError(f"{f} no tiene {c}")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["prob_up"]   = df["prob_up"].astype(float)
        df["prob_down"] = df["prob_down"].astype(float)
        df["actual"]    = df["actual"].astype(int)       # {-1, +1}
        dfs.append(df[["timestamp","prob_up","prob_down","actual"]])
    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp")
    df["weekday"] = df["timestamp"].dt.weekday       # 0=Lunes .. 6=Domingo
    df["hour_utc"]= df["timestamp"].dt.hour
    df["month"]   = df["timestamp"].dt.tz_convert(None).dt.to_period("M")
    return df

def _apply_policy(df, th_up, th_dn):
    sig = np.where(df["prob_up"]>=th_up, 1,
          np.where(df["prob_down"]>=th_dn, -1, 0)).astype(int)
    df2 = df[sig!=0].copy()
    df2["signal"] = sig[sig!=0]
    # PnL por trade con stake simple (fijo o %)
    bank = BANK0
    pnls=[]
    for _ts, row in df2.iterrows():
        stake = (bank*STAKE_PCT) if STAKE_DYNAMIC else STAKE_FIX
        stake = float(max(0.0, min(bank, stake)))
        win = 1 if row["signal"]==row["actual"] else 0
        pnl = stake*PROFIT_F if win else -stake
        bank = max(0.0, bank + pnl)
        pnls.append(pnl)
    df2["pnl"]=pnls
    return df2

def _bucket_summary(df2, by_cols, name):
    g = df2.groupby(by_cols, sort=True)
    out = pd.DataFrame({
        "bets": g.size(),
        "wins": g.apply(lambda x: int((x["signal"].values==x["actual"].values).sum())),
        "pnl":  g["pnl"].sum()
    })
    out["hit_rate"] = out["wins"]/out["bets"].clip(lower=1)
    out = out.sort_index()
    out.to_csv(OUT_DIR / f"perf_by_{name}.csv")
    return out

def _suggest_blocks(by_weekday, by_hour):
    # Weekdays a bloquear (id 0..6)
    block_wd = by_weekday.query("bets>=@MIN_BETS_PER_BUCKET and pnl<0 and hit_rate<@PCT_HIT_FLOOR").index.tolist()
    # Horas UTC a bloquear (0..23)
    block_hr = by_hour.query("bets>=@MIN_BETS_PER_BUCKET and pnl<0 and hit_rate<@PCT_HIT_FLOOR").index.tolist()
    return block_wd, block_hr

def main():
    # thresholds
    if BEST_JSON.exists():
        bj = json.loads(BEST_JSON.read_text(encoding="utf-8"))
        th = bj.get("best_threshold", {})
        if "th_up" in th:  globals()["TH_UP"] = float(th["th_up"])
        if "th_dn" in th:  globals()["TH_DN"] = float(th["th_dn"])

    df  = _read_preds()
    df2 = _apply_policy(df, TH_UP, TH_DN)

    # Tablas
    by_month   = _bucket_summary(df2, ["month"], "month")
    by_weekday = _bucket_summary(df2, ["weekday"], "weekday")
    by_hour    = _bucket_summary(df2, ["hour_utc"], "hour")

    # Sugerencias de bloqueos
    block_wd, block_hr = _suggest_blocks(by_weekday, by_hour)
    suggestion = {
        "threshold_used": {"th_up": TH_UP, "th_dn": TH_DN},
        "min_bets_per_bucket": MIN_BETS_PER_BUCKET,
        "hit_floor": PCT_HIT_FLOOR,
        "suggest_block_weekdays": block_wd,
        "suggest_block_hours_utc": block_hr
    }
    (OUT_DIR/"bad_periods_suggestion.json").write_text(json.dumps(suggestion, indent=2), encoding="utf-8")

    print("✅ Guardado:")
    print("  • perf_by_month.csv")
    print("  • perf_by_weekday.csv (0=Lun .. 6=Dom)")
    print("  • perf_by_hour.csv (UTC)")
    print("  • bad_periods_suggestion.json (bloqueos sugeridos)")

if __name__ == "__main__":
    main()
