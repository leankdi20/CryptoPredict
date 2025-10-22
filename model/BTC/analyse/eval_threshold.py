# model/BTC/analyse/eval_threshold.py
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# ——— silenciar solo este warning específico de pandas ———
warnings.filterwarnings(
    "ignore",
    message="Converting to PeriodArray/Index representation will drop timezone information.",
    category=UserWarning,
)

# ========================== CONFIG ==========================
DATA_DIR      = Path("model/BTC/preds")
OUT_DIR       = Path("model/BTC/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANK0         = 250.0
STAKE_DYNAMIC = False     # Monte Carlo mensual asume fijo para exactitud
STAKE_PCT     = 0.02
STAKE_FIX     = 5.0

FEE_M         = 0.0
PROFIT_F      = 0.885

# Sweep (simétrico)
TH_GRID       = np.arange(0.512, 0.531, 0.001)  # 0.512 .. 0.530 inclusive

# Asimétrico (opcional)
ASYMMETRIC    = False
TH_GRID_UP    = np.arange(0.50, 0.531, 0.005)
TH_GRID_DN    = np.arange(0.50, 0.531, 0.005)

# Ventana temporal (opcional)
DATE_START    = pd.Timestamp("2020-01-01", tz="UTC")
DATE_END      = pd.Timestamp("2100-01-01 23:59:59", tz="UTC")

# Score “fair”
ALPHA_BETS    = 0.050
BETA_NET      = 0.250

# Monte Carlo
MC_N_SIMS     = 10000       # nº de simulaciones
RANDOM_SEED   = 1337        # semilla reproducible

# ======================== HELPERS ===========================
def _read_all_preds(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("pred_iter*.csv"))
    if not files:
        raise FileNotFoundError(f"No encontré CSVs en {data_dir}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        rn = {c: c.strip().lower() for c in df.columns}
        df.rename(columns=rn, inplace=True)
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["timestamp"] = ts
        for c in ("prob_up", "prob_down", "actual"):
            if c not in df.columns:
                raise ValueError(f"{f} no tiene columna '{c}'")
        df["prob_up"] = df["prob_up"].astype(float)
        df["prob_down"] = df["prob_down"].astype(float)
        df["actual"] = df["actual"].astype(int)
        dfs.append(df[["timestamp", "prob_up", "prob_down", "actual"]])
    out = pd.concat(dfs, ignore_index=True).sort_values("timestamp")
    out = out[(out["timestamp"] >= DATE_START) & (out["timestamp"] <= DATE_END)]
    out.reset_index(drop=True, inplace=True)
    return out

def _to_month_period(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts, utc=True)
    if isinstance(s, pd.DatetimeIndex):
        return s.tz_convert(None).to_period("M")
    else:
        return s.dt.tz_convert(None).dt.to_period("M")

def _apply_monthly_fee(equity: pd.Series, fee_m: float) -> pd.Series:
    if fee_m == 0:
        return equity
    idx = equity.index
    if isinstance(idx, pd.DatetimeIndex):
        months = idx.tz_convert(None).to_period("M")
    else:
        months = _to_month_period(idx)
    eq = equity.copy()
    changes = months != months.shift(1)
    changes.iloc[0] = False
    eq[changes] = (eq[changes] - fee_m).clip(lower=0.0)
    return eq

def _max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    dd = series - roll_max
    return float(dd.min())

def _monthly_cv_count(ts: pd.Series) -> float:
    if ts.empty:
        return 0.0
    counts = _to_month_period(ts).value_counts().sort_index()
    m = counts.mean()
    if m == 0 or np.isnan(m):
        return 0.0
    return float(counts.std(ddof=0) / m)

def _monthly_cv_pnl(trade_ts: pd.Series, pnl: pd.Series) -> float:
    if trade_ts.empty:
        return 0.0
    df = pd.DataFrame({"ts": trade_ts.values, "pnl": pnl.values})
    df["month"] = _to_month_period(pd.to_datetime(df["ts"], utc=True))
    agg = df.groupby("month", sort=True)["pnl"].sum()
    mean_abs = np.abs(agg.mean())
    if mean_abs == 0 or np.isnan(mean_abs):
        return 0.0
    return float(agg.std(ddof=0) / (mean_abs + 1e-9))

def _simulate(df: pd.DataFrame, th_up: float, th_dn: float):
    """Simula trades y equity para un threshold dado. Devuelve dict + equity + df_tr."""
    sig = np.where(df["prob_up"] >= th_up, 1,
          np.where(df["prob_down"] >= th_dn, -1, 0)).astype(int)
    mask = sig != 0
    df_tr = df.loc[mask].copy()
    df_tr["signal"] = sig[mask]

    if df_tr.empty:
        empty_equity = pd.Series([BANK0], index=pd.to_datetime([df["timestamp"].iloc[0]], utc=True))
        return dict(th_up=th_up, th_dn=th_dn, bets=0, wins=0, hit_rate=np.nan,
                    coverage=0.0, pnl_net=0.0, bank_final=BANK0, ret_pct=0.0,
                    dd_max=0.0, cv_bets=0.0, cv_net=0.0, score=-np.inf), empty_equity, df_tr

    bank = BANK0
    trade_pnls, equity_idx, equity_vals = [], [], []

    for _, row in df_tr.iterrows():
        base_stake = (bank * STAKE_PCT) if STAKE_DYNAMIC else STAKE_FIX
        stake = float(max(0.0, min(bank, base_stake)))
        if stake == 0 or bank <= 0:
            pnl = 0.0
        else:
            win = int(row["signal"] == row["actual"])
            pnl = stake * PROFIT_F if win else -stake
        bank = max(0.0, bank + pnl)
        trade_pnls.append(pnl)
        equity_idx.append(row["timestamp"])
        equity_vals.append(bank)

    equity = pd.Series(equity_vals, index=pd.to_datetime(equity_idx, utc=True))
    equity = _apply_monthly_fee(equity, FEE_M)
    bank_final = float(equity.iloc[-1])
    pnl_net = bank_final - BANK0

    bets = int(mask.sum())
    wins = int(np.sum(df_tr["signal"].values == df_tr["actual"].values))
    hit_rate = wins / max(bets, 1)
    coverage = bets / len(df)
    dd_max = _max_drawdown(equity)
    ret_pct = pnl_net / BANK0 if BANK0 > 0 else np.nan

    cv_bets = _monthly_cv_count(df_tr["timestamp"])
    cv_net  = _monthly_cv_pnl(df_tr["timestamp"], pd.Series(trade_pnls, index=df_tr.index))

    score = (pnl_net / BANK0) - ALPHA_BETS * cv_bets - BETA_NET * cv_net

    summary = dict(th_up=th_up, th_dn=th_dn, bets=bets, wins=wins,
                   hit_rate=hit_rate, coverage=coverage,
                   pnl_net=pnl_net, bank_final=bank_final, ret_pct=ret_pct,
                   dd_max=dd_max, cv_bets=cv_bets, cv_net=cv_net, score=score)
    # anexamos PnL por trade para posibles análisis
    df_tr["pnl"] = trade_pnls
    return summary, equity, df_tr

def _monthly_table_from_trades(df_tr: pd.DataFrame, equity: pd.Series) -> pd.DataFrame:
    if df_tr.empty:
        return pd.DataFrame(columns=["month","bets","wins","pnl","bank_eom"]).set_index("month")
    df_tr = df_tr.copy()
    df_tr["month"] = _to_month_period(df_tr["timestamp"])
    df_tr["win"] = (df_tr["signal"] == df_tr["actual"]).astype(int)  # ← NUEVO

    g = df_tr.groupby("month", sort=True)
    pnl_m   = g["pnl"].sum()
    bets_m  = g.size()
    wins_m  = g["win"].sum()        # ← SIN apply, sin warning

    # equity fin de mes:
    eq = equity.copy()
    eq_month = eq.index.tz_convert(None).to_period("M")
    bank_eom = pd.Series(index=pnl_m.index, dtype=float)
    for m in pnl_m.index:
        mask_m = (eq_month == m)
        bank_eom.loc[m] = float(eq.loc[mask_m].iloc[-1]) if mask_m.any() else np.nan

    out = pd.DataFrame({"bets": bets_m.astype(int),
                        "wins": wins_m.astype(int),
                        "pnl": pnl_m.astype(float),
                        "bank_eom": bank_eom.astype(float)})
    out.index.name = "month"
    return out

def _monte_carlo_months(pnl_by_month: pd.Series, n_sims=10000, bank0=100.0, seed=1337):
    """Bootstrap por meses (con reemplazo). Devuelve DataFrame con pnl_total/bank_final."""
    rng = np.random.default_rng(seed)
    months = pnl_by_month.index.to_list()
    pnl = pnl_by_month.values.astype(float)
    m = len(months)
    if m == 0:
        return pd.DataFrame(columns=["pnl_total", "bank_final"])
    # sample con reemplazo m meses por simulación
    picks = rng.integers(low=0, high=m, size=(n_sims, m))
    pnl_samples = pnl[picks].sum(axis=1)
    bank_final = bank0 + pnl_samples
    return pd.DataFrame({"pnl_total": pnl_samples, "bank_final": bank_final})

# ===================== MAIN ==========================
def main():
    df = _read_all_preds(DATA_DIR)
    results = []

    # —— Sweep —— #
    if not ASYMMETRIC:
        for i, th in enumerate(TH_GRID, start=1):
            res, _eq, _dftr = _simulate(df, th_up=th, th_dn=th)
            results.append(res)
            print(
                f"Iter {i:>3}: @={th:0.6f} | "
                f"gain_net=${res['pnl_net']:,.0f} | bank_final=${res['bank_final']:,.0f} | "
                f"cv_bets={res['cv_bets']:.3f} cv_net={res['cv_net']:.3f} score={res['score']:.3f}"
            )
    else:
        i = 0
        for thu in TH_GRID_UP:
            for thd in TH_GRID_DN:
                i += 1
                res, _eq, _dftr = _simulate(df, th_up=thu, th_dn=thd)
                results.append(res)
                print(
                    f"Iter {i:>3}: @up={thu:0.6f} @dn={thd:0.6f} | "
                    f"gain_net=${res['pnl_net']:,.0f} | bank_final=${res['bank_final']:,.0f} | "
                    f"cv_bets={res['cv_bets']:.3f} cv_net={res['cv_net']:.3f} score={res['score']:.3f}"
                )

    res_df = pd.DataFrame(results).sort_values(by=["score", "pnl_net"], ascending=[False, False])
    out_csv = OUT_DIR / ("threshold_sweep_fair_asym.csv" if ASYMMETRIC else "threshold_sweep_fair.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\n✅ Resultados guardados en: {out_csv}")

    # —— Elegir mejor threshold —— #
    best = res_df.iloc[0]
    best_up = float(best["th_up"])
    best_dn = float(best["th_dn"])
    print(f"\n🏆 Mejor threshold: UP={best_up:.3f} DOWN={best_dn:.3f} | score={best['score']:.3f} | gain_net=${best['pnl_net']:,.0f}")

    # —— Re-simular con el mejor para sacar tabla mensual —— #
    best_summary, best_equity, best_dftr = _simulate(df, th_up=best_up, th_dn=best_dn)
    monthly = _monthly_table_from_trades(best_dftr, best_equity)
    monthly_path = OUT_DIR / "monthly_results_best.csv"
    monthly.to_csv(monthly_path)
    print(f"📄 Tabla mensual guardada en: {monthly_path} (filas={len(monthly)})")

    # —— Monte Carlo mensual —— #
    if STAKE_DYNAMIC:
        print("⚠️ Aviso: Monte Carlo mensual es una aproximación con stake dinámico (no conserva path exacto).")
    mc = _monte_carlo_months(monthly["pnl"].fillna(0.0), n_sims=MC_N_SIMS, bank0=BANK0, seed=RANDOM_SEED)
    mc_path = OUT_DIR / "montecarlo_samples_best.csv"
    mc.to_csv(mc_path, index=False)

    # métricas MC
    p_profit_total = float((mc["pnl_total"] > 0).mean())
    p_bank_gt_bank0 = float((mc["bank_final"] > BANK0).mean())
    q = mc.quantile([0.05, 0.5, 0.95])
    print("\n🎲 Monte Carlo (por meses, bootstrap con reemplazo):")
    print(f"• sims={MC_N_SIMS} | meses únicos={len(monthly)}")
    print(f"• P(pnl_total>0) = {p_profit_total:.3f}")
    print(f"• Bank final (ppf): 5%={q['bank_final'].loc[0.05]:.2f} | 50%={q['bank_final'].loc[0.5]:.2f} | 95%={q['bank_final'].loc[0.95]:.2f}")
    print(f"• P(bank_final > BANK0={BANK0:.2f}) = {p_bank_gt_bank0:.3f}")
    print(f"📄 Muestras MC guardadas en: {mc_path}")

    # —— Resumen final del mejor threshold —— #
    best_summary_path = OUT_DIR / "best_threshold_summary.json"
    best_summary_out = {
        "best_threshold": {"th_up": best_up, "th_dn": best_dn},
        "bank0": BANK0,
        "stake_dynamic": STAKE_DYNAMIC,
        "stake_pct": STAKE_PCT,
        "stake_fix": STAKE_FIX,
        "profit_f": PROFIT_F,
        "fee_m": FEE_M,
        "summary": best_summary,
        "monte_carlo": {
            "n_sims": MC_N_SIMS,
            "p_profit_total": p_profit_total,
            "bank_final_ppf": {
                "p05": float(q["bank_final"].loc[0.05]),
                "p50": float(q["bank_final"].loc[0.5]),
                "p95": float(q["bank_final"].loc[0.95]),
            }
        }
    }
    import json
    best_summary_path.write_text(json.dumps(best_summary_out, indent=2), encoding="utf-8")
    print(f"📝 Resumen JSON guardado en: {best_summary_path}")

if __name__ == "__main__":
    main()
