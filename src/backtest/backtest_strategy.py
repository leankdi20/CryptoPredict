# src/backtest/backtest_strategy.py
from __future__ import annotations
import argparse
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass

ROOT       = Path(__file__).resolve().parents[2]
META_PATH  = ROOT / "model" / "model_meta.json"
FEATS_CSV  = ROOT / "data" / "derived" / "BTCUSDT" / "features_1h.csv"
BEST_PKL   = ROOT / "model" / "models" / "best_model_xgb.joblib"
OUT_TRADES = ROOT / "archivosextras" / "backtest_trades_1h.csv"

def _load_meta():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    feat_cols = meta["feat_cols"]

    price_yes = float(meta.get("price_yes", meta.get("polymarket", {}).get("price_yes", 0.5)))
    pm = meta.get("polymarket", {})
    ret_yes = float(pm.get("return_win",    1.0 / max(price_yes, 1e-9) - 1.0))
    ret_no  = float(pm.get("return_win_no", 1.0 / max(1.0 - price_yes, 1e-9) - 1.0))

    thr_up   = float(meta.get("policy_thr_up",   0.53))
    thr_down = float(meta.get("policy_thr_down", 0.53))
    min_ev   = float(meta.get("policy_min_ev",   0.02))
    min_kl   = float(meta.get("policy_min_kelly",0.03))
    return feat_cols, ret_yes, ret_no, thr_up, thr_down, min_ev, min_kl

def _kelly_frac(p, b):
    return float(max(0.0, min(1.0, (b * p - (1 - p)) / max(b, 1e-9))))

def _resolve_model():
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        p = meta.get("model_pkl")
        if p:
            pp = Path(p)
            for c in (pp, ROOT / pp, ROOT / "model" / "models" / pp.name):
                if c.exists():
                    return c
    except Exception:
        pass
    return BEST_PKL

@dataclass
class PolicyResult:
    n_bets: int
    winrate: float
    pnl_total: float
    pnl_mean: float
    dd_max: float
    sharpe_h: float

def _metrics(pnl_series: pd.Series) -> tuple[float, float]:
    eq = pnl_series.cumsum()
    dd = eq - eq.cummax()
    dd_min = float(dd.min()) if len(dd) else 0.0
    mu = pnl_series.mean() if len(pnl_series) else 0.0
    sd = pnl_series.std(ddof=1) if len(pnl_series) > 1 else 0.0
    sharpe = float(mu / sd) if sd > 0 else 0.0
    return abs(dd_min), sharpe

def run_backtest(thr_up: float|None, thr_down: float|None,
                 kelly_safety: float = 0.25,
                 min_ev: float|None = None,
                 min_kelly: float|None = None,
                 dt_from: str|None = None, dt_to: str|None = None) -> PolicyResult:

    feat_cols, ret_yes, ret_no, thr_u_meta, thr_d_meta, min_ev_meta, min_kl_meta = _load_meta()
    if thr_up   is None: thr_up   = thr_u_meta
    if thr_down is None: thr_down = thr_d_meta
    if min_ev   is None: min_ev   = min_ev_meta
    if min_kelly is None: min_kelly = min_kl_meta

    model_path = _resolve_model()
    model = joblib.load(model_path)

    df = pd.read_csv(FEATS_CSV, parse_dates=["datetime"])
    if "target_up" not in df.columns:
        raise ValueError("No encuentro 'target_up' en features_1h.csv. Generalo en feature_engineering/make_features.")

    df = df.dropna(subset=feat_cols + ["target_up"]).copy().reset_index(drop=True)

    if dt_from:
        df = df[df["datetime"] >= pd.Timestamp(dt_from, tz="UTC")]
    if dt_to:
        df = df[df["datetime"] <  pd.Timestamp(dt_to,   tz="UTC")]
    df = df.reset_index(drop=True)

    p_up = model.predict_proba(df[feat_cols].astype(float))[:, 1]
    df["p_up"] = p_up
    df["p_dn"] = 1.0 - df["p_up"]
    df["EV_up"] = df["p_up"] * ret_yes - (1.0 - df["p_up"]) * 1.0
    df["EV_dn"] = df["p_dn"] * ret_no  - (1.0 - df["p_dn"]) * 1.0

    rows, pnls = [], []
    for _, r in df.iterrows():
        side, b, p, ev = "NO_BET", 0.0, 0.0, 0.0
        if (r.p_up >= thr_up) and (r.EV_up >= min_ev):
            side, b, p, ev = "UP", ret_yes, float(r.p_up), float(r.EV_up)
        elif (r.p_dn >= thr_down) and (r.EV_dn >= min_ev):
            side, b, p, ev = "DOWN", ret_no, float(r.p_dn), float(r.EV_dn)

        k_raw = _kelly_frac(p, b) if side != "NO_BET" else 0.0
        if k_raw < min_kelly:
            side, b, p, ev, k_raw = "NO_BET", 0.0, 0.0, 0.0, 0.0

        stake = k_raw * kelly_safety

        if side == "UP":
            payoff = (b if int(r.target_up) == 1 else -1.0) * stake
        elif side == "DOWN":
            payoff = (b if int(r.target_up) == 0 else -1.0) * stake
        else:
            payoff = 0.0

        pnls.append(payoff)
        rows.append({
            "datetime": r["datetime"],
            "p_up": round(float(r.p_up), 6),
            "p_dn": round(float(r.p_dn), 6),
            "EV": round(float(ev), 6),
            "kelly_raw": round(k_raw, 6),
            "stake": round(stake, 6),
            "side": side,
            "target_up": int(r.target_up),
            "payoff": round(payoff, 6),
        })

    trades = pd.DataFrame(rows)
    OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(OUT_TRADES, index=False)
    print(f"✅ Guardado detalle: {OUT_TRADES}")

    bets = trades[trades.side != "NO_BET"].copy()
    n_bets = len(bets)
    winrate = (bets.assign(win=np.where(
        (bets.side == "UP") & (bets.target_up == 1), 1,
        np.where((bets.side == "DOWN") & (bets.target_up == 0), 1, 0)
    ))["win"].mean() if n_bets else 0.0)

    pnl_total = float(bets["payoff"].sum())
    pnl_mean  = float(bets["payoff"].mean()) if n_bets else 0.0
    dd_max, sharpe_h = _metrics(trades["payoff"])

    print("\n📊 RESUMEN POLÍTICA")
    print(f"  Bets: {n_bets:,}")
    print(f"  Winrate: {winrate:.3f}")
    print(f"  PnL total: {pnl_total:.3f}")
    print(f"  PnL medio por bet: {pnl_mean:.3f}")
    print(f"  Max Drawdown (unit): {dd_max:.3f}")
    print(f"  Sharpe horario: {sharpe_h:.3f}")

    return PolicyResult(n_bets, winrate, pnl_total, pnl_mean, dd_max, sharpe_h)

def run_grid(g_from=0.50, g_to=0.60, g_step=0.01, kelly_safety=0.25, min_ev=None, min_kelly=None,
             dt_from=None, dt_to=None):
    rows = []
    thr = g_from
    while thr <= g_to + 1e-12:
        print(f"\n—— grid thr={thr:.3f} (ambos lados) ——")
        res = run_backtest(thr_up=thr, thr_down=thr,
                           kelly_safety=kelly_safety,
                           min_ev=min_ev, min_kelly=min_kelly,
                           dt_from=dt_from, dt_to=dt_to)
        rows.append({
            "thr": round(thr, 3),
            "bets": res.n_bets,
            "winrate": round(res.winrate, 3),
            "pnl_total": round(res.pnl_total, 3),
            "pnl_mean": round(res.pnl_mean, 3),
            "dd_max": round(res.dd_max, 3),
            "sharpe_h": round(res.sharpe_h, 3),
        })
        thr = round(thr + g_step, 3)
    df = pd.DataFrame(rows)
    print("\n📈 GRID RESULTADOS")
    print(df.to_string(index=False))
    out = ROOT / "archivosextras" / "backtest_grid.csv"
    df.to_csv(out, index=False)
    print(f"✅ Guardado grid: {out}")

def parse_args():
    ap = argparse.ArgumentParser(description="Backtest horario con misma política que el live.")
    ap.add_argument("--thr_up", type=float, default=None)
    ap.add_argument("--thr_down", type=float, default=None)
    ap.add_argument("--kelly_safety", type=float, default=0.25)
    ap.add_argument("--min_ev", type=float, default=None, help="EV mínimo (default: meta.policy_min_ev)")
    ap.add_argument("--min_kelly", type=float, default=None, help="Kelly mínimo (default: meta.policy_min_kelly)")
    ap.add_argument("--dt_from", type=str, default=None)
    ap.add_argument("--dt_to", type=str, default=None)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--g_from", type=float, default=0.50)
    ap.add_argument("--g_to", type=float, default=0.60)
    ap.add_argument("--g_step", type=float, default=0.01)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.grid:
        run_grid(args.g_from, args.g_to, args.g_step,
                 kelly_safety=args.kelly_safety,
                 min_ev=args.min_ev, min_kelly=args.min_kelly,
                 dt_from=args.dt_from, dt_to=args.dt_to)
    else:
        run_backtest(args.thr_up, args.thr_down,
                     kelly_safety=args.kelly_safety,
                     min_ev=args.min_ev, min_kelly=args.min_kelly,
                     dt_from=args.dt_from, dt_to=args.dt_to)
