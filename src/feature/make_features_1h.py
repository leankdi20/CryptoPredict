import pandas as pd
from pathlib import Path
from src.features.feature_engineering import build_features, DT_COL
from src.io._io_utils import save_incremental_csv

RAW_CSV   = Path("data/raw/BTCUSDT/ohlcv_spot_1h.csv")
OUT_FEATS = Path("data/derived/BTCUSDT/features_1h.csv")
LOOKBACK  = 1000  # velas para recomputar indicadores sin “cortes”

def main():
    ohlcv = pd.read_csv(RAW_CSV, parse_dates=[DT_COL])
    ohlcv[DT_COL] = pd.to_datetime(ohlcv[DT_COL], utc=True)

    if OUT_FEATS.exists():
        cur = pd.read_csv(OUT_FEATS, parse_dates=[DT_COL])
        last_dt = cur[DT_COL].max()
        base = ohlcv[ohlcv[DT_COL] <= last_dt].tail(LOOKBACK)
        tail = ohlcv[ohlcv[DT_COL] >  last_dt]
        recompute = pd.concat([base, tail], ignore_index=True)
        recomputed = build_features(recompute)
        new_part  = recomputed[recomputed[DT_COL] > last_dt]
    else:
        new_part  = build_features(ohlcv)

    full = save_incremental_csv(new_part, OUT_FEATS, key=DT_COL, parse_date_keys=[DT_COL])

    print(f"✅ Features actualizados → {OUT_FEATS} ({len(full):,} filas)")

if __name__ == "__main__":
    main()
