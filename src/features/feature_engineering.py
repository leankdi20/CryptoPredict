# src/features/feature_engineering.py
import pandas as pd
import numpy as np

DT_COL = "datetime"

def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd_line(series, fast=12, slow=26):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df[DT_COL] = pd.to_datetime(df[DT_COL], utc=True)
    for c in ["open","high","low","close","volume","quote_volume_usd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open","high","low","close","volume"]).sort_values(DT_COL).reset_index(drop=True)

    # Retornos y lags
    df["ret_1"]  = df["close"].pct_change(1)
    df["logret_1"] = np.log(df["close"]).diff(1)
    df["ret_2"]  = df["close"].pct_change(2)
    df["ret_3"]  = df["close"].pct_change(3)
    df["ret_4"]  = df["close"].pct_change(4)
    df["ret_6"]  = df["close"].pct_change(6)
    df["ret_8"]  = df["close"].pct_change(8)
    df["ret_24"] = df["close"].pct_change(24)

    # Volatilidad
    df["vol_24"] = df["ret_1"].rolling(24).std()
    df["vol_72"] = df["ret_1"].rolling(72).std()

    # Medias
    df["sma_20"]  = df["close"].rolling(20).mean()
    df["sma_50"]  = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    df["ema_20"]  = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"]  = df["close"].ewm(span=50, adjust=False).mean()

    df["sma_ratio_20_50"]   = df["sma_20"] / (df["sma_50"] + 1e-12)
    df["sma_ratio_50_200"]  = df["sma_50"] / (df["sma_200"] + 1e-12)

    # RSI / MACD / Bandas
    df["rsi_14"] = rsi(df["close"], 14)
    df["macd"]   = macd_line(df["close"], 12, 26)
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    bb_mid = df["sma_20"]
    bb_std = df["close"].rolling(20).std()
    df["bb_pos"] = (df["close"] - bb_mid) / (2*bb_std + 1e-12)

    # Volumen
    df["vol_z20"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-12)

    # Velas y ATR
    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = (df["high"] - df[["close","open"]].max(axis=1)).clip(lower=0)
    df["lower_wick"] = (df[["close","open"]].min(axis=1) - df["low"]).clip(lower=0)
    df["body_pct"] = df["body"] / (df["close"].shift(1) + 1e-12)
    df["upper_wick_pct"] = df["upper_wick"] / (df["close"].shift(1) + 1e-12)
    df["lower_wick_pct"] = df["lower_wick"] / (df["close"].shift(1) + 1e-12)

    tr = pd.concat([
        (df["high"]-df["low"]),
        (df["high"]-df["close"].shift(1)).abs(),
        (df["low"]-df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["true_range"] = tr
    df["atr14"] = tr.rolling(14).mean()

    # Tendencia/régimen
    df["slope_sma20"] = df["sma_20"].diff()
    df["trend_flag"]  = (df["sma_20"] > df["sma_50"]).astype(int)
    df["vol_regime"]  = (df["atr14"] > df["atr14"].rolling(100).median()).astype(int)

    # Estacionalidad (1H)
    dt = df[DT_COL].dt
    df["sin_hour"] = np.sin(2*np.pi*dt.hour/24)
    df["cos_hour"] = np.cos(2*np.pi*dt.hour/24)
    df["sin_dow"]  = np.sin(2*np.pi*dt.dayofweek/7)
    df["cos_dow"]  = np.cos(2*np.pi*dt.dayofweek/7)

    # Ratios con ATR
    df["body_over_atr"]  = df["body"]  / (df["atr14"] + 1e-12)
    df["range_over_atr"] = df["range"] / (df["atr14"] + 1e-12)
    df["close_sma20_atr"]= (df["close"] - df["sma_20"]) / (df["atr14"] + 1e-12)

    # Lags
    df["close_lag1"]   = df["close"].shift(1)
    df["ret_1_lag1"]   = df["ret_1"].shift(1)

    # Target (próxima vela)
    df["target_ret"] = df["close"].shift(-1) / df["close"] - 1.0
    df["target_up"]  = (df["target_ret"] > 0).astype(int)

    feat_cols = [
        "ret_1","logret_1","ret_2","ret_3","ret_4","ret_6","ret_8","ret_24",
        "vol_24","vol_72",
        "sma_20","sma_50","sma_200","ema_20","ema_50",
        "sma_ratio_20_50","sma_ratio_50_200",
        "rsi_14","macd","macd_signal","macd_hist",
        "bb_pos","vol_z20","quote_volume_usd",
        "close_lag1","ret_1_lag1",
        "body","range","upper_wick","lower_wick","body_pct","upper_wick_pct","lower_wick_pct",
        "true_range","atr14","slope_sma20","trend_flag","vol_regime",
        "sin_hour","cos_hour","sin_dow","cos_dow",
        "body_over_atr","range_over_atr","close_sma20_atr"
    ]
    keep = [DT_COL,"open","high","low","close","volume","target_up","target_ret"] + feat_cols
    df = df[keep].dropna().reset_index(drop=True)
    return df
