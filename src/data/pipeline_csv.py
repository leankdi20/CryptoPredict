# src/data/pipeline_csv.py
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
OUT_DIR     = ROOT / "data" / "raw" / "BTCUSDT"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV     = OUT_DIR / "ohlcv_spot_1h.csv"

SYMBOL      = "BTC/USDT"
TIMEFRAME   = "1h"
EXCHANGE_ID = "binance"

# Costa Rica TZ (sin DST)
TZ_CR       = ZoneInfo("America/Costa_Rica")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)

def to_utc_ts_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza columnas. Esperamos: timestamp(ms), open, high, low, close, volume
    Agregamos: datetime(UTC ISO8601) y quote_volume_usd≈volume*close para BTC/USDT
    """
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df.copy()
    df = df[cols]
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    # como es par USDT, aproximamos quote_volume_usd = close * volume
    df["quote_volume_usd"] = pd.to_numeric(df["close"], errors="coerce") * pd.to_numeric(df["volume"], errors="coerce")
    # tipos numéricos
    for c in ["open","high","low","close","volume","quote_volume_usd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open","high","low","close","volume"]).sort_values("datetime").reset_index(drop=True)
    return df

def _load_last_csv() -> pd.DataFrame:
    if OUT_CSV.exists():
        return pd.read_csv(OUT_CSV)
    return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime","quote_volume_usd"])

def _fetch_from_ccxt(since_ms: int | None, until_ms: int | None) -> pd.DataFrame:
    """
    Descarga OHLCV 1H de Binance usando ccxt, desde 'since_ms' hasta 'until_ms' (no estricto).
    Devuelve DataFrame con columnas estándar.
    """
    try:
        import ccxt  # noqa
    except Exception as e:
        print(f"⚠️ ccxt no está disponible ({e}). No se descargan velas nuevas.")
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    ex_cls = getattr(ccxt, EXCHANGE_ID)
    ex = ex_cls({"enableRateLimit": True})
    ex.load_markets()

    ohlcv_all = []
    limit = 1000
    # si no hay since, pedimos el último bloque
    cursor = since_ms
    tries = 0

    while True:
        tries += 1
        try:
            ohlcv = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=cursor, limit=limit)
        except Exception as e:
            print(f"⚠️ Error de red/ccxt: {e}")
            break
        if not ohlcv:
            break
        ohlcv_all.extend(ohlcv)
        # avanzar cursor
        last_ts = ohlcv[-1][0]
        # cortar si ya pasamos el until
        if until_ms is not None and last_ts >= until_ms:
            break
        # evitar bucles infinitos
        if cursor is not None and last_ts <= cursor:
            break
        cursor = last_ts + 1

        # por seguridad, limita iteraciones
        if tries > 10_000:
            print("⚠️ Demasiados ciclos de descarga, corto por seguridad.")
            break

    if not ohlcv_all:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    df = pd.DataFrame(ohlcv_all, columns=["timestamp","open","high","low","close","volume"])
    return _ensure_columns(df)

def _print_timing_banner(last_dt_utc: datetime, now_utc: datetime):
    """
    Explica los tiempos de cierre de vela:
      - Binance timbra en UTC (hora Binance = UTC).
      - Velas 1H cierran en minuto 00 de cada hora UTC (…18:00, 19:00, 20:00…).
      - Costa Rica = UTC-6 (sin DST).
    """
    next_close_utc = floor_to_hour(now_utc) + timedelta(hours=1)
    print("No hay velas nuevas para DESCARGAR.")
    print(f"  • Última RAW: {last_dt_utc} | Ahora UTC: {now_utc}")
    print(f"  • Siguiente cierre esperado (UTC): {next_close_utc}")
    # Ayuda de zonas horarias
    last_cr  = last_dt_utc.astimezone(TZ_CR)
    now_cr   = now_utc.astimezone(TZ_CR)
    next_cr  = next_close_utc.astimezone(TZ_CR)
    print("  • Ayuda horarios:")
    print(f"      - Última RAW (Costa Rica): {last_cr}")
    print(f"      - Ahora (Costa Rica):      {now_cr}")
    print(f"      - Próximo cierre (CR):     {next_cr}")
    print("  • Regla: la vela con timestamp 18:00 UTC representa 17:00–18:00 y se confirma a las 18:00:00 UTC.")
    print("           Apostás para la HORA SIGUIENTE (apenas cierra la vela actual).")

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    # Cargamos RAW actual
    last = _load_last_csv()
    if not last.empty:
        last["datetime"] = pd.to_datetime(last["datetime"], utc=True, errors="coerce")
        last = last.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # Estado actual
    now_utc = utc_now()
    now_floor = floor_to_hour(now_utc)

    # Fecha/hora desde la cual pedir nuevas velas
    since_dt = None
    if not last.empty:
        last_dt = last["datetime"].iloc[-1]
        since_dt = last_dt + timedelta(hours=1)  # próxima vela posterior al último registro
    else:
        # si no hay archivo, arrancamos 2022-01-01 UTC
        since_dt = datetime(2022,1,1,0,0,0,tzinfo=timezone.utc)

    # Hasta dónde tiene sentido bajar: hasta la hora cerrada más reciente (no la hora en curso)
    until_dt = now_floor  # última hora cerrada (inclusive)

    # Si no hay nada nuevo que bajar (archivo ya está al día)
    if not last.empty and since_dt > until_dt:
        # imprimir banner de tiempos para que sea claro
        _print_timing_banner(last["datetime"].iloc[-1], now_utc)
        # igual devolvemos 0 para que siga el pipeline (features pueden estar atrasados)
        return

    # Descarga incremental
    df_new = _fetch_from_ccxt(since_ms=to_utc_ts_ms(since_dt) if since_dt else None,
                              until_ms=to_utc_ts_ms(until_dt))

    if df_new.empty:
        # no hubo nuevas velas desde el exchange
        if last.empty:
            print("⚠️ No se descargó nada y no existe RAW previo. Verificá conexión/ccxt.")
        else:
            _print_timing_banner(last["datetime"].iloc[-1], now_utc)
        return

    # Unimos con el historial y guardamos
    combined = pd.concat([last, df_new], ignore_index=True) if not last.empty else df_new
    combined = _ensure_columns(combined).drop_duplicates(subset=["timestamp"]).sort_values("datetime").reset_index(drop=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"✅ {OUT_CSV.as_posix()} actualizado | filas totales: {len(combined):,}")

    # Info útil
    print(f"⏱️ Última vela: {combined['datetime'].iloc[-1]}")

        # ── Info Polymarket: ventana horaria en ET (Nueva York) ───────────────────
    from zoneinfo import ZoneInfo
    TZ_NY = ZoneInfo("America/New_York")
    TZ_CR = ZoneInfo("America/Costa_Rica")

    now_utc = utc_now()
    now_et  = now_utc.astimezone(TZ_NY)
    # ventana ET actual (ej. "3–4 PM ET")
    et_start = now_et.replace(minute=0, second=0, microsecond=0)
    et_end   = et_start + timedelta(hours=1)

    print("🕒 Polymarket (ET):")
    print(f"  • Ventana actual: {et_start:%Y-%m-%d %I:%M %p} – {et_end:%I:%M %p} ET")
    print(f"  • Cierre de esta ventana (ET):  {et_end}")
    print(f"    = UTC:       {et_end.astimezone(timezone.utc)}")
    print(f"    = Costa Rica:{et_end.astimezone(TZ_CR)}")
    print("  • Regla práctica: apostá tras el cierre de una vela UTC (minuto :00).")
    print("    Esa predicción cubre la ventana ET en curso que cierra 1h después.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
