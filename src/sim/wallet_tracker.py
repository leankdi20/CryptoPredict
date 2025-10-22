import pandas as pd
from pathlib import Path
from datetime import datetime
import pytz

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "BTCUSDT" / "ohlcv_spot_1h.csv"
LOG_CSV = ROOT / "archivosextras" / "decisions_log.csv"
WALLET_CSV = ROOT / "archivosextras" / "wallet_history.csv"

INITIAL_BANK = 250.0
STAKE_PER_BET = 5.0

def update_wallet():
    now = datetime.now(tz=pytz.UTC).replace(minute=0, second=0, microsecond=0)

    try:
        decisions = pd.read_csv(LOG_CSV, parse_dates=["ts_entry", "ts_settle"])
    except FileNotFoundError:
        print("⚠️ No se encontró decisions_log.csv")
        return

    try:
        raw_data = pd.read_csv(RAW_CSV)
        raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"], unit="ms").dt.floor("h")

    except FileNotFoundError:
        print("⚠️ No se encontró ohlcv_spot_1h.csv")
        return

    try:
        wallet = pd.read_csv(WALLET_CSV, parse_dates=["ts_entry", "ts_settle"])
    except FileNotFoundError:
        wallet = pd.DataFrame()

    processed_pairs = (
    set(zip(wallet["ts_entry"], wallet["ts_settle"])) if not wallet.empty else set()
)
    new_rows = []
    last_bank = wallet["bank_after"].iloc[-1] if not wallet.empty else INITIAL_BANK

    for i, row in decisions.iterrows():
        if (row["ts_entry"], row["ts_settle"]) in processed_pairs:
            continue
        if row["ts_settle"] > now:
            continue

        ts_entry = row["ts_entry"]
        ts_settle = row["ts_settle"]
        signal = row["signal"]
        price_entry = row["price_entry"]

        settle_naive = pd.Timestamp(ts_settle).replace(tzinfo=None)
        row_candle = raw_data[raw_data["timestamp"] == settle_naive]

        if row_candle.empty:
            print(f"❗ No hay vela para {settle_naive}")
            continue

        price_settle = row_candle["close"].values[0]
        result = "WIN" if (signal == "UP" and price_settle > price_entry) or (signal == "DOWN" and price_settle < price_entry) else "LOSE"
        pnl = STAKE_PER_BET if result == "WIN" else -STAKE_PER_BET
        bank_before = last_bank
        bank_after = bank_before + pnl
        last_bank = bank_after  # 🔁 acumula para la próxima iteración

        new_rows.append({
            "ts_entry": ts_entry,
            "ts_settle": ts_settle,
            "side": signal,
            "stake": STAKE_PER_BET,
            "price_entry": price_entry,
            "price_settle": price_settle,
            "result": result,
            "pnl": pnl,
            "bank_before": bank_before,
            "bank_after": bank_after,
            "source_row_id": i
        })

    if new_rows:
        wallet = pd.concat([wallet, pd.DataFrame(new_rows)], ignore_index=True)
        wallet.to_csv(WALLET_CSV, index=False)
        print(f"✅ Wallet actualizado ({len(new_rows)} nuevas apuestas).")
        print(f"💰 Último bank: {wallet['bank_after'].iloc[-1]:.2f} USDT | Último resultado: {new_rows[-1]['result']}")
    else:
        print("ℹ️ No hay apuestas nuevas para procesar.")

if __name__ == "__main__":
    update_wallet()
