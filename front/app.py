from flask import Flask, render_template
import pandas as pd
from pathlib import Path

app = Flask(__name__)

@app.route('/')
def dashboard():
    # === Ruta absoluta al CSV ===
    BASE_DIR = Path(__file__).resolve().parent.parent
    df_path = BASE_DIR / "archivosextras" / "wallet_history.csv"

    # Mostrar mensaje si el archivo no existe
    if not df_path.exists():
        return f"<h3 style='color:red;'>⚠️ No se encontró el archivo: {df_path}</h3>", 404

    # === Cargar CSV ===
    df = pd.read_csv(df_path)

    # Asegurar que las columnas esperadas existan
    required_cols = ["ts_settle", "pnl", "result"]
    for col in required_cols:
        if col not in df.columns:
            return f"<h3 style='color:red;'>❌ Falta la columna '{col}' en {df_path.name}</h3>", 500

    # === Ordenar de más reciente a más antigua ===
    df["ts_settle"] = pd.to_datetime(df["ts_settle"], errors="coerce")
    df = df.sort_values("ts_settle", ascending=False).reset_index(drop=True)

    # === Datos para tabla ===
    headers = df.columns.tolist()
    data = df.to_dict(orient="records")

    # === Gráfico: ganancias/pérdidas por día ===
    df["date"] = df["ts_settle"].dt.date
    chart_data = df.groupby("date")["pnl"].sum().reset_index()
    chart_labels = chart_data["date"].astype(str).tolist()
    chart_values = chart_data["pnl"].round(2).tolist()

    # === “No_Bet” logs ===
    nobet_path = BASE_DIR / "archivosextras" / "no_bet_log.csv"
    if nobet_path.exists():
        nobet_df = pd.read_csv(nobet_path, parse_dates=["timestamp"])
        nobet_df = nobet_df.sort_values("timestamp", ascending=False).head(15)
        nobets = nobet_df.to_dict(orient="records")
    else:
        nobets = ["No hay registros de NO_BET."]


        # === Estadísticas resumen ===
    total_bets = len(df[df["result"].isin(["WIN", "LOSE"])])
    wins = len(df[df["result"] == "WIN"])
    losses = len(df[df["result"] == "LOSE"])

    win_pct = round((wins / total_bets) * 100, 2) if total_bets > 0 else 0
    loss_pct = round((losses / total_bets) * 100, 2) if total_bets > 0 else 0
    total_pnl = round(df["pnl"].sum(), 2)


    # === Renderizar plantilla ===
    return render_template(
        "dashboard.html",
        headers=headers,
        data=data,
        chart_labels=chart_labels,
        chart_values=chart_values,
        nobets=nobets,
        total_bets=total_bets,
        wins=wins,
        losses=losses,
        win_pct=win_pct,
        loss_pct=loss_pct,
        total_pnl=total_pnl
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
