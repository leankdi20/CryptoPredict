from flask import Flask, render_template, render_template_string
import pandas as pd
from pathlib import Path

app = Flask(__name__)

@app.route('/')
def tabla():
    BASE_DIR = Path(__file__).resolve().parent.parent
    df_path = BASE_DIR / "archivosextras" / "wallet_history.csv"

    df = pd.read_csv(df_path)
    # Opcional: redondear o formatear números
    if 'profit' in df.columns:
        df['profit'] = df['profit'].round(3)
    if 'balance_after' in df.columns:
        df['balance_after'] = df['balance_after'].round(2)

    table_html = df.to_html(classes='table table-striped table-hover table-bordered', index=False)
    return render_template('dashboard.html', table=table_html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,  debug=True)
