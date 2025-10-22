from pathlib import Path
import subprocess, sys, os, datetime

# === Config básica ===
BASE = Path(__file__).resolve().parent
PY = sys.executable
SHOW_CONSOLE = True  # True para ver prints; False para Scheduler silencioso

LOG = BASE / "archivosextras" / "training_log.txt"
LOG.parent.mkdir(exist_ok=True)

# Forzar UTF-8 en hijos
ENV = os.environ.copy()
ENV["PYTHONIOENCODING"] = "utf-8"
ENV["PYTHONUTF8"] = "1"

# Cada entrada puede ser:
# - "package.module"  -> se ejecuta con: python -m package.module
# - "path/to/script.py" -> se ejecuta pasándole la ruta al intérprete
# - ("module_or_path", ["arg1", "arg2"])
SCRIPTS = [
    ("src.features.feature_engineering", []),
    ("model.train.train_model", []),
    ("src.backtest.backtest_strategy", []),
    ("model.BTC.analyse.eval_threshold", []), 
    ("model.BTC.analyse.find_bad_periods", []),  # ← nuevo paso semanal
    ("scripts.update_thresholds_meta", []), 
    ("model.BTC.analyse.update_bad_periods_meta", []),
]


def run(entry):
    if isinstance(entry, tuple):
        script, args = entry
    else:
        script, args = entry, []

    # decidir si es módulo (usar -m) o archivo .py
    run_as_module = False
    script_path = BASE / script
    if script.endswith(".py") or script_path.exists():
        # usar ruta de archivo si existe o si explícitamente tiene .py
        script_file = script_path if script_path.exists() else Path(script)
        cmd = [PY, "-u", str(script_file)] + args
        label = str(script_file)
    else:
        # intentar como módulo
        run_as_module = True
        cmd = [PY, "-u", "-m", script] + args
        label = f"-m {script}"

    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n==== {datetime.datetime.now()} :: RUN {label} ====\n")

        if SHOW_CONSOLE:
            proc = subprocess.Popen(
                cmd,
                cwd=BASE,
                env=ENV,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            for line in proc.stdout:
                print(line, end="")  # consola
                f.write(line)        # log
            code = proc.wait()
        else:
            res = subprocess.run(
                cmd,
                cwd=BASE,
                env=ENV,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            code = res.returncode

        f.write(f"---- EXIT CODE {code} ({label}) ----\n")
        if code != 0:
            raise SystemExit(code)

if __name__ == "__main__":
    print(f"🏁 Iniciando ciclo de entrenamiento con {PY}...")
    for s in SCRIPTS:
        run(s)
    print(f"✅ Entrenamiento OK. Ver log: {LOG}")
