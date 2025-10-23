from pathlib import Path
import subprocess, sys, os, datetime, time

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

# === Scripts a ejecutar semanalmente ===
SCRIPTS = [
    ("src.features.feature_engineering", []),
    ("model.train.train_model", []),
    ("src.backtest.backtest_strategy", []),
    ("model.BTC.analyse.eval_threshold", []),
    ("model.BTC.analyse.find_bad_periods", []),
    ("scripts.update_thresholds_meta", []),
    ("model.BTC.analyse.update_bad_periods_meta", []),
]

# === Intervalo de espera entre ciclos ===
# (7 días = 7 * 24 * 3600 segundos)
SLEEP_SECONDS = 7 * 24 * 3600


def run(entry):
    """Ejecuta un módulo o script con logging."""
    if isinstance(entry, tuple):
        script, args = entry
    else:
        script, args = entry, []

    # decidir si es módulo (usar -m) o archivo .py
    run_as_module = False
    script_path = BASE / script
    if script.endswith(".py") or script_path.exists():
        script_file = script_path if script_path.exists() else Path(script)
        cmd = [PY, "-u", str(script_file)] + args
        label = str(script_file)
    else:
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


def ciclo_entrenamiento():
    """Ejecuta la lista completa de scripts."""
    print(f"🏁 Iniciando ciclo de entrenamiento con {PY} ({datetime.datetime.now()})")
    for s in SCRIPTS:
        run(s)
    print(f"✅ Entrenamiento completado ({datetime.datetime.now()}). Ver log: {LOG}\n")


if __name__ == "__main__":
    print("🕒 Scheduler semanal iniciado. Ctrl+C para detener.\n")
    while True:
        ciclo_entrenamiento()
        print(f"😴 Durmiendo por {SLEEP_SECONDS/3600/24:.1f} días...\n")
        time.sleep(SLEEP_SECONDS)
