# main.py — orquestador horario
from __future__ import annotations
import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path
import subprocess

# ✅ IMPORT CORRECTO
from src.sim.wallet_tracker import update_wallet

# ── Paths / entorno ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
PY   = sys.executable
LOG  = ROOT / "archivosextras" / "trading_log.txt"
LOCK = ROOT / "archivosextras" / ".hourly.lock"
LOG.parent.mkdir(parents=True, exist_ok=True)

ENV = os.environ.copy()
ENV["PYTHONIOENCODING"] = "utf-8"
ENV["PYTHONUTF8"] = "1"

PIPELINE = [
    ("src.data.pipeline_csv",            []),
    ("src.feature.make_features_1h",     []),
    ("src.preprocessing.make_Xy_1h",     []),
    ("src.inference.update_and_predict", []),
    ("src.decision.decide_and_recommend", []),  # <- corre antes que wallet ✅
]
BACKTEST_MOD = ("src.backtest.backtest_strategy", [])


# ── Helpers ────────────────────────────────────────────────────────────────────
def _stamp() -> str:
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _append_log(msg: str, echo: bool = True):
    line = f"{_stamp()} :: {msg}\n"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line)
    if echo:
        print(msg)

def run_module(mod: str, args: list[str], quiet: bool=False) -> int:
    cmd = [PY, "-u", "-m", mod, *args]
    header = f"\n==== {_stamp()} :: RUN {mod} {' '.join(args)} ====\n"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(header)
        if quiet:
            res = subprocess.run(
                cmd, cwd=ROOT, env=ENV,
                stdout=f, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace"
            )
            code = res.returncode
        else:
            proc = subprocess.Popen(
                cmd, cwd=ROOT, env=ENV,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1
            )
            for l in proc.stdout:
                print(l, end="")
                f.write(l)
            code = proc.wait()
        f.write(f"---- EXIT CODE {code} ({mod}) ----\n")
    return code

def acquire_lock() -> bool:
    """Intenta adquirir el lock; False si hay otro proceso reciente (<70 min)."""
    try:
        if LOCK.exists():
            age = time.time() - LOCK.stat().st_mtime
            if age < 70 * 60:
                return False
        LOCK.write_text(_stamp(), encoding="utf-8")
        return True
    except Exception:
        # Si hay problemas con el lock, preferimos no bloquear la corrida.
        return True

def release_lock():
    try:
        if LOCK.exists():
            LOCK.unlink()
    except Exception:
        pass

def sleep_until_next_hour(offset_seconds: int = 61):
    """Duerme hasta la próxima hora + offset_seconds (por ej. hh:01)."""
    now = dt.datetime.now()
    next_hour = now.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
    wake_at = next_hour + dt.timedelta(seconds=offset_seconds)
    secs = (wake_at - now).total_seconds()
    time.sleep(max(1, int(secs)))

def _wait_for_decision_csv(timeout: float = 3.0) -> bool:
    """Espera hasta 'timeout' a que exista y tenga contenido decisions_log.csv."""
    from time import monotonic, sleep
    csv = ROOT / "archivosextras" / "decisions_log.csv"
    t0 = monotonic()
    while monotonic() - t0 < timeout:
        try:
            if csv.exists() and csv.stat().st_size > 0:
                return True
        except Exception:
            pass
        sleep(0.1)
    return csv.exists() and csv.stat().st_size > 0


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Ciclo horario: pipeline→features→infer→decide→wallet")
    ap.add_argument("--loop", action="store_true", help="correr en bucle, cada hora")
    ap.add_argument("--once", action="store_true", help="correr una sola vez (default)")
    ap.add_argument("--quiet", action="store_true", help="no imprimir en consola (solo log)")
    ap.add_argument("--backtest", action="store_true", help="correr backtest al final")
    ap.add_argument("--force", action="store_true", help="ignorar lock y forzar el ciclo")
    return ap.parse_args()


# ── Ciclo ─────────────────────────────────────────────────────────────────────
def run_cycle(quiet: bool=False, do_backtest: bool=False, force: bool=False) -> int:
    if not force and not acquire_lock():
        _append_log("🔒 lock activo; ciclo saltado.", echo=True)
        return 0
    if force:
        print("⚠️  --force: ignorando lock y forzando ciclo.")

    try:
        # 1) Pipeline horario (hasta decidir)
        for mod, args in PIPELINE:
            code = run_module(mod, args, quiet=quiet)
            if code != 0:
                return code

        # 2) ✅ Actualizar billetera inmediatamente después de decidir
        _append_log("RUN wallet_tracker.update_wallet()", echo=True)
        if not _wait_for_decision_csv(timeout=3.0):
            _append_log("⚠️ decisions_log.csv no está listo; continúo igual.", echo=True)
        try:
            update_wallet()
            _append_log("wallet_tracker: OK", echo=True)
        except Exception as e:
            _append_log(f"wallet_tracker: ERROR → {e!r}", echo=True)

        # 3) Opcional: backtest
        if do_backtest:
            run_module(BACKTEST_MOD[0], BACKTEST_MOD[1], quiet=quiet)

        return 0
    finally:
        release_lock()
        time.sleep(2)


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    print(f"🏁 Iniciando ciclo con {PY}  @ {ROOT}")

    if args.loop and not args.once:
        while True:
            print(f"\n⏳ {_stamp()}  → nuevo ciclo")
            rc = run_cycle(quiet=args.quiet, do_backtest=args.backtest, force=args.force)
            if rc != 0:
                print(f"❌ ciclo terminó con código {rc}. Reintentando en la próxima hora…")

            # ❌ (REMOVIDO) update_wallet() duplicado aquí: ya corre dentro de run_cycle()

            print("🕒 durmiendo hasta la próxima hora (hh:01)…")
            sleep_until_next_hour(offset_seconds=61)
    else:
        rc = run_cycle(quiet=args.quiet, do_backtest=args.backtest, force=args.force)
        if rc == 0:
            print(f"✅ Ciclo OK. Log → {LOG}")
        else:
            print(f"❌ Ciclo falló con código {rc}. Revisá el log → {LOG}")
        time.sleep(5)
