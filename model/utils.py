from datetime import datetime, timedelta
import pytz
import joblib
import json
import pandas as pd

def load_model_meta(meta_json_path, model_file_path):
    try:
        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except FileNotFoundError:
        meta = {}

    model = joblib.load(model_file_path)
    calibrator = None

    if "calibrator_path" in meta:
        calibrator = joblib.load(meta["calibrator_path"])

    return model, calibrator


def apply_policy(prob_up, be_up=0.53, thr_up=0.527, min_ev=0.02, min_kelly=0.02):
    p_up = prob_up
    p_down = 1 - p_up

    EV_up = (p_up - be_up) / be_up
    EV_dn = (p_down - 0.47) / 0.47

    k_up = (p_up - be_up) / (1 - be_up)
    k_dn = (p_down - 0.47) / (1 - 0.47)

    if p_up >= thr_up and EV_up >= min_ev and k_up >= min_kelly:
        return "BET_UP", p_up, k_up, EV_up
    elif p_down >= 0.527 and EV_dn >= min_ev and k_dn >= min_kelly:
        return "BET_DOWN", p_down, k_dn, EV_dn
    else:
        return "NO_BET", None, None, None


def _et_window_from_now():
    """Devuelve la hora actual ET (Nueva York) y el cierre estimado de la hora actual"""
    utc_now = datetime.now(pytz.UTC)
    et = pytz.timezone("America/New_York")
    et_now = utc_now.astimezone(et)

    start_et = et_now.replace(minute=0, second=0, microsecond=0)
    end_et = start_et + timedelta(hours=1)
    end_utc = end_et.astimezone(pytz.UTC)
    end_cr = end_utc.astimezone(pytz.timezone("America/Costa_Rica"))
    return start_et, end_et, end_utc, end_cr


def _fmt_hour(dt):
    return dt.strftime("%I%p").lstrip("0").replace("am", "AM").replace("pm", "PM")
