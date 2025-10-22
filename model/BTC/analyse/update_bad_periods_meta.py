# model/update_bad_periods_meta.py
from pathlib import Path
import json

META = Path("model/model_meta.json")
SUGG = Path("model/BTC/analysis/bad_periods_suggestion.json")

def main():
    if not SUGG.exists():
        raise FileNotFoundError(str(SUGG))
    meta = json.loads(META.read_text(encoding="utf-8"))
    sug  = json.loads(SUGG.read_text(encoding="utf-8"))

    meta["policy_block_weekdays"]  = sug.get("suggest_block_weekdays", [])
    meta["policy_block_hours_utc"] = sug.get("suggest_block_hours_utc", [])
    meta.setdefault("policy_block_months", [])  # dejar vacío por ahora

    # backup
    backup = META.with_suffix(".backup.json")
    backup.write_text(META.read_text(encoding="utf-8"), encoding="utf-8")

    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("✔ meta actualizado con bloqueos sugeridos.")
    print(f"🗂 backup: {backup}")

if __name__ == "__main__":
    main()
