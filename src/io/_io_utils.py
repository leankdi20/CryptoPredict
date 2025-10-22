from pathlib import Path
import pandas as pd

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def save_incremental_csv(
    df_new: pd.DataFrame,
    path: Path,
    key: str = "timestamp",
    parse_date_keys: list[str] | None = None,
):
    """
    Concatena old+new, deduplica por `key`, asegura tipos consistentes y ordena por `key`.
    - `parse_date_keys`: columnas a parsear como datetime al leer el CSV existente.
    """
    ensure_dir(path)

    # Leer existente con parse_dates si corresponde
    if path.exists():
        if parse_date_keys:
            old = pd.read_csv(path, parse_dates=parse_date_keys)
        else:
            old = pd.read_csv(path)
    else:
        old = None

    # Concat
    if old is not None and not old.empty:
        comb = pd.concat([old, df_new], ignore_index=True)
    else:
        comb = df_new.copy()

    # Forzar datetime en keys si se indicó
    if parse_date_keys and (key in parse_date_keys):
        comb[key] = pd.to_datetime(comb[key], utc=True, errors="coerce")

    # Dedup + sort
    comb = (
        comb.drop_duplicates(subset=[key])
            .sort_values(key)
            .reset_index(drop=True)
    )

    comb.to_csv(path, index=False)
    return comb
