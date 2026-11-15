import time

import requests
import pandas as pd

from .config import DATA_URL, DATA_RAW_DIR, PRIORITY_CURRENCIES


def download_raw_data(force_download=False, max_age_hours=12):
    """Return the local copy of the BNM daily rates, refreshing it when stale.

    The file used to be downloaded once and then reused forever, so the app kept
    showing the rates from the day it was first run. It is now refreshed when it
    is older than ``max_age_hours``. If the refresh fails and an older copy
    exists, the older copy is used rather than breaking the app.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_RAW_DIR / "daily_1200.csv"
    stale = (
        path.exists()
        and max_age_hours is not None
        and (time.time() - path.stat().st_mtime) > max_age_hours * 3600
    )
    if force_download or stale or not path.exists():
        try:
            response = requests.get(DATA_URL, timeout=90)
            response.raise_for_status()
            if b"date" not in response.content[:2000].lower():
                raise ValueError("Downloaded file does not look like the rates CSV.")
            temporary = path.with_suffix(".csv.part")
            temporary.write_bytes(response.content)
            temporary.replace(path)
        except Exception:
            if not path.exists():
                raise
    return path


def load_exchange_rates(force_download=False):
    # Returns daily middle rates as MYR per one unit of foreign currency.
    path = download_raw_data(force_download)
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    if "date" not in raw.columns:
        raise ValueError(f"Expected date column. Found: {list(raw.columns)}")

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"])

    currency_columns = [
        c for c in raw.columns if len(c) == 3 and c.isalpha()
    ]
    if not currency_columns:
        currency_columns = [c for c in PRIORITY_CURRENCIES if c in raw.columns]
    if not currency_columns:
        raise ValueError("No three-letter currency columns found.")

    if "rate_type" in raw.columns:
        rate_type = raw["rate_type"].astype(str).str.strip().str.lower()
        middle = raw.loc[rate_type.eq("middle")].copy()
        if not middle.empty:
            raw = middle
        else:
            bs = raw.loc[rate_type.isin(["buying", "selling"])].copy()
            if bs.empty:
                raise ValueError("No middle or buying/selling rate rows found.")
            raw = bs.groupby("date", as_index=False)[currency_columns].mean()

    for col in currency_columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    rates = (
        raw[["date"] + currency_columns]
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")
        .dropna(how="all")
    )
    rates = rates.loc[:, rates.notna().any()]
    rates.columns = [str(c).upper() for c in rates.columns]
    return rates


def get_currency_series(rates, currency):
    currency = currency.upper()
    if currency not in rates.columns:
        raise ValueError(
            f"{currency} is unavailable. Available: {', '.join(rates.columns)}"
        )
    series = pd.to_numeric(rates[currency], errors="coerce").dropna()
    series = series[series > 0].sort_index()
    series.name = currency
    return series


def available_priority_currencies(rates):
    return [c for c in PRIORITY_CURRENCIES if c in rates.columns]
