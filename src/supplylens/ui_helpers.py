"""Data access for the Streamlit pages.

Everything a page needs from the forecasting stack goes through here, and
everything expensive is cached. Before this, every widget interaction re-read
the Keras model from disk, re-ran the network, re-fitted ARIMA and rewrote two
CSV files - once per currency on the page.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from .config import (
    CURRENCY_NAMES, DISPLAY_HORIZONS, LEGACY_MODEL_VERSIONS, MODEL_DIR,
    MODEL_VERSION, PRIORITY_CURRENCIES,
)
from .data import get_currency_series, load_exchange_rates

HORIZON_LABELS = {5: "1 week", 21: "1 month", 42: "2 months", 63: "3 months"}
HORIZON_SHORT = {5: "1W", 21: "1M", 42: "2M", 63: "3M"}


# --------------------------------------------------------------------------
# Rates
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Loading Bank Negara reference rates…")
def cached_rates():
    return load_exchange_rates()


def get_current_rate(rates, currency):
    series = get_currency_series(rates, currency)
    return float(series.iloc[-1]), series.index[-1]


def priority_currencies(rates):
    return [c for c in PRIORITY_CURRENCIES if c in rates.columns]


def currency_label(code):
    if code == "MYR":
        return "MYR — Malaysian Ringgit"
    return f"{code} — {CURRENCY_NAMES.get(code, code)}"


def currency_rate_precision(code):
    return {
        "USD": 4, "CNY": 6, "IDR": 8, "SGD": 4,
        "THB": 5, "JPY": 6, "KRW": 7, "PHP": 6,
        "VND": 7, "BND": 4, "EUR": 4, "GBP": 4,
    }.get(str(code).upper(), 6)


def format_rate(value, code):
    return f"{float(value):.{currency_rate_precision(code)}f}"


def horizon_label(horizon, long=False):
    name = HORIZON_LABELS.get(int(horizon), f"{int(horizon)} sessions")
    return f"{name} ({int(horizon)} trading sessions)" if long else name


# --------------------------------------------------------------------------
# Trained models
# --------------------------------------------------------------------------
@st.cache_data(ttl=120, show_spinner=False)
def trained_currencies():
    """Currencies that have a model file, newest bundle family first.

    The previous version globbed broad version prefixes and could match newer
    version suffixes as currency codes. That is why
    the home page counted four trained currencies when three existed.
    """
    found = set()
    known_prefixes = [f"cnn_lstm_{v}_" for v in [MODEL_VERSION] + list(LEGACY_MODEL_VERSIONS) if v != "v1"]
    known_prefixes.sort(key=len, reverse=True)  # match longer version prefixes before shorter legacy prefixes
    for path in MODEL_DIR.glob("cnn_lstm_*.keras"):
        stem = path.stem
        for prefix in known_prefixes:
            if stem.startswith(prefix):
                code = stem[len(prefix):]
                break
        else:
            code = stem[len("cnn_lstm_"):]  # V1 naming: cnn_lstm_USD.keras
        if len(code) == 3 and code.isalpha():
            found.add(code.upper())
    return sorted(found)


@st.cache_data(ttl=60, show_spinner=False)
def _bundle_signature(currency):
    from .forecasting import bundle_signature
    try:
        return bundle_signature(currency)
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False, max_entries=16)
def _cached_bundle(currency, signature):
    # ``signature`` is only a cache key: retraining changes it and forces a reload.
    from .forecasting import load_model_bundle
    return load_model_bundle(currency)


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_forecast(currency, signature, panel_stamp):
    from .forecasting import forecast_currency
    rates_panel = cached_rates()
    bundle = _cached_bundle(currency, signature)
    return forecast_currency(rates_panel, currency, bundle=bundle)


def _rates_panel_stamp(rates):
    """Cache key covering every currency used by the joint V2.2 model."""
    stamp = []
    for column in rates.columns:
        series = pd.to_numeric(rates[column], errors="coerce").dropna()
        if series.empty:
            stamp.append((str(column), 0, None, None))
        else:
            stamp.append((
                str(column), int(len(series)),
                str(pd.Timestamp(series.index[-1]).date()),
                float(series.iloc[-1]),
            ))
    return tuple(stamp)


# --------------------------------------------------------------------------
# Forecast access used by every page
# --------------------------------------------------------------------------
@dataclass
class ForecastResult:
    currency: str
    status: str                      # "ok" | "no_model" | "error"
    current_rate: float
    latest_date: pd.Timestamp
    forecast: pd.DataFrame = None
    metadata: dict = field(default_factory=dict)
    message: str = ""

    @property
    def ok(self):
        return self.status == "ok"

    @property
    def model_version(self):
        return str(self.metadata.get("model_version", "legacy")).replace("_", ".").upper()

    def checkpoint(self, horizon):
        """Rate to plan with at ``horizon`` sessions, plus its 80% range if known."""
        horizon = int(horizon)
        if not self.ok:
            return {
                "horizon": horizon, "rate": self.current_rate, "method": "Latest reference rate",
                "low": None, "high": None, "date": None, "has_model": False,
                "is_flat": True, "change_pct": 0.0,
            }
        row = self.forecast.iloc[min(horizon, len(self.forecast)) - 1]
        selected = row.get("selected_forecast_rate", np.nan)
        if pd.notna(selected):
            rate = float(selected)
            method = str(row.get("selected_model") or "CNN-LSTM")
        else:
            rate = float(row["forecast_rate"])
            method = "CNN-LSTM"
        low, high = row.get("selected_lower_80", np.nan), row.get("selected_upper_80", np.nan)
        if pd.isna(low) or pd.isna(high):
            # Legacy bundles only calibrate the CNN-LSTM path.
            if method == "CNN-LSTM":
                low, high = row.get("lower_80", np.nan), row.get("upper_80", np.nan)
        has_range = pd.notna(low) and pd.notna(high)
        return {
            "horizon": horizon,
            "rate": rate,
            "method": method,
            "low": float(low) if has_range else None,
            "high": float(high) if has_range else None,
            "date": pd.Timestamp(row["date"]),
            "has_model": True,
            "is_flat": bool(np.isclose(rate, self.current_rate, rtol=1e-9)),
            "change_pct": (rate / self.current_rate - 1) * 100,
        }

    def checkpoints(self):
        return [self.checkpoint(h) for h in DISPLAY_HORIZONS]


def get_forecast(rates, currency):
    """Cached forecast for one currency. Never raises: check ``.status``."""
    currency = currency.upper()
    series = get_currency_series(rates, currency)
    current_rate, latest_date = float(series.iloc[-1]), series.index[-1]
    signature = _bundle_signature(currency)
    if signature is None:
        return ForecastResult(
            currency, "no_model", current_rate, latest_date,
            message=f"No trained model for {currency} yet.",
        )
    try:
        forecast, metadata = _cached_forecast(
            currency, signature, _rates_panel_stamp(rates),
        )
    except FileNotFoundError as exc:
        return ForecastResult(currency, "no_model", current_rate, latest_date, message=str(exc))
    except Exception as exc:  # a broken bundle must not take the whole page down
        return ForecastResult(currency, "error", current_rate, latest_date, message=str(exc))
    return ForecastResult(currency, "ok", current_rate, latest_date, forecast, metadata)


# --------------------------------------------------------------------------
# Backwards-compatible helpers (used by older notebooks / integrations)
# --------------------------------------------------------------------------
def selected_forecast_value(forecast, horizon):
    row = forecast.iloc[min(int(horizon), len(forecast)) - 1]
    selected = row.get("selected_forecast_rate", np.nan)
    if selected is not None and np.isfinite(float(selected)):
        model_name = row.get("selected_model") or "CNN-LSTM"
        return float(selected), str(model_name)
    return float(row["forecast_rate"]), "CNN-LSTM (legacy/default fallback)"


def get_forecast_rate(rates, currency, horizon):
    result = get_forecast(rates, currency)
    if result.status == "no_model":
        raise FileNotFoundError(result.message)
    if result.status == "error":
        raise RuntimeError(result.message)
    value, _ = selected_forecast_value(result.forecast, horizon)
    return value, result.forecast, result.metadata
