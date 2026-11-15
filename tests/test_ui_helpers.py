import numpy as np
import pandas as pd

from src.supplylens import forecasting, ui_helpers


def _clear():
    for cached in (ui_helpers.trained_currencies, ui_helpers._bundle_signature,
                   ui_helpers._cached_bundle, ui_helpers._cached_forecast):
        cached.clear()


def test_trained_currencies_does_not_invent_codes_from_v2_1_files(monkeypatch, tmp_path):
    for name in ["cnn_lstm_v2_1_USD.keras", "cnn_lstm_v2_CNY.keras", "cnn_lstm_IDR.keras",
                 "cnn_lstm_v2_1_CNY.keras", "notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(ui_helpers, "MODEL_DIR", tmp_path)
    _clear()
    assert ui_helpers.trained_currencies() == ["CNY", "IDR", "USD"]
    _clear()


def test_get_forecast_reports_missing_model_instead_of_raising(monkeypatch, tmp_path):
    for name in ("MODEL_DIR", "SCALER_DIR", "METADATA_DIR"):
        monkeypatch.setattr(forecasting, name, tmp_path)
    _clear()
    dates = pd.bdate_range("2024-01-01", periods=120)
    rates = pd.DataFrame({"USD": np.linspace(4.0, 4.1, len(dates))}, index=dates)
    result = ui_helpers.get_forecast(rates, "usd")
    assert result.status == "no_model" and not result.ok
    checkpoint = result.checkpoint(21)
    assert checkpoint["rate"] == rates["USD"].iloc[-1]
    assert checkpoint["low"] is None and checkpoint["has_model"] is False
    _clear()
