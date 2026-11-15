import numpy as np
import pandas as pd

from src.supplylens import forecasting


class IdentityScaler:
    def transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


class ZeroReturnModel:
    def predict(self, X, verbose=0):
        return np.zeros((len(X), 63), dtype=float)


def test_forecast_uses_validation_selected_method_at_checkpoints(monkeypatch, tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=400)
    values = 4.0 * np.exp(np.cumsum(np.random.default_rng(4).normal(0, 0.001, len(dates))))
    series = pd.Series(values, index=dates, name="USD")

    selection = pd.DataFrame([
        {"currency": "USD", "horizon": h, "selected_model": "Random walk",
         "selection_reason": "Test selection policy"}
        for h in [5, 21, 42, 63]
    ])
    calibration = pd.DataFrame([
        {"model": "Random walk", "horizon": h, "q80_abs_log_error": 0.01,
         "q95_abs_log_error": 0.02}
        for h in range(1, 64)
    ])
    metadata = {"model_version": "v2_1", "lookback": 90, "forecast_horizon": 63,
                "feature_columns": ["log_rate", "return_1d", "momentum_5", "momentum_10",
                                    "momentum_20", "volatility_5", "volatility_20",
                                    "volatility_60", "range_20"],
                "arima_included": False}
    monkeypatch.setattr(
        forecasting, "load_model_bundle",
        lambda currency: (ZeroReturnModel(), IdentityScaler(), IdentityScaler(), metadata, calibration, selection),
    )
    monkeypatch.setattr(forecasting, "FORECAST_DIR", tmp_path)

    result, returned_metadata = forecasting.forecast_currency(series, "USD")
    row5 = result.iloc[4]
    assert row5["selected_model"] == "Random walk"
    assert np.isclose(row5["selected_forecast_rate"], series.iloc[-1])
    assert np.isclose(row5["selected_lower_80"], series.iloc[-1] * np.exp(-0.01))
    assert returned_metadata["selection_mode"].startswith("selection-partition")
    assert (tmp_path / "USD_forecast_v2_1.csv").exists()
