import numpy as np
import pandas as pd

from src.supplylens import forecasting
from src.supplylens.multivariate import make_multivariate_feature_frame


CURRENCIES = ["USD", "CNY", "IDR", "SGD", "THB", "JPY", "KRW"]


class IdentityScaler:
    def transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


class JointModel:
    def __init__(self, target_index=2, extreme=False):
        self.target_index = target_index
        self.extreme = extreme

    def predict(self, X, verbose=0):
        result = np.zeros((len(X), len(CURRENCIES) * 63), dtype=np.float32)
        if self.extreme:
            start = self.target_index * 63
            result[:, start:start + 63] = 20.0
        return result


def _panel():
    dates = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(12)
    return pd.DataFrame({
        code: (0.5 + i) * np.exp(np.cumsum(rng.normal(0, 0.001, len(dates))))
        for i, code in enumerate(CURRENCIES)
    }, index=dates)


def _bundle(panel, extreme=False):
    columns = make_multivariate_feature_frame(panel, CURRENCIES).columns.tolist()
    metadata = {
        "model_type": "joint_multicurrency",
        "model_version": "v2_2",
        "input_currencies": CURRENCIES,
        "target_index": 2,
        "lookback": 90,
        "forecast_horizon": 63,
        "input_feature_columns": columns,
        "arima_included": False,
    }
    calibration = pd.DataFrame([
        {"model": method, "horizon": h, "q80_abs_log_error": 0.01, "q95_abs_log_error": 0.02}
        for method in ["CNN-LSTM", "Random walk", "Historical drift (252)", "Recent drift (20)"]
        for h in range(1, 64)
    ])
    selection = pd.DataFrame([
        {"horizon": h, "selected_model": "CNN-LSTM", "selection_reason": "test"}
        for h in [5, 21, 42, 63]
    ])
    return (
        JointModel(extreme=extreme), IdentityScaler(), IdentityScaler(),
        metadata, calibration, selection,
    )


def test_joint_forecast_uses_panel_and_returns_valid_selected_rates(monkeypatch, tmp_path):
    panel = _panel()
    monkeypatch.setattr(forecasting, "load_model_bundle", lambda currency: _bundle(panel))
    monkeypatch.setattr(forecasting, "FORECAST_DIR", tmp_path)
    result, metadata = forecasting.forecast_currency(panel, "IDR")
    assert metadata["model_type"] == "joint_multicurrency"
    assert result["forecast_cnn_lstm"].notna().all()
    selected = result.loc[[4, 20, 41, 62], "selected_forecast_rate"]
    assert np.isfinite(selected.to_numpy()).all()
    assert (selected > 0).all()


def test_rejected_joint_cnn_path_is_not_reinserted(monkeypatch, tmp_path):
    panel = _panel()
    monkeypatch.setattr(forecasting, "load_model_bundle", lambda currency: _bundle(panel, extreme=True))
    monkeypatch.setattr(forecasting, "FORECAST_DIR", tmp_path)
    result, _ = forecasting.forecast_currency(panel, "IDR")
    assert result["forecast_cnn_lstm"].isna().all()
    selected = result.loc[[4, 20, 41, 62]]
    assert selected["selected_model"].eq("Random walk").all()
    assert np.isfinite(selected["selected_forecast_rate"].to_numpy()).all()
