import numpy as np
import pandas as pd

from src.supplylens.evaluation import evaluate_walk_forward, select_benchmark_methods


class DummyScaler:
    def inverse_transform(self, values):
        return values


class DummyModel:
    def __init__(self, daily_return=0.0002):
        self.daily_return = daily_return

    def predict(self, X, verbose=0):
        return np.full((len(X), 63), self.daily_return, dtype=float)


def _sample_data(n=1800, origin_count=500):
    rates = 4.0 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.001, n)))
    dates = pd.bdate_range("2015-01-01", periods=n)
    origins = np.arange(1000, 1000 + origin_count)
    X = np.zeros((len(origins), 90, 9), dtype=float)
    return rates, dates, origins, X


def test_walk_forward_uses_non_overlapping_test_origins_and_separate_skills():
    rates, dates, origins, X = _sample_data()
    metrics, details = evaluate_walk_forward(
        DummyModel(), X, DummyScaler(), origins, rates, dates,
        currency="USD", include_arima=False,
    )
    assert set(metrics["model"]) >= {
        "CNN-LSTM", "Random walk", "Historical drift (252)", "Recent drift (20)"
    }
    assert details["origin_index"].nunique() <= 8
    assert metrics["n_origins"].nunique() == 1
    assert "mae_skill_vs_random_walk_pct" in metrics.columns
    assert "rmse_skill_vs_random_walk_pct" in metrics.columns
    assert np.allclose(
        metrics["skill_vs_random_walk_pct"],
        metrics["mae_skill_vs_random_walk_pct"],
        equal_nan=True,
    )
    rw = metrics[metrics["model"] == "Random walk"]
    assert np.allclose(rw["mae_skill_vs_random_walk_pct"], 0.0)
    assert np.allclose(rw["rmse_skill_vs_random_walk_pct"], 0.0)
    assert rw["directional_accuracy_pct"].isna().all()
    assert rw["directional_status"].str.contains("no directional change").all()


def test_selection_evaluation_uses_horizon_specific_non_overlapping_origins():
    rates, dates, origins, X = _sample_data(origin_count=250)
    metrics, details = evaluate_walk_forward(
        DummyModel(), X, DummyScaler(), origins, rates, dates,
        currency="USD", horizons=[5, 21], spacing=63,
        spacing_by_horizon={5: 5, 21: 21}, include_arima=False,
    )
    counts = metrics[metrics["model"] == "Random walk"].set_index("horizon")["n_origins"]
    assert counts.loc[5] > counts.loc[21]
    for horizon, group in details.groupby("horizon"):
        unique_origins = np.sort(group["origin_index"].unique())
        assert np.all(np.diff(unique_origins) >= horizon)


def test_selector_uses_lowest_validation_mae_and_random_walk_fallback():
    validation_metrics = pd.DataFrame([
        {"currency": "USD", "horizon": 5, "model": "Random walk", "n_origins": 12,
         "mae": 0.10, "rmse": 0.15, "absolute_error_std": 0.02},
        {"currency": "USD", "horizon": 5, "model": "CNN-LSTM", "n_origins": 12,
         "mae": 0.08, "rmse": 0.16, "absolute_error_std": 0.03},
        {"currency": "USD", "horizon": 21, "model": "Random walk", "n_origins": 4,
         "mae": 0.10, "rmse": 0.15, "absolute_error_std": 0.02},
        {"currency": "USD", "horizon": 21, "model": "CNN-LSTM", "n_origins": 4,
         "mae": 0.08, "rmse": 0.16, "absolute_error_std": 0.03},
    ])
    policy = select_benchmark_methods(validation_metrics, min_origins=8)
    h5 = policy[policy["horizon"] == 5].iloc[0]
    h21 = policy[policy["horizon"] == 21].iloc[0]
    assert h5["selected_model"] == "CNN-LSTM"
    assert not bool(h5["selection_fallback"])
    assert h21["selected_model"] == "Random walk"
    assert bool(h21["selection_fallback"])


def test_skill_columns_are_not_conflated():
    rates, dates, origins, X = _sample_data(origin_count=250)
    metrics, _ = evaluate_walk_forward(
        DummyModel(daily_return=0.001), X, DummyScaler(), origins, rates, dates,
        currency="IDR", horizons=[5, 21], include_arima=False,
    )
    cnn = metrics[metrics["model"] == "CNN-LSTM"]
    # The two metrics are independently calculated, not aliases of each other.
    assert np.allclose(cnn["mae_skill_vs_random_walk_pct"], cnn["skill_vs_random_walk_pct"])
    assert np.allclose(cnn["mae_skill_ci_lower_pct"], cnn["skill_ci_lower_pct"])
    assert np.allclose(cnn["mae_skill_ci_upper_pct"], cnn["skill_ci_upper_pct"])
    assert np.isfinite(cnn["rmse_skill_vs_random_walk_pct"]).all()


def test_method_specific_calibration_has_model_and_horizon_rows():
    from src.supplylens.evaluation import calibrate_method_forecast_intervals

    rng = np.random.default_rng(11)
    n = 700
    rates = 4.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    origins = np.arange(150, 350)
    X = np.zeros((len(origins), 90, 9), dtype=float)
    table = calibrate_method_forecast_intervals(
        DummyModel(), X, DummyScaler(), origins, rates,
        horizon=10, include_arima=False, origin_stride=2, min_samples=30,
    )
    assert {"CNN-LSTM", "Random walk", "Historical drift (252)", "Recent drift (20)"}.issubset(set(table["model"]))
    assert set(table["horizon"]) == set(range(1, 11))
    assert table["interval_available"].all()
