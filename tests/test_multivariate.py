import numpy as np
import pandas as pd

from src.supplylens.multivariate import (
    make_multivariate_feature_frame,
    make_multivariate_supervised_samples,
)


CURRENCIES = ["USD", "CNY", "IDR", "SGD", "THB", "JPY", "KRW"]


def _panel(n=1400):
    rng = np.random.default_rng(22)
    dates = pd.bdate_range("2018-01-01", periods=n)
    common = rng.normal(0, 0.0005, n)
    panel = {}
    for i, code in enumerate(CURRENCIES):
        specific = rng.normal(0, 0.0004 + i * 0.00003, n)
        returns = common * (0.4 + i * 0.05) + specific
        panel[code] = (1.0 + i) * np.exp(np.cumsum(returns))
    return pd.DataFrame(panel, index=dates)


def test_multivariate_features_have_currency_major_order():
    panel = _panel()
    frame = make_multivariate_feature_frame(panel, CURRENCIES)
    assert frame.shape[1] == 7 * 9
    assert frame.columns[0] == "USD__log_rate"
    assert frame.columns[8] == "USD__range_20"
    assert frame.columns[9] == "CNY__log_rate"
    assert frame.columns[-1] == "KRW__range_20"


def test_joint_samples_have_seven_output_blocks_and_chronological_splits():
    panel = _panel()
    data = make_multivariate_supervised_samples(panel, CURRENCIES, lookback=90, horizon=63)
    assert data["train"]["X"].shape[1:] == (90, 63)
    assert data["train"]["y"].shape[1] == 7 * 63
    split_names = ["train", "validation", "selection", "calibration", "test"]
    assert all(len(data[name]["X"]) > 0 for name in split_names)
    for left, right in zip(split_names[:-1], split_names[1:]):
        assert data[left]["origins"].max() < data[right]["origins"].min()
    assert data["aligned_rates"].shape[1] == 7
