import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, LOOKBACK, FORECAST_HORIZON, SPLIT_FRACTIONS


def make_feature_frame(rate_series):
    """Create leakage-safe features using observations available at each date."""
    rates = rate_series.astype(float).sort_index()
    if (rates <= 0).any():
        raise ValueError("Exchange rates must be strictly positive.")

    log_rate = np.log(rates)
    returns = log_rate.diff()
    rolling_high_20 = log_rate.rolling(20, min_periods=20).max()
    rolling_low_20 = log_rate.rolling(20, min_periods=20).min()

    frame = pd.DataFrame(index=rates.index)
    frame["log_rate"] = log_rate
    frame["return_1d"] = returns
    frame["momentum_5"] = log_rate.diff(5)
    frame["momentum_10"] = log_rate.diff(10)
    frame["momentum_20"] = log_rate.diff(20)
    frame["volatility_5"] = returns.rolling(5, min_periods=5).std()
    frame["volatility_20"] = returns.rolling(20, min_periods=20).std()
    frame["volatility_60"] = returns.rolling(60, min_periods=60).std()
    frame["range_20"] = rolling_high_20 - rolling_low_20
    return frame[FEATURE_COLUMNS]


def make_supervised_samples(rate_series, lookback=LOOKBACK, horizon=FORECAST_HORIZON):
    """Create five chronological partitions with target paths contained in each split.

    V2.1 partitions (aligned observations):
      train 60%, early-stop validation 10%, method selection 10%,
      calibration 5%, untouched final test 15%.

    Input windows may look backwards across a split boundary because those
    observations are already known at the forecast origin. Target windows never
    cross a partition boundary.
    """
    rates = rate_series.astype(float).sort_index()
    features = make_feature_frame(rates)
    returns = np.log(rates).diff().rename("target_return")
    aligned = features.join(returns).dropna().sort_index()

    feature_values = aligned[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    target_values = aligned["target_return"].to_numpy(dtype=np.float32)
    rate_values = rates.reindex(aligned.index).to_numpy(dtype=np.float64)
    dates = aligned.index
    n = len(aligned)

    minimum = lookback + horizon + 300
    if n < minimum:
        raise ValueError(
            f"Only {n} usable observations after feature engineering. "
            f"At least {minimum} are recommended for the V2.1 five-way split."
        )

    train_end = int(n * SPLIT_FRACTIONS["train"]) - 1
    validation_end = int(n * (SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["validation"])) - 1
    selection_end = int(n * (
        SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["validation"] + SPLIT_FRACTIONS["selection"]
    )) - 1
    calibration_end = int(n * (
        SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["validation"]
        + SPLIT_FRACTIONS["selection"] + SPLIT_FRACTIONS["calibration"]
    )) - 1

    X, y, origins, labels = [], [], [], []
    for origin in range(lookback - 1, n - horizon):
        target_start = origin + 1
        target_end = origin + horizon

        if target_end <= train_end:
            label = "train"
        elif target_start > train_end and target_end <= validation_end:
            label = "validation"
        elif target_start > validation_end and target_end <= selection_end:
            label = "selection"
        elif target_start > selection_end and target_end <= calibration_end:
            label = "calibration"
        elif target_start > calibration_end and target_end < n:
            label = "test"
        else:
            continue

        X.append(feature_values[origin - lookback + 1:origin + 1])
        y.append(target_values[target_start:target_end + 1])
        origins.append(origin)
        labels.append(label)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    origins = np.asarray(origins, dtype=np.int64)
    labels = np.asarray(labels)

    result = {}
    for label in ["train", "validation", "selection", "calibration", "test"]:
        mask = labels == label
        result[label] = {
            "X": X[mask],
            "y": y[mask],
            "origins": origins[mask],
        }

    result["aligned_rates"] = rate_values
    result["aligned_dates"] = dates
    result["lookback"] = lookback
    result["horizon"] = horizon
    result["split_boundaries"] = {
        "train_end": int(train_end),
        "validation_end": int(validation_end),
        "selection_end": int(selection_end),
        "calibration_end": int(calibration_end),
        "n_aligned_observations": int(n),
        "fractions": dict(SPLIT_FRACTIONS),
    }
    return result
