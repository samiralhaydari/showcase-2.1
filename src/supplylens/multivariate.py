"""Leakage-safe multivariate features and samples for joint currency modelling."""
import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, LOOKBACK, FORECAST_HORIZON, SPLIT_FRACTIONS
from .features import make_feature_frame


def make_multivariate_feature_frame(rates, currencies):
    """Return one aligned feature frame with nine channels per currency.

    Column order is currency-major and follows ``currencies`` exactly.
    Missing observations are retained as NaN here and removed during alignment.
    """
    if not isinstance(rates, pd.DataFrame):
        raise TypeError("Multivariate inference requires a rates DataFrame.")
    currencies = [str(c).upper() for c in currencies]
    missing = [c for c in currencies if c not in rates.columns]
    if missing:
        raise ValueError(f"Missing cross-currency input columns: {missing}")
    if not currencies:
        raise ValueError("At least one currency is required.")

    panel = rates.loc[:, currencies].copy().sort_index()
    for currency in currencies:
        panel[currency] = pd.to_numeric(panel[currency], errors="coerce")
        valid = panel[currency].dropna()
        if (valid <= 0).any():
            raise ValueError(f"{currency} contains non-positive exchange rates.")

    frames = []
    for currency in currencies:
        frame = make_feature_frame(panel[currency])
        frame.columns = [f"{currency}__{name}" for name in FEATURE_COLUMNS]
        frames.append(frame)

    result = pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan)
    return result


def make_multivariate_supervised_samples(
    rates,
    currencies,
    lookback=LOOKBACK,
    horizon=FORECAST_HORIZON,
):
    """Build joint X windows and currency-major multi-output targets.

    X shape: (samples, lookback, n_currencies * n_features)
    y shape: (samples, n_currencies * horizon)

    Targets are laid out as [currency_0's horizon, currency_1's horizon, ...].
    Five chronological partitions mirror the V2.1 protocol. No target window
    crosses a partition boundary.
    """
    currencies = [str(c).upper() for c in currencies]
    if not isinstance(rates, pd.DataFrame):
        raise TypeError("rates must be a pandas DataFrame.")
    missing = [c for c in currencies if c not in rates.columns]
    if missing:
        raise ValueError(f"Missing currencies from rate panel: {missing}")

    panel = rates.loc[:, currencies].copy().sort_index()
    for currency in currencies:
        panel[currency] = pd.to_numeric(panel[currency], errors="coerce")
        valid = panel[currency].dropna()
        if (valid <= 0).any():
            raise ValueError(f"{currency} contains non-positive exchange rates.")

    feature_frame = make_multivariate_feature_frame(panel, currencies)
    log_rates = np.log(panel)
    target_frame = log_rates.diff()
    target_frame.columns = [f"{c}__target_return" for c in currencies]

    aligned = pd.concat([feature_frame, target_frame], axis=1)
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna().sort_index()

    feature_columns = list(feature_frame.columns)
    target_columns = list(target_frame.columns)
    feature_values = aligned[feature_columns].to_numpy(dtype=np.float32)
    target_values = aligned[target_columns].to_numpy(dtype=np.float32)
    aligned_rates = panel.reindex(aligned.index).to_numpy(dtype=np.float64)
    dates = aligned.index
    n = len(aligned)
    n_currencies = len(currencies)

    minimum = int(lookback) + int(horizon) + 300
    if n < minimum:
        raise ValueError(
            f"Only {n} common usable observations remain for {currencies}. "
            f"At least {minimum} are recommended for the five-way chronological split."
        )

    train_end = int(n * SPLIT_FRACTIONS["train"]) - 1
    validation_end = int(
        n * (SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["validation"])
    ) - 1
    selection_end = int(
        n * (
            SPLIT_FRACTIONS["train"]
            + SPLIT_FRACTIONS["validation"]
            + SPLIT_FRACTIONS["selection"]
        )
    ) - 1
    calibration_end = int(
        n * (
            SPLIT_FRACTIONS["train"]
            + SPLIT_FRACTIONS["validation"]
            + SPLIT_FRACTIONS["selection"]
            + SPLIT_FRACTIONS["calibration"]
        )
    ) - 1

    X, y, origins, labels = [], [], [], []
    for origin in range(int(lookback) - 1, n - int(horizon)):
        target_start = origin + 1
        target_end = origin + int(horizon)

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

        X.append(feature_values[origin - int(lookback) + 1:origin + 1])
        # (horizon, currencies) -> (currencies, horizon) -> currency-major flat vector
        future = target_values[target_start:target_end + 1, :].T.reshape(-1)
        y.append(future)
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

    result["aligned_rates"] = aligned_rates
    result["aligned_dates"] = dates
    result["lookback"] = int(lookback)
    result["horizon"] = int(horizon)
    result["currencies"] = currencies
    result["feature_columns"] = feature_columns
    result["target_columns"] = target_columns
    result["n_features_per_currency"] = len(FEATURE_COLUMNS)
    result["split_boundaries"] = {
        "train_end": int(train_end),
        "validation_end": int(validation_end),
        "selection_end": int(selection_end),
        "calibration_end": int(calibration_end),
        "n_aligned_observations": int(n),
        "fractions": dict(SPLIT_FRACTIONS),
    }
    return result
