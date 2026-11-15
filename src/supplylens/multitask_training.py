"""Joint multi-currency CNN-LSTM training for MYR SupplyLens V2.2."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

from .config import (
    MODEL_VERSION, MODEL_DIR, SCALER_DIR, METADATA_DIR, DISPLAY_HORIZONS,
    WALK_FORWARD_SPACING, MIN_SELECTION_ORIGINS, CALIBRATION_ORIGIN_STRIDE,
    CALIBRATION_MIN_SAMPLES, SPLIT_FRACTIONS, LOOKBACK, FORECAST_HORIZON,
    PRIORITY_CURRENCIES,
)
from .data import load_exchange_rates
from .multivariate import make_multivariate_supervised_samples
from .evaluation import (
    evaluate_walk_forward,
    select_benchmark_methods,
    calibrate_method_forecast_intervals,
)


def build_joint_cnn_lstm(lookback, n_input_features, output_size):
    """A LiteModel-compatible Sequential CNN-LSTM with one shared multi-output head."""
    return keras.Sequential(
        [
            keras.Input(shape=(lookback, n_input_features)),
            keras.layers.Conv1D(64, 3, padding="causal", activation="relu"),
            keras.layers.Conv1D(32, 3, padding="causal", activation="relu"),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.LSTM(64, activation="tanh", recurrent_activation="sigmoid", use_bias=True, dropout=0.0, recurrent_dropout=0.0, unroll=False),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(output_size, name="joint_future_log_returns"),
        ],
        name="MYR_SupplyLens_Joint_CNN_LSTM_V2_2",
    )


def _scale_X(X, scaler):
    shape = X.shape
    transformed = scaler.transform(X.reshape(-1, shape[-1]))
    return transformed.reshape(shape).astype(np.float32)


def _scale_joint_y(part_y, scalers_y, currencies, horizon):
    scaled = np.empty_like(part_y, dtype=np.float32)
    for index, currency in enumerate(currencies):
        start = index * horizon
        stop = start + horizon
        block = part_y[:, start:stop]
        scaled[:, start:stop] = scalers_y[currency].transform(
            block.reshape(-1, 1)
        ).reshape(block.shape)
    return scaled


class TargetOutputAdapter:
    """Expose one currency's 63-output head to the existing evaluation engine."""
    def __init__(self, model, target_index, horizon):
        self.model = model
        self.target_index = int(target_index)
        self.horizon = int(horizon)

    def predict(self, X, verbose=0):
        prediction = np.asarray(self.model.predict(X, verbose=verbose), dtype=np.float64)
        if prediction.ndim != 2:
            raise ValueError(f"Joint model must return a 2D matrix; got {prediction.shape}.")
        start = self.target_index * self.horizon
        stop = start + self.horizon
        if prediction.shape[1] < stop:
            raise ValueError(
                f"Joint model output has {prediction.shape[1]} columns; expected at least {stop}."
            )
        return prediction[:, start:stop]


def _paths(currency):
    return {
        "model": MODEL_DIR / f"cnn_lstm_{MODEL_VERSION}_{currency}.keras",
        "scaler_X": SCALER_DIR / f"scaler_X_{MODEL_VERSION}_{currency}.joblib",
        "scaler_y": SCALER_DIR / f"scaler_y_{MODEL_VERSION}_{currency}.joblib",
        "metadata": METADATA_DIR / f"metadata_{MODEL_VERSION}_{currency}.json",
        "metrics": METADATA_DIR / f"metrics_{MODEL_VERSION}_{currency}.csv",
        "details": METADATA_DIR / f"walk_forward_details_{MODEL_VERSION}_{currency}.csv",
        "calibration": METADATA_DIR / f"calibration_{MODEL_VERSION}_{currency}.csv",
        "selection_metrics": METADATA_DIR / f"selection_metrics_{MODEL_VERSION}_{currency}.csv",
        "selection_details": METADATA_DIR / f"selection_details_{MODEL_VERSION}_{currency}.csv",
        "selection_policy": METADATA_DIR / f"selection_policy_{MODEL_VERSION}_{currency}.csv",
    }


def _atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload, path):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _matched_selection_metrics(details, currency, horizons, min_origins, include_arima):
    """Recompute selection scores on identical origin sets for every compared method."""
    methods = [
        "Random walk", "CNN-LSTM",
        "Historical drift (252)", "Recent drift (20)",
    ]
    if include_arima:
        methods.append("ARIMA(1,1,1)")

    rows = []
    for horizon_value in horizons:
        group = details[details["horizon"] == int(horizon_value)]
        if group.empty:
            raise ValueError(f"No selection details for {currency}, horizon {horizon_value}.")

        pivot = group.pivot_table(
            index="origin_index", columns="model",
            values="absolute_error", aggfunc="first",
        )
        available = [name for name in methods if name in pivot.columns]
        if "Random walk" not in available:
            raise ValueError(f"Random-walk selection baseline is missing for {currency}.")

        common = pivot[available].dropna()
        enough = len(common) >= int(min_origins)

        for method in available:
            if enough:
                errors = common[method].to_numpy(dtype=np.float64)
                n_origins = int(len(errors))
            else:
                # Force the selector's explicit Random-walk fallback when the
                # matched sample is too small; do not compare models on different dates.
                errors = group.loc[group["model"] == method, "absolute_error"].to_numpy(dtype=np.float64)
                n_origins = 0

            mae = float(np.mean(errors)) if len(errors) else np.nan
            rmse = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else np.nan
            error_std = float(np.std(errors, ddof=1)) if len(errors) > 1 else np.nan
            rows.append({
                "currency": currency,
                "horizon": int(horizon_value),
                "model": method,
                "n_origins": n_origins,
                "mae": mae,
                "rmse": rmse,
                "absolute_error_std": error_std,
                "matched_origin_count": int(len(common)),
                "selection_comparison": "common matched origins" if enough else "random-walk fallback",
            })

    return pd.DataFrame(rows)



def _resolve_training_device(device="auto", require_gpu=False):
    """Choose a real TensorFlow accelerator without hiding available GPUs."""
    physical_gpus = tf.config.list_physical_devices("GPU")

    # Enable memory growth before the first TensorFlow op on each GPU.
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Runtime may already be initialized; the GPU remains usable.
            pass

    requested = "auto" if device is None else str(device).strip().lower()
    if requested in {"auto", "default"}:
        selected = "/GPU:0" if physical_gpus else "/CPU:0"
    elif requested in {"gpu", "/gpu:0", "gpu:0"}:
        if not physical_gpus:
            raise RuntimeError(
                "GPU training was requested, but TensorFlow cannot see a GPU. "
                "In Colab choose Runtime > Change runtime type > T4 GPU (or another GPU), "
                "then reconnect and rerun the notebook from the beginning."
            )
        selected = "/GPU:0"
    elif requested in {"cpu", "/cpu:0", "cpu:0"}:
        if require_gpu:
            raise RuntimeError("require_gpu=True conflicts with device='CPU'.")
        selected = "/CPU:0"
    else:
        raise ValueError("device must be 'auto', 'GPU', or 'CPU'.")

    if require_gpu and not physical_gpus:
        raise RuntimeError(
            "GPU_REQUIRED=True, but no GPU is visible to TensorFlow. "
            "Select a GPU runtime in Colab and rerun the notebook."
        )

    gpu_name = None
    if selected.startswith("/GPU") and physical_gpus:
        try:
            details = tf.config.experimental.get_device_details(physical_gpus[0])
            gpu_name = details.get("device_name") or physical_gpus[0].name
        except Exception:
            gpu_name = physical_gpus[0].name

    return selected, gpu_name


def train_joint_multicurrency_model(
    rates=None,
    currencies=None,
    epochs=1000,
    batch_size=128,
    lookback=LOOKBACK,
    horizon=FORECAST_HORIZON,
    force_download=False,
    include_arima=True,
    calibration_stride=CALIBRATION_ORIGIN_STRIDE,
    patience=70,
    seed=42,
    verbose=1,
    device="auto",
    require_gpu=False,
):
    """Train one shared network to forecast all currencies simultaneously.

    Every currency contributes nine input channels and one 63-return target head.
    The trained joint model is stored in each currency's existing artifact slot;
    metadata tells inference which output slice to use.
    TensorFlow automatically uses the selected GPU when available; CPU is an explicit fallback.
    """
    currencies = [str(c).upper() for c in (currencies or PRIORITY_CURRENCIES)]
    if len(currencies) < 2 or len(set(currencies)) != len(currencies):
        raise ValueError("Provide at least two unique currencies in a fixed order.")

    if rates is None:
        rates = load_exchange_rates(force_download=force_download)
    missing = [c for c in currencies if c not in rates.columns]
    if missing:
        raise ValueError(f"Dataset is missing requested currencies: {missing}")

    data = make_multivariate_supervised_samples(
        rates, currencies, lookback=lookback, horizon=horizon
    )
    partitions = {
        name: data[name]
        for name in ["train", "validation", "selection", "calibration", "test"]
    }
    for name, part in partitions.items():
        if len(part["X"]) == 0:
            raise ValueError(f"{name} partition is empty.")

    n_currencies = len(currencies)
    n_features = len(data["feature_columns"])
    output_size = n_currencies * int(horizon)

    scaler_X = StandardScaler()
    scaler_X.fit(
        partitions["train"]["X"].reshape(-1, n_features)
    )
    X_scaled = {
        name: _scale_X(part["X"], scaler_X)
        for name, part in partitions.items()
    }

    scalers_y = {}
    for index, currency in enumerate(currencies):
        start = index * int(horizon)
        stop = start + int(horizon)
        target_train = partitions["train"]["y"][:, start:stop].reshape(-1, 1)
        scalers_y[currency] = StandardScaler().fit(target_train)

    y_scaled = {
        name: _scale_joint_y(
            part["y"], scalers_y, currencies, int(horizon)
        )
        for name, part in partitions.items()
    }

    training_device, gpu_name = _resolve_training_device(
        device=device, require_gpu=bool(require_gpu)
    )
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    with tf.device(training_device):
        model = build_joint_cnn_lstm(
            int(lookback), n_features, output_size
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss=keras.losses.Huber(),
            metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
        )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(patience) if patience else 0,
            restore_best_weights=True,
            min_delta=1e-6,
        ) if patience else keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(5, int(patience // 3) if patience else 15),
            min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]

    # The model variables and training ops are placed on the selected device.
    # Input arrays are prepared on CPU and transferred batch-by-batch by Keras.
    with tf.device(training_device):
        history = model.fit(
            X_scaled["train"],
            y_scaled["train"],
            validation_data=(X_scaled["validation"], y_scaled["validation"]),
            epochs=int(epochs),
            batch_size=int(batch_size),
            callbacks=callbacks,
            shuffle=False,
            verbose=verbose,
        )

    reports = {}
    for target_index, currency in enumerate(currencies):
        adapter = TargetOutputAdapter(model, target_index, int(horizon))
        target_scaler = scalers_y[currency]
        aligned_target_rates = data["aligned_rates"][:, target_index]

        _raw_selection_metrics, selection_details = evaluate_walk_forward(
            adapter,
            X_scaled["selection"],
            target_scaler,
            partitions["selection"]["origins"],
            aligned_target_rates,
            data["aligned_dates"],
            currency=currency,
            horizons=DISPLAY_HORIZONS,
            spacing=WALK_FORWARD_SPACING,
            spacing_by_horizon={h: h for h in DISPLAY_HORIZONS},
            include_arima=bool(include_arima),
        )
        selection_metrics = _matched_selection_metrics(
            selection_details,
            currency,
            DISPLAY_HORIZONS,
            MIN_SELECTION_ORIGINS,
            bool(include_arima),
        )
        selection_policy = select_benchmark_methods(
            selection_metrics, min_origins=MIN_SELECTION_ORIGINS
        )

        calibration_table = calibrate_method_forecast_intervals(
            adapter,
            X_scaled["calibration"],
            target_scaler,
            partitions["calibration"]["origins"],
            aligned_target_rates,
            horizon=int(horizon),
            include_arima=bool(include_arima),
            origin_stride=int(calibration_stride),
            min_samples=CALIBRATION_MIN_SAMPLES,
        )

        metrics, details = evaluate_walk_forward(
            adapter,
            X_scaled["test"],
            target_scaler,
            partitions["test"]["origins"],
            aligned_target_rates,
            data["aligned_dates"],
            currency=currency,
            horizons=DISPLAY_HORIZONS,
            spacing=WALK_FORWARD_SPACING,
            include_arima=bool(include_arima),
        )

        reports[currency] = {
            "metrics": metrics,
            "details": details,
            "selection_metrics": selection_metrics,
            "selection_details": selection_details,
            "selection_policy": selection_policy,
            "calibration": calibration_table,
            "target_index": target_index,
        }

    for folder in [MODEL_DIR, SCALER_DIR, METADATA_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    shared_model_path = MODEL_DIR / f"cnn_lstm_{MODEL_VERSION}_shared.keras"
    temporary_model_path = MODEL_DIR / f"cnn_lstm_{MODEL_VERSION}_shared.part.keras"
    model.save(temporary_model_path)
    temporary_model_path.replace(shared_model_path)

    shared_scaler_path = SCALER_DIR / f"scaler_X_{MODEL_VERSION}_shared.joblib"
    temporary_scaler_path = SCALER_DIR / f"scaler_X_{MODEL_VERSION}_shared.part.joblib"
    joblib.dump(scaler_X, temporary_scaler_path)
    temporary_scaler_path.replace(shared_scaler_path)

    for currency in currencies:
        target_index = reports[currency]["target_index"]
        report = reports[currency]
        paths = _paths(currency)

        # The same shared network and input scaler are copied into the established
        # per-currency bundle names so existing bundle discovery remains compatible.
        shutil.copy2(shared_model_path, paths["model"])
        shutil.copy2(shared_scaler_path, paths["scaler_X"])
        joblib.dump(scalers_y[currency], paths["scaler_y"])

        _atomic_csv(report["metrics"], paths["metrics"])
        _atomic_csv(report["details"], paths["details"])
        _atomic_csv(report["calibration"], paths["calibration"])
        _atomic_csv(report["selection_metrics"], paths["selection_metrics"])
        _atomic_csv(report["selection_details"], paths["selection_details"])
        _atomic_csv(report["selection_policy"], paths["selection_policy"])

        cnn_rows = report["metrics"][report["metrics"]["model"] == "CNN-LSTM"]
        beats = int((cnn_rows["mae_skill_vs_random_walk_pct"] > 0).sum())
        best_val = float(min(history.history["val_loss"]))
        latest_common_date = pd.Timestamp(data["aligned_dates"][-1]).date().isoformat()
        metadata = {
            "model_version": MODEL_VERSION,
            "model_type": "joint_multicurrency",
            "currency": currency,
            "target_index": int(target_index),
            "target_currencies": currencies,
            "input_currencies": currencies,
            "rate_convention": f"MYR per 1 {currency}",
            "source": "Bank Negara Malaysia via data.gov.my",
            "source_url": "https://data.gov.my/data-catalogue/exchangerates_daily_1200",
            "rate_type": "middle",
            "lookback": int(lookback),
            "forecast_horizon": int(horizon),
            "n_features_per_currency": int(data["n_features_per_currency"]),
            "n_input_features": int(n_features),
            "feature_columns": data["feature_columns"],
            "input_feature_columns": data["feature_columns"],
            "output_layout": "currency-major; each currency owns one consecutive forecast_horizon block",
            "architecture": (
                f"Conv1D(64,kernel=3,causal) -> Conv1D(32,kernel=3,causal) -> "
                f"MaxPooling1D(2) -> LSTM(64) -> Dropout(0.25) -> "
                f"Dense(128,ReLU) -> Dense({output_size})"
            ),
            "training_strategy": (
                "One shared multi-output CNN-LSTM jointly trained on all target currencies. "
                "Each currency's input contains nine historical features for every currency "
                "in target_currencies; all currency outputs are optimized in the same training step."
            ),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_device": training_device,
            "accelerator": "GPU" if training_device.startswith("/GPU") else "CPU",
            "gpu_name": gpu_name,
            "latest_observation_date": latest_common_date,
            "latest_rate": float(data["aligned_rates"][-1, target_index]),
            "common_observations": int(len(data["aligned_dates"])),
            "train_samples": int(len(partitions["train"]["X"])),
            "validation_samples": int(len(partitions["validation"]["X"])),
            "selection_samples": int(len(partitions["selection"]["X"])),
            "calibration_samples": int(len(partitions["calibration"]["X"])),
            "test_candidate_samples": int(len(partitions["test"]["X"])),
            "walk_forward_origins": int(report["details"]["origin_index"].nunique()),
            "walk_forward_spacing": int(WALK_FORWARD_SPACING),
            "split_fractions": dict(SPLIT_FRACTIONS),
            "selection_method": (
                "Per-currency, per-horizon selection on a separate chronological selection "
                "partition using common matched forecast origins. Lowest selection-partition MAE wins; RMSE and absolute-error "
                "standard deviation break ties. Final test data is not used for selection."
            ),
            "selection_min_origins": int(MIN_SELECTION_ORIGINS),
            "selection_policy": report["selection_policy"][
                ["horizon", "selected_model", "n_selection_origins"]
            ].to_dict("records"),
            "benchmarks": [
                "Random walk", "Historical drift (252)", "Recent drift (20)",
                "ARIMA(1,1,1)" if include_arima else "ARIMA disabled",
            ],
            "cnn_lstm_horizons_beating_random_walk": beats,
            "cnn_lstm_horizons_evaluated": int(len(cnn_rows)),
            "best_validation_loss": best_val,
            "epochs_completed": int(len(history.history["loss"])),
            "max_epochs_requested": int(epochs),
            "calibration_method": (
                "Method-specific empirical absolute log-error quantiles from a separate "
                "chronological calibration partition. Calibration bands are descriptive "
                "scenario ranges, not formal coverage guarantees."
            ),
            "calibration_origin_stride": int(calibration_stride),
            "calibration_min_samples": int(CALIBRATION_MIN_SAMPLES),
            "arima_included": bool(include_arima),
            "metric_definitions": {
                "mae_skill_vs_random_walk_pct": "100 * (1 - model_MAE / random_walk_MAE)",
                "rmse_skill_vs_random_walk_pct": "100 * (1 - model_RMSE / random_walk_RMSE)",
                "directional_accuracy_pct": "Not applicable to random walk because it predicts no directional change.",
            },
        }
        _atomic_json(metadata, paths["metadata"])

    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    _atomic_csv(
        history_frame,
        METADATA_DIR / f"training_history_{MODEL_VERSION}.csv"
    )

    return {
        "model_path": str(shared_model_path),
        "currencies": currencies,
        "metadata": {
            "model_version": MODEL_VERSION,
            "training_device": training_device,
            "accelerator": "GPU" if training_device.startswith("/GPU") else "CPU",
            "gpu_name": gpu_name,
            "epochs_completed": int(len(history.history["loss"])),
            "best_validation_loss": float(min(history.history["val_loss"])),
            "common_observations": int(len(data["aligned_dates"])),
            "train_samples": int(len(partitions["train"]["X"])),
        },
        "reports": reports,
        "history": history.history,
        "data": data,
        "model": model,
    }
