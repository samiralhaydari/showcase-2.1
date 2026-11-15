from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_lstm(lookback=90, n_features=9, forecast_horizon=63):
    model = keras.Sequential(
        [
            keras.Input(shape=(lookback, n_features)),
            layers.Conv1D(32, 3, padding="causal", activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.LSTM(32),
            layers.Dropout(0.20),
            layers.Dense(32, activation="relu"),
            layers.Dense(forecast_horizon, name="future_log_returns"),
        ],
        name="MYR_SupplyLens_CNN_LSTM_V2_1",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.Huber(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model
