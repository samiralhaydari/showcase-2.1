"""Run a trained CNN-LSTM bundle with NumPy only.

The web app only needs a forward pass through a very small network. Importing
TensorFlow for that costs roughly 700 MB of memory and several seconds, which is
more than free hosting tiers comfortably allow. This module reads the weights
straight out of the ``.keras`` file and evaluates the same layers in NumPy.

It is deliberately strict: any layer or setting it does not recognise raises
``UnsupportedModel`` and the caller falls back to Keras. Training still uses
TensorFlow and is unaffected.

Outputs match Keras to floating-point precision (see tests/test_lite_inference.py).
"""
import io
import json
import re
import zipfile

import numpy as np


class UnsupportedModel(Exception):
    """The saved model uses something this evaluator does not implement."""


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _activation(name):
    table = {
        "relu": lambda x: np.maximum(x, 0.0),
        "linear": lambda x: x,
        "tanh": np.tanh,
        "sigmoid": _sigmoid,
    }
    if name not in table:
        raise UnsupportedModel(f"activation {name!r}")
    return table[name]


def _snake(class_name):
    # Mirrors Keras' automatic layer naming: Conv1D -> conv1d, MaxPooling1D -> max_pooling1d.
    name = re.sub(r"(.)([A-Z][a-z0-9]+)", r"\1_\2", class_name)
    return re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower()


def _require(config, **expected):
    for key, value in expected.items():
        actual = config.get(key)
        if isinstance(actual, (list, tuple)) and len(actual) == 1:
            actual = actual[0]
        if actual != value:
            raise UnsupportedModel(f"{config.get('name')}: {key}={config.get(key)!r}, expected {value!r}")


class LiteModel:
    """Drop-in replacement for the Keras model at inference time (``predict`` only)."""

    def __init__(self, path):
        import h5py

        with zipfile.ZipFile(path) as archive:
            config = json.loads(archive.read("config.json"))
            weights = h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r")
            if config.get("class_name") != "Sequential":
                raise UnsupportedModel(f"model class {config.get('class_name')!r}")

            self.steps, counts = [], {}
            for layer in config["config"]["layers"]:
                kind, cfg = layer["class_name"], layer["config"]
                base = _snake(kind)
                index = counts.get(base, 0)
                counts[base] = index + 1
                group = f"layers/{base}" if index == 0 else f"layers/{base}_{index}"

                def var(i, cell=False, group=group):
                    key = f"{group}/cell/vars/{i}" if cell else f"{group}/vars/{i}"
                    if key not in weights:
                        raise UnsupportedModel(f"missing weights {key}")
                    return np.asarray(weights[key], dtype=np.float64)

                if kind == "InputLayer":
                    self.input_shape = tuple(cfg["batch_shape"][1:])
                elif kind == "Conv1D":
                    _require(cfg, padding="causal", strides=1, dilation_rate=1, groups=1,
                             data_format="channels_last", use_bias=True)
                    self.steps.append(("conv", var(0), var(1), _activation(cfg["activation"])))
                elif kind == "MaxPooling1D":
                    _require(cfg, padding="valid", data_format="channels_last")
                    pool = cfg["pool_size"][0] if isinstance(cfg["pool_size"], list) else cfg["pool_size"]
                    stride = cfg["strides"][0] if isinstance(cfg["strides"], list) else cfg["strides"]
                    if pool != stride:
                        raise UnsupportedModel("pool_size != strides")
                    self.steps.append(("pool", int(pool)))
                elif kind == "LSTM":
                    _require(cfg, return_sequences=False, go_backwards=False, stateful=False,
                             use_bias=True, activation="tanh", recurrent_activation="sigmoid")
                    self.steps.append(("lstm", var(0, True), var(1, True), var(2, True), int(cfg["units"])))
                elif kind == "Dropout":
                    continue  # identity at inference
                elif kind == "Dense":
                    _require(cfg, use_bias=True)
                    self.steps.append(("dense", var(0), var(1), _activation(cfg["activation"])))
                else:
                    raise UnsupportedModel(f"layer {kind}")
            weights.close()
        if not self.steps:
            raise UnsupportedModel("no layers")

    def predict(self, X, verbose=0):
        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 3 or tuple(x.shape[1:]) != tuple(self.input_shape):
            raise ValueError(f"Expected input of shape (n, {self.input_shape}), got {x.shape}.")
        for step in self.steps:
            kind = step[0]
            if kind == "conv":
                _, kernel, bias, act = step
                k = kernel.shape[0]
                padded = np.pad(x, ((0, 0), (k - 1, 0), (0, 0)))  # causal: pad the past only
                out = np.zeros((x.shape[0], x.shape[1], kernel.shape[2]))
                for offset in range(k):
                    out += padded[:, offset:offset + x.shape[1], :] @ kernel[offset]
                x = act(out + bias)
            elif kind == "pool":
                size = step[1]
                steps = x.shape[1] // size
                x = x[:, :steps * size, :].reshape(x.shape[0], steps, size, x.shape[2]).max(axis=2)
            elif kind == "lstm":
                _, kernel, recurrent, bias, units = step
                h = np.zeros((x.shape[0], units))
                c = np.zeros((x.shape[0], units))
                for t in range(x.shape[1]):
                    z = x[:, t, :] @ kernel + h @ recurrent + bias
                    i = _sigmoid(z[:, :units])
                    f = _sigmoid(z[:, units:2 * units])
                    g = np.tanh(z[:, 2 * units:3 * units])
                    o = _sigmoid(z[:, 3 * units:])
                    c = f * c + i * g
                    h = o * np.tanh(c)
                x = h
            elif kind == "dense":
                _, kernel, bias, act = step
                x = act(x @ kernel + bias)
        return x.astype(np.float32)
