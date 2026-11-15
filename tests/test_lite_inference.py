"""The NumPy evaluator must reproduce Keras exactly, or the app's forecasts would
silently differ from what was evaluated during training."""
import numpy as np
import pytest

from src.supplylens.config import MODEL_DIR
from src.supplylens.lite_inference import LiteModel

MODELS = sorted(MODEL_DIR.glob("cnn_lstm_*.keras"))


@pytest.mark.skipif(not MODELS, reason="no trained model files in artifacts/models")
@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.stem)
def test_lite_model_matches_keras(path):
    keras = pytest.importorskip("tensorflow.keras")
    reference = keras.models.load_model(path)
    lite = LiteModel(path)
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1.5, size=(8,) + lite.input_shape).astype(np.float32)
    expected = reference.predict(X, verbose=0)
    np.testing.assert_allclose(lite.predict(X), expected, rtol=1e-4, atol=1e-5)
