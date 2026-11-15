from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
SCALER_DIR = PROJECT_ROOT / "artifacts" / "scalers"
METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"
FORECAST_DIR = PROJECT_ROOT / "artifacts" / "forecasts"
DB_PATH = PROJECT_ROOT / "data" / "supplylens.sqlite"

DATA_URL = "https://storage.data.gov.my/finsector/exr/daily_1200.csv"
DATASET_PAGE = "https://data.gov.my/data-catalogue/exchangerates_daily_1200"

PRIORITY_CURRENCIES = ["USD", "CNY", "IDR", "SGD", "THB", "JPY", "KRW"]
CURRENCY_NAMES = {
    "USD": "US Dollar", "CNY": "Chinese Yuan", "IDR": "Indonesian Rupiah",
    "SGD": "Singapore Dollar", "THB": "Thai Baht", "JPY": "Japanese Yen",
    "KRW": "South Korean Won", "PHP": "Philippine Peso", "VND": "Vietnamese Dong",
    "BND": "Brunei Dollar", "EUR": "Euro", "GBP": "British Pound",
    "AUD": "Australian Dollar", "CAD": "Canadian Dollar", "CHF": "Swiss Franc",
    "HKD": "Hong Kong Dollar", "TWD": "Taiwan Dollar", "INR": "Indian Rupee",
    "NZD": "New Zealand Dollar"
}

# V2.2 writes a joint multi-currency artifact family; V2.1, V2 and V1 remain fallbacks.
MODEL_VERSION = "v2_2"
LEGACY_MODEL_VERSIONS = ["v2_1", "v2", "v1"]
LOOKBACK = 90
FORECAST_HORIZON = 63
DISPLAY_HORIZONS = [5, 21, 42, 63]
# Test origins use common 63-observation spacing to avoid overlapping test targets.
WALK_FORWARD_SPACING = FORECAST_HORIZON
# V2.1 chronological partitions: 60% train, 10% early-stop validation,
# 10% method selection, 5% calibration, 15% untouched final test.
SPLIT_FRACTIONS = {
    "train": 0.60,
    "validation": 0.10,
    "selection": 0.10,
    "calibration": 0.05,
    "test": 0.15,
}
MIN_SELECTION_ORIGINS = 8
CALIBRATION_MIN_SAMPLES = 30
CALIBRATION_ORIGIN_STRIDE = 5
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42
ARIMA_MAX_HISTORY = 756
FEATURE_COLUMNS = [
    "log_rate",
    "return_1d",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "volatility_5",
    "volatility_20",
    "volatility_60",
    "range_20",
]

BENCHMARK_METHODS = [
    "CNN-LSTM",
    "Random walk",
    "Historical drift (252)",
    "Recent drift (20)",
    "ARIMA(1,1,1)",
]
