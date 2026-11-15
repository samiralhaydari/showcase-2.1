# MYR SupplyLens V2.2 — Joint Multi-Currency CNN-LSTM

**Predict. Plan. Procure. Profit.**

V2.2 replaces seven independent univariate neural networks with one shared multi-output CNN-LSTM trained jointly on the seven currencies used by the application:

`USD, CNY, IDR, SGD, THB, JPY, KRW`

## What changed

- Each currency contributes the same nine leakage-safe features.
- The model input is a 90-session window with 63 channels (9 features × 7 currencies).
- One shared CNN-LSTM predicts 63 future daily log returns for all seven currencies simultaneously (441 outputs).
- All currency targets are standardized separately, then optimized together in one training pass.
- The app selects the correct 63-output slice for the currency currently being viewed.
- The existing per-currency artifact naming convention is preserved, so the dashboard's bundle discovery and NumPy `LiteModel` inference continue to work.
- V2.1, V2 and V1 bundles remain as legacy fallbacks.

## Training design

- GPU-first TensorFlow execution: Colab GPU is selected automatically, with GPU memory growth enabled. CPU is an explicit fallback.
- The Conv1D and cuDNN-compatible LSTM layers train on the selected GPU; pandas/scikit-learn preprocessing and ARIMA benchmarks remain CPU-side.
- Maximum 1,000 epochs by default, with early stopping on a chronological validation partition.
- Batch size 128 by default.
- Five chronological partitions: 60% train, 10% early-stop validation, 10% method selection, 5% calibration, 15% untouched final test.
- Scaling is fitted only on the training partition.
- Method selection is per currency and per horizon on the separate selection partition.
- Calibration uses a separate chronological partition.
- Final test origins are spaced by 63 observations to avoid overlapping target blocks.
- Benchmark comparisons include random walk, historical drift, recent drift and optional ARIMA(1,1,1).

The seven currencies are learned together. This is not seven independent processes running concurrently: a single shared model updates all currency outputs in the same optimizer step. TensorFlow places the model and training operations on the selected GPU; CPU threads handle preprocessing and benchmark calculations.

## Colab

1. Put `finished-product.zip` in the root of Google Drive (`My Drive`).
2. Open `MYR_SupplyLens_V2_2_Joint_Training.ipynb` in Google Colab.
3. In Colab choose Runtime → Change runtime type → T4 GPU (or another available GPU), then run cells from top to bottom. The notebook requires a visible GPU by default.
4. The notebook extracts the archive into `MyDrive/MYR_SupplyLens_V2_2`, applies the V2.2 source patch, runs tests, refreshes the official BNM data if enabled, and trains the joint model.
5. New bundles are written under `artifacts/models`, `artifacts/scalers` and `artifacts/metadata` using the `_v2_2_` suffix.
6. The notebook creates `MYR_SupplyLens_V2_2_GitHub_Release.zip` in the root of My Drive.

## Important deployment requirement

V2.2 changes the inference input from `(90, 9)` to `(90, 63)`. **Do not upload only the new model artifacts to the old V2.1 app source.** The updated `forecasting.py`, `ui_helpers.py`, `config.py` and Streamlit pages must be deployed together with the new artifacts. The notebook includes those source changes in the GitHub release archive.

## Runtime safety

The notebook does not install or upgrade TensorFlow, NumPy, SciPy or scikit-learn. It checks the installed scientific stack and stops if required packages are missing. It does not programmatically restart or kill the Colab runtime.

## Data and limitations

Bank Negara Malaysia daily middle reference rates via data.gov.my, quoted as MYR per one unit of foreign currency. These are reference rates, not necessarily bank transaction rates. Correlation is learned from historical co-movement and is not evidence of causation. The model does not observe future macroeconomic shocks or policy decisions. Forecasts and empirical error ranges are planning estimates, not guarantees or financial advice.
