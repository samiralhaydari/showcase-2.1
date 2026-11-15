# MYR SupplyLens: interface redesign and performance fix

Model, features, training and evaluation are untouched. Existing `_v2_1_`, V2 and V1 bundles load as before.

## Install

**Option A (simplest).** Unzip `MYR_SupplyLens_UI_Redesign.zip` into the Drive project folder
`MyDrive/MYR_SupplyLens/`, overwrite when asked, then rerun the Streamlit launch cell.
Do not rerun Cell 2 afterwards with the old patch zip: it would put the old `app.py` and pages back.

**Option B (through the notebook).** Rename the zip to `MYR_SupplyLens_V2_1_1_Patch.zip`, replace the one
in the root of My Drive, and run the notebook as usual. The zip has the same root-level layout Cell 2 expects.
`evaluation.py` in this zip is the same already-patched file you sent me, so Cell 8 has nothing new to change.

No launch flags change. The theme is applied from `src/supplylens/ui.py`, so no `.streamlit/config.toml` is needed.
Requires Streamlit 1.40 or newer (Colab already ships newer).

## What was wrong

1. **Slow on every click.** `forecast_currency` ran in full on every widget interaction: it re-read the
   Keras model from Drive, re-ran the network, re-fitted ARIMA and rewrote two CSV files. Supplier Comparison
   did that once per foreign-currency supplier, on every keystroke. Forecasts are now cached per currency and
   recomputed only when the rates or the model files change. A retrained model is picked up without a restart.
2. **Wrong model count on the home page.** `trained_currencies()` matched `cnn_lstm_v2_1_USD.keras` with the
   V2 pattern and counted a phantom currency called `1_USD`. Home said 4 trained models; there are 3.
3. **Rates never refreshed.** The CSV was downloaded once and reused forever. It now refreshes when older than
   12 hours and falls back to the saved copy if the download fails.
4. **Cost increases shown in green.** `st.metric` colours any increase green; a higher invoice cost is bad news.
5. **Pages could crash.** Only `FileNotFoundError` was handled; any other model error took the page down.
6. **Dead features.** Saved invoices were never shown anywhere and nothing ever saved a supplier, so
   "Saved suppliers" was always 0. There is now an exposure book and a supplier directory button.
7. **Presentation.** Sidebar entry called "app", overlapping chart labels, raw column names such as
   `mae_skill_vs_random_walk_pct`, a horizon axis with ticks at 37 and 53, a bar chart of two identical bars.

## Hosting

See `DEPLOY.md`. The app now runs the trained networks with NumPy instead of TensorFlow
(`src/supplylens/lite_inference.py`, with a Keras fallback), so it fits on free hosting.
Forecasts are unchanged: a test compares the two for every model file.

## Still to do on the modelling side (not changed here)

- CNY and IDR are on old V2 bundles; only USD is V2.1. SGD, THB, JPY and KRW have no model.
  The app now says so on every page instead of hiding it.
- On the USD test set no method beats random walk with confidence (every 95% interval spans zero).
  The UI is built around that: it leads with the 80% range and explains why the point forecast is often flat.

## Files

New: `pages/00_Overview.py`, `src/supplylens/ui.py`, `src/supplylens/lite_inference.py`, `src/supplylens/assets/`,
`tests/test_ui_helpers.py`, `tests/test_lite_inference.py`, `requirements.txt`, `DEPLOY.md`.
Rewritten: `app.py`, `pages/01`–`05`, `src/supplylens/ui_helpers.py`.
Small edits: `forecasting.py` (optional `bundle=` argument, `bundle_signature`, NumPy loader), `data.py` (refresh),
`database.py` (delete helpers), both requirements files (Streamlit floor).
Unchanged: `evaluation.py`, `training.py`, `model.py`, `features.py`, `business.py`, `config.py`, scripts.
