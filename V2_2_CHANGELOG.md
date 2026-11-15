# V2.2 changelog

- Joint multi-currency CNN-LSTM: 7 currency input panels, 9 features per currency.
- One shared network forecasts 63 log-return steps for each of the 7 currencies.
- GPU-first TensorFlow training, up to 1,000 epochs with validation early stopping and automatic CPU fallback when explicitly allowed.
- GPU memory growth enabled; Conv1D and cuDNN-compatible LSTM training placed on the detected GPU.
- Per-currency target scaling and common input scaling.
- Per-currency benchmark selection and calibration retained.
- Selection scores recomputed on common matched origins.
- Inference now passes the complete currency panel to V2.2 bundles.
- Legacy V2.1/V2/V1 artifact loading remains available.
- Regression tests cover multivariate input shape and rejection of unsafe CNN paths.
