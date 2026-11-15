# Putting MYR SupplyLens online

This folder is ready to push as it is. It contains the redesigned app, the trained
models from Drive (`artifacts/`) and a copy of the BNM rates as a fallback.

## 1. Push to GitHub

Create an empty repository on github.com (private is fine), then from this folder:

    git init
    git add .
    git commit -m "MYR SupplyLens: redesigned interface, hosted build"
    git branch -M main
    git remote add origin https://github.com/<your-username>/<repo-name>.git
    git push -u origin main

Check on GitHub that `artifacts/models/` shows the `.keras` files. If it is empty,
an old `.gitignore` is still excluding them.

## 2. Deploy on Streamlit Community Cloud

1. Go to share.streamlit.io and sign in with the same GitHub account.
2. Create app, choose the repository, branch `main`, main file `app.py`.
3. In Advanced settings choose Python 3.12.
4. Deploy. The first build takes a few minutes; after that you get a public link.

GitHub only stores the code. Streamlit Community Cloud is what runs it.

## What to expect

- **It sleeps.** With no visitors for a while the app goes to sleep and needs about a
  minute to wake. Open the link before a demo.
- **Saved invoices and suppliers are temporary.** They live in a local SQLite file that
  is reset whenever the app restarts or redeploys.
- **Rates refresh themselves.** The app re-downloads the BNM file when its copy is more
  than 12 hours old and falls back to the bundled copy if the download fails.
- **No TensorFlow on the server.** The app runs the trained networks with NumPy
  (`src/supplylens/lite_inference.py`), which cuts memory from about 740 MB to about
  280 MB. `tests/test_lite_inference.py` checks the output against Keras for every model
  file. Training is unchanged and still uses TensorFlow in Colab.

## After joint V2.2 training in Colab

V2.2 changes the model input from one currency's 9 features to a joint panel of
63 features (9 features × 7 currencies). Therefore deploy the updated source
code and artifacts together. Do not copy only the model weights into an
unmodified V2.1 app.

The Colab notebook creates `MYR_SupplyLens_V2_2_GitHub_Release.zip`. Extract
its contents over the repository after reviewing the files, then:

    git add .
    git status
    git commit -m "MYR SupplyLens V2.2 joint multi-currency model"
    git push

Streamlit Community Cloud redeploys automatically.

Streamlit Cloud redeploys automatically.

The current scalers were saved by two scikit-learn versions (USD V2.1 with 1.6.1, the older
bundles with 1.9.1). Both load and give the same numbers, but the server log will show a
version warning until every currency is retrained in the same Colab session.
