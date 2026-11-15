import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.supplylens import ui
from src.supplylens.config import (
    DISPLAY_HORIZONS, LEGACY_MODEL_VERSIONS, METADATA_DIR, MODEL_VERSION, PRIORITY_CURRENCIES,
)
from src.supplylens.ui_helpers import HORIZON_LABELS, horizon_label

ui.masthead(
    "Model report card",
    "Should I trust these forecasts?",
    "Every forecasting method is scored on data it never saw during training. This page shows which method "
    "the app uses at each horizon, and whether the neural network actually beats the simplest alternative.",
)

LEGACY = {
    "skill_vs_random_walk_pct": "mae_skill_vs_random_walk_pct",
    "skill_ci_lower_pct": "mae_skill_ci_lower_pct",
    "skill_ci_upper_pct": "mae_skill_ci_upper_pct",
}


def _artifact(kind, version, currency, suffix):
    return METADATA_DIR / (f"{kind}_{currency}.{suffix}" if version == "v1" else f"{kind}_{version}_{currency}.{suffix}")


def _stamp():
    """Changes whenever an evaluation file changes, so retraining refreshes the cache."""
    if not METADATA_DIR.exists():
        return ()
    return tuple(sorted((p.name, p.stat().st_mtime_ns) for p in METADATA_DIR.glob("met*")))


@st.cache_data(show_spinner=False)
def load_reports(stamp):
    reports = []
    for currency in PRIORITY_CURRENCIES:
        for version in [MODEL_VERSION] + [v for v in LEGACY_MODEL_VERSIONS if v != MODEL_VERSION]:
            meta_path = _artifact("metadata", version, currency, "json")
            metric_path = _artifact("metrics", version, currency, "csv")
            if not (meta_path.exists() and metric_path.exists()):
                continue
            metrics = pd.read_csv(metric_path)
            for old, new in LEGACY.items():  # V2 files used MAE-only column names
                if new not in metrics.columns and old in metrics.columns:
                    metrics[new] = metrics[old]
            metrics["currency"] = currency
            policy_path = _artifact("selection_policy", version, currency, "csv")
            policy = pd.read_csv(policy_path) if version == MODEL_VERSION and policy_path.exists() else None
            reports.append({
                "currency": currency, "version": version,
                "metadata": json.loads(meta_path.read_text(encoding="utf-8")),
                "metrics": metrics, "policy": policy,
            })
            break
    return reports


reports = load_reports(_stamp())
if not reports:
    ui.note(
        "No evaluation files were found. Run the training cells in the Colab notebook; the report card fills "
        "in as soon as a model has been trained.",
        tag="Nothing to report yet", kind="caution",
    )
    ui.footer()
    st.stop()

by_code = {r["currency"]: r for r in reports}
missing = [c for c in PRIORITY_CURRENCIES if c not in by_code]
legacy = [r["currency"] for r in reports if r["version"] != MODEL_VERSION]


def verdict(row):
    low, high = row.get("mae_skill_ci_lower_pct"), row.get("mae_skill_ci_upper_pct")
    if pd.isna(low) or pd.isna(high):
        return "unknown", "no interval"
    if low > 0:
        return "better", "better than random walk"
    if high < 0:
        return "worse", "worse than random walk"
    return "inconclusive", "no clear difference"


cnn_rows = []
for report in reports:
    cnn = report["metrics"][report["metrics"]["model"] == "CNN-LSTM"]
    for _, row in cnn.iterrows():
        if int(row["horizon"]) in DISPLAY_HORIZONS and "mae_skill_vs_random_walk_pct" in row:
            cnn_rows.append({**row.to_dict(), "verdict": verdict(row)[0], "verdict_text": verdict(row)[1]})
cnn_frame = pd.DataFrame(cnn_rows)
wins = int((cnn_frame["verdict"] == "better").sum()) if not cnn_frame.empty else 0
total = len(cnn_frame)

ui.tiles([
    {"label": "Currencies evaluated", "value": f"{len(reports)} of {len(PRIORITY_CURRENCIES)}",
     "kind": "accent" if not missing else "warn",
     "note": f"Not trained: {', '.join(missing)}" if missing else "All priority currencies covered"},
    {"label": "On current V2.2 method", "value": f"{len(reports) - len(legacy)} of {len(reports)}",
     "kind": "accent" if not legacy else "warn",
     "note": f"Retrain to upgrade: {', '.join(legacy)}" if legacy else "No legacy bundles in use"},
    {"label": "CNN-LSTM clearly beats random walk", "value": f"{wins} of {total}",
     "note": "currency-horizon tests where the 95% interval is entirely above zero"},
])

if total and wins == 0:
    ui.note(
        "On the held-out test data, the CNN-LSTM does not beat “the rate stays where it is” with statistical "
        "confidence at any horizon. Its error is within a percent or two of the random walk either way, and "
        "every confidence interval spans zero. This is the normal result for daily exchange rates, and it is why "
        "the app picks a method per horizon and reports ranges instead of leaning on one model.",
        tag="The headline", kind="insight",
    )

card, benchmarks, method, limits = st.tabs(["Scorecard", "All benchmarks", "How it works", "Data & limits"])

# ---- Scorecard ---------------------------------------------------------------------
with card:
    st.subheader("Method in use, by currency and horizon")
    header = "".join(f'<th>{HORIZON_LABELS[h]}</th>' for h in DISPLAY_HORIZONS)
    body = []
    for report in reports:
        code, version, policy = report["currency"], report["version"], report["policy"]
        tag = ui.chip("V2.2", "on") if version == MODEL_VERSION else ui.chip(f"{version.upper()} · RETRAIN", "legacy")
        cells = []
        for h in DISPLAY_HORIZONS:
            chosen, origins = "CNN-LSTM", None
            if policy is not None and not policy.empty:
                match = policy[policy["horizon"] == h]
                if not match.empty:
                    chosen = str(match.iloc[0]["selected_model"])
                    origins = match.iloc[0].get("n_selection_origins")
                    if bool(match.iloc[0].get("selection_fallback", False)):
                        chosen += " (fallback)"
            skill = cnn_frame[(cnn_frame["currency"] == code) & (cnn_frame["horizon"] == h)] if total else cnn_frame
            line = ""
            if len(skill):
                s = skill.iloc[0]
                line = (f'<span class="sl-verdict">CNN-LSTM {s["mae_skill_vs_random_walk_pct"]:+.2f}% vs random walk'
                        f'<br>→ {s["verdict_text"]}</span>')
            basis = (f'<span class="sl-verdict">chosen from {int(origins)} selection windows</span>'
                     if origins is not None and pd.notna(origins) else
                     '<span class="sl-verdict">default: no selection step in this bundle</span>')
            cells.append(f'<td>{ui.chip(chosen, "dark")}{basis}{line}</td>')
        body.append(f'<tr><td><b>{code}</b><br>{tag}</td>{"".join(cells)}</tr>')
    ui.html(f'<div class="sl-scroll"><table class="sl-table"><thead><tr><th>Currency</th>{header}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')
    st.caption(
        "The dark chip is the method the app forecasts with. The line under it is the CNN-LSTM's error on the "
        "untouched test set relative to random walk: positive is better."
    )

    if total:
        st.subheader("CNN-LSTM against random walk, with uncertainty")
        focus = ui.choose("Currency", [r["currency"] for r in reports], key="insight_currency")
        sub = cnn_frame[cnn_frame["currency"] == focus].sort_values("horizon")
        fig = go.Figure()
        fig.add_hline(y=0, line=dict(color=ui.INK, width=1))
        has_ci = sub["mae_skill_ci_lower_pct"].notna().all() if "mae_skill_ci_lower_pct" in sub else False
        fig.add_trace(go.Scatter(
            x=[HORIZON_LABELS[int(h)] for h in sub["horizon"]], y=sub["mae_skill_vs_random_walk_pct"],
            mode="markers+text", name="CNN-LSTM",
            text=[f"{v:+.2f}%" for v in sub["mae_skill_vs_random_walk_pct"]], textposition="middle right",
            textfont=dict(family=ui.FONT_MONO, size=12, color=ui.INK),
            marker=dict(color=ui.BLUE, size=12, line=dict(color=ui.CARD, width=2)), cliponaxis=False,
            error_y=dict(
                type="data", symmetric=False, color=ui.BLUE, thickness=2, width=7,
                array=(sub["mae_skill_ci_upper_pct"] - sub["mae_skill_vs_random_walk_pct"]) if has_ci else None,
                arrayminus=(sub["mae_skill_vs_random_walk_pct"] - sub["mae_skill_ci_lower_pct"]) if has_ci else None,
                visible=bool(has_ci),
            ),
            hovertemplate="%{x}<br>MAE improvement %{y:+.2f}%<extra></extra>",
        ))
        ui.style_figure(fig, height=320, legend=False)
        fig.update_xaxes(type="category", title_text="Forecast horizon", range=[-0.4, len(sub) - 0.4])
        fig.update_yaxes(title_text="Error vs random walk (above 0 = better)", ticksuffix="%")
        ui.chart(fig, key="skill_chart")
        origins = int(sub["n_origins"].iloc[0]) if "n_origins" in sub and len(sub) else 0
        st.caption(
            f"Dots are the improvement in mean absolute error on {origins} non-overlapping test windows; bars "
            "are 95% bootstrap intervals. A bar that crosses zero means the difference could be chance."
        )

# ---- Benchmarks --------------------------------------------------------------------
with benchmarks:
    st.subheader("Every method on the untouched test set")
    code = ui.choose("Currency", [r["currency"] for r in reports], key="bench_currency")
    metrics = by_code[code]["metrics"].copy()
    if "directional_accuracy_pct" in metrics.columns:
        metrics.loc[metrics["model"] == "Random walk", "directional_accuracy_pct"] = np.nan
    columns = {
        "horizon": "Horizon (sessions)", "model": "Method", "n_origins": "Test windows",
        "mae": "MAE", "rmse": "RMSE", "mape_pct": "MAPE",
        "directional_accuracy_pct": "Direction right",
        "mae_skill_vs_random_walk_pct": "MAE vs random walk",
        "mae_skill_ci_lower_pct": "95% low", "mae_skill_ci_upper_pct": "95% high",
    }
    view = metrics[[c for c in columns if c in metrics.columns]].rename(columns=columns)
    view = view.sort_values(["Horizon (sessions)", "MAE"])
    percent = st.column_config.NumberColumn(format="%.2f%%")
    ui.table(view, column_config={
        "MAE": st.column_config.NumberColumn(format="%.6f"),
        "RMSE": st.column_config.NumberColumn(format="%.6f"),
        "MAPE": percent, "Direction right": percent, "MAE vs random walk": percent,
        "95% low": percent, "95% high": percent,
    })
    st.caption(
        "Sorted best-first within each horizon. MAE and RMSE are in MYR per unit. Random walk predicts no "
        "change, so it has no direction score. Positive “MAE vs random walk” means lower error than random walk."
    )

# ---- Method ------------------------------------------------------------------------
with method:
    st.subheader("From rates to a forecast")
    steps = [
        ("90 × 63", "90 sessions × 9 features across 7 currencies"),
        ("Conv1D 64 + 32", "Causal patterns across all currency channels"),
        ("MaxPool 2", "Halve the sequence"),
        ("LSTM 64", "Shared temporal representation"),
        ("Dropout 0.25", "Regularise shared learning"),
        ("Dense 128", "Combine currency relationships"),
        ("7 × 63 returns", "63 future log returns for each currency"),
    ]
    ui.html('<div class="sl-pipe">' + '<span class="sl-pipe-arrow">→</span>'.join(
        f'<div class="sl-pipe-step"><b>{a}</b><span>{b}</span></div>' for a, b in steps) + '</div>')

    left, right = st.columns(2)
    with left:
        st.markdown("**What the network sees**")
        features = [
            ("log_rate", "Log of the exchange-rate level"),
            ("return_1d", "One-session log return"),
            ("momentum_5 / 10 / 20", "Log change over 5, 10 and 20 sessions"),
            ("volatility_5 / 20 / 60", "Rolling standard deviation of returns"),
            ("range_20", "20-session high-low range of log rates"),
        ]
        ui.html('<table class="sl-table"><thead><tr><th>Feature</th><th>Meaning</th></tr></thead><tbody>' + "".join(
            f'<tr><td><code>{a}</code></td><td>{b}</td></tr>' for a, b in features) + '</tbody></table>')
        st.caption(
            "The same nine features are calculated for each of the seven currencies. "
            "All 63 channels enter the shared model together."
        )
    with right:
        st.markdown("**How history is split, oldest to newest**")
        splits = [("Train", 60, ui.BLUE), ("Early stop", 10, ui.TEAL), ("Pick method", 10, ui.ORANGE),
                  ("Calibrate", 5, ui.VIOLET), ("Final test", 15, ui.ROSE)]
        ui.html(
            '<div style="display:flex;gap:2px;height:30px;margin:0.3rem 0 0.4rem 0">' + "".join(
                f'<div style="flex:{w};background:{c};color:#fff;font-family:{ui.FONT_MONO};font-size:0.72rem;'
                f'display:flex;align-items:center;justify-content:center">{w}%</div>' for _, w, c in splits)
            + '</div><div class="sl-key">' + "".join(
                f'<span><i class="sq" style="background:{c}"></i>{n}</span>' for n, _, c in splits) + '</div>'
        )
        st.markdown(
            "- The method for each horizon is picked on its own slice, never on the final test.\n"
            "- Lowest error wins; with fewer than eight usable windows, random walk is used.\n"
            "- Ranges come from a further slice and describe past errors. They are not guarantees.\n"
            "- Test windows are 63 sessions apart so their targets never overlap."
        )
    st.markdown("**Benchmarks the network must beat:** random walk, 252-session drift, 20-session drift and ARIMA(1,1,1).")

    for report in reports:
        meta = report["metadata"]
        trained_at = pd.to_datetime(meta.get("trained_at_utc"), errors="coerce")
        when = f"{trained_at:%d %b %Y, %H:%M} UTC" if pd.notna(trained_at) else "date unknown"
        with st.expander(f"{report['currency']} bundle · {report['version'].replace('_', '.').upper()} · trained {when}"):
            ui.tiles([
                {"label": "Data up to", "value": meta.get("latest_observation_date", "—"), "small": True},
                {"label": "Training samples", "value": f"{meta.get('train_samples', 0):,}", "small": True},
                {"label": "Calibration samples", "value": f"{meta.get('calibration_samples', 0):,}", "small": True},
                {"label": "Test windows", "value": meta.get("walk_forward_origins", "—"), "small": True},
            ])
            st.caption(meta.get("selection_method", "Legacy bundle: no separate method-selection step."))

# ---- Limits ------------------------------------------------------------------------
with limits:
    st.subheader("What these numbers can and cannot tell you")
    st.markdown(
        "- **Source.** Bank Negara Malaysia daily middle rates via data.gov.my, quoted as MYR per one unit of "
        "foreign currency.\n"
        "- **Reference, not transaction.** Your bank's rate will differ from the central-bank middle rate.\n"
        "- **Small samples.** There are only a handful of non-overlapping test windows per currency, so the "
        "confidence intervals are wide. Read them before reading the point estimate.\n"
        "- **Ranges describe the past.** The 80% and 95% ranges are built from historical errors and carry no "
        "coverage guarantee, least of all through a policy shock.\n"
        "- **Not advice.** Forecasts are planning inputs, not financial advice."
    )
    st.markdown("[Dataset documentation on data.gov.my](https://data.gov.my/data-catalogue/exchangerates_daily_1200)")

ui.footer()
