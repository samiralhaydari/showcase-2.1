import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.supplylens import ui
from src.supplylens.config import MODEL_VERSION
from src.supplylens.data import get_currency_series
from src.supplylens.ui_helpers import (
    HORIZON_SHORT, cached_rates, currency_label, format_rate, get_forecast,
    horizon_label, priority_currencies, trained_currencies,
)

ui.masthead(
    "FX outlook",
    "Where is the rate, and where could it go?",
    "The Bank Negara reference rate, the forecast at four planning horizons, and the range the rate has "
    "typically stayed within over the same stretch of time.",
)

try:
    rates = cached_rates()
except Exception as exc:
    ui.note(f"Rates could not be loaded: {ui.esc(exc)}", tag="Rates unavailable", kind="caution")
    st.stop()

codes = priority_currencies(rates) or list(rates.columns)
trained = trained_currencies()

pick, span = st.columns([3, 2])
with pick:
    currency = ui.choose(
        "Currency", codes, key="fx_currency",
        format_func=lambda c: f"{c} ●" if c in trained else c,
        help="● marks currencies with a trained forecast model.",
    )
with span:
    window = ui.choose(
        "History shown", [63, 126, 252, 756], default=252, key="fx_window",
        format_func=lambda n: {63: "3 months", 126: "6 months", 252: "1 year", 756: "3 years"}[n],
    )

series = get_currency_series(rates, currency)
result = get_forecast(rates, currency)
current = result.current_rate
history = series.tail(window)
month_ago = float(series.iloc[-22]) if len(series) > 22 else float(series.iloc[0])
year = series.tail(252)
position = (current - year.min()) / ((year.max() - year.min()) or 1) * 100
change = (current / month_ago - 1) * 100

if result.ok:
    model_value, model_note = result.model_version, "Benchmarked forecast available"
    if result.metadata.get("model_version") != MODEL_VERSION:
        model_note = "Older bundle: retrain for joint cross-currency forecasting"
elif result.status == "no_model":
    model_value, model_note = "None", "Train the joint seven-currency model in the Colab notebook"
else:
    model_value, model_note = "Error", "The model bundle could not be used"

ui.tiles([
    {"label": f"MYR per 1 {currency}", "value": format_rate(current, currency), "kind": "accent",
     "note": f"{currency_label(currency).split(' — ')[1]} · {result.latest_date:%d %b %Y}"},
    {"label": "Last 21 sessions", "value": f"{change:+.2f}%",
     "note": ("Imports in this currency got dearer" if change > 0.005 else
              "Imports in this currency got cheaper" if change < -0.005 else "Flat over the month")},
    {"label": "Position in 1-year range", "value": f"{position:.0f}%",
     "note": f"Low {format_rate(year.min(), currency)} · High {format_rate(year.max(), currency)}"},
    {"label": "Forecast model", "value": model_value, "small": True,
     "kind": "" if result.ok and result.metadata.get("model_version") == MODEL_VERSION else "warn",
     "note": model_note},
])

# ---- Chart ---------------------------------------------------------------------
show_reference = False
if result.ok:
    show_reference = st.toggle(
        "Also draw the raw CNN-LSTM path", value=False,
        help="The neural network's day-by-day path, shown for reference. The diamonds are the forecasts "
             "the app actually uses.",
    )

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=history.index, y=history.values, name="BNM middle rate", mode="lines",
    line=dict(color=ui.BLUE, width=2),
    hovertemplate="%{x|%d %b %Y}<br>%{y:.6g}<extra>Reference rate</extra>",
))

checkpoints = result.checkpoints() if result.ok else []
if result.ok:
    anchor_x, anchor_y = result.latest_date, current
    ranged = [cp for cp in checkpoints if cp["low"] is not None]
    if ranged:
        xs = [anchor_x] + [cp["date"] for cp in ranged]
        fig.add_trace(go.Scatter(
            x=xs, y=[anchor_y] + [cp["low"] for cp in ranged], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=[anchor_y] + [cp["high"] for cp in ranged], mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(200,98,10,0.14)",
            name="80% range (historical errors)", hoverinfo="skip",
        ))
    if show_reference:
        forecast = result.forecast
        fig.add_trace(go.Scatter(
            x=forecast["date"], y=forecast["forecast_rate"], name="CNN-LSTM path (reference)",
            mode="lines", line=dict(color=ui.VIOLET, width=1.5, dash="dot"),
            hovertemplate="%{x|%d %b %Y}<br>%{y:.6g}<extra>CNN-LSTM</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[anchor_x] + [cp["date"] for cp in checkpoints],
        y=[anchor_y] + [cp["rate"] for cp in checkpoints],
        mode="lines", line=dict(color=ui.ORANGE, width=1.5, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))
    hover = [
        f"{horizon_label(cp['horizon'])} · {cp['method']}<br>{format_rate(cp['rate'], currency)}"
        + (f"<br>80% range {format_rate(cp['low'], currency)} – {format_rate(cp['high'], currency)}"
           if cp["low"] is not None else "<br>No range available")
        for cp in checkpoints
    ]
    fig.add_trace(go.Scatter(
        x=[cp["date"] for cp in checkpoints], y=[cp["rate"] for cp in checkpoints],
        mode="markers+text", name="Forecast used by the app",
        text=[HORIZON_SHORT.get(cp["horizon"], str(cp["horizon"])) for cp in checkpoints],
        textposition="top center", textfont=dict(family=ui.FONT_MONO, size=11, color=ui.INK),
        marker=dict(color=ui.ORANGE, size=11, symbol="diamond", line=dict(color=ui.CARD, width=2)),
        hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
    ))
    fig.add_vline(x=anchor_x, line=dict(color=ui.NEUTRAL, width=1, dash="dot"))
    fig.add_annotation(
        x=anchor_x, y=1, yref="paper", text="today", showarrow=False, xanchor="right", xshift=-6,
        font=dict(family=ui.FONT_MONO, size=11, color=ui.FAINT),
    )

ui.style_figure(fig, height=430)
fig.update_yaxes(title_text=f"MYR per 1 {currency}", tickformat=".6~g")
fig.update_layout(hovermode="closest")
ui.chart(fig, key="fx_chart")

# ---- Checkpoints ----------------------------------------------------------------
if result.ok:
    st.subheader("Rates to plan with")
    cards = []
    for cp in checkpoints:
        tone = ui.cost_tone(cp["change_pct"], tolerance=0.005)
        if cp["low"] is not None:
            half_width = (cp["high"] / cp["low"] - 1) * 50
            band = (f"80% range {format_rate(cp['low'], currency)} – {format_rate(cp['high'], currency)}"
                    f" (about ±{half_width:.1f}%)")
        else:
            band = "No range: too few calibration samples"
        cards.append(
            '<div class="sl-cp"><div class="sl-cp-head">'
            f'<span class="sl-cp-h">{horizon_label(cp["horizon"])}</span>'
            f'<span class="sl-cp-date">≈ {cp["date"]:%d %b %Y}</span></div>'
            f'<div class="sl-cp-rate">{format_rate(cp["rate"], currency)}</div>'
            f'<div class="sl-delta {tone}">{ui.delta_text(cp["change_pct"], "%")} vs today</div>'
            f'<div class="sl-cp-range">{band}</div>{ui.chip(cp["method"], "dark")}</div>'
        )
    ui.html(f'<div class="sl-cps">{"".join(cards)}</div>')

    flat = [cp for cp in checkpoints if cp["method"] == "Random walk"]
    if len(flat) == len(checkpoints):
        ui.note(
            f"At every horizon, the method that scored best on held-back {currency} data was the random walk: "
            "“the rate stays where it is”. So the plan rate equals today's rate and the forecast line is flat. "
            "That is a finding, not a fault. The orange range is the part that should shape your budget.",
            tag="Why the forecast is flat", kind="insight",
        )
    elif flat:
        ui.note(
            f"At {len(flat)} of {len(checkpoints)} horizons the best-scoring method was the random walk "
            "(“the rate stays where it is”), so those plan rates equal today's rate.",
            tag="Reading the cards", kind="insight",
        )
    if result.metadata.get("model_version") != MODEL_VERSION:
        ui.note(
            f"{currency} is still on an older model bundle ({result.model_version}). It has no per-horizon method "
            "selection, so the CNN-LSTM output is used directly. Retrain it in the notebook to bring it in line "
            "with the V2.2 joint currency model.",
            tag="Older model", kind="caution",
        )
elif result.status == "no_model":
    ui.note(
        f"There is no trained model for <b>{currency}</b> yet, so only history is shown. Train it in the Colab "
        f"notebook (<code>python scripts/train_model.py --currency {currency} --epochs 40</code>) and it will "
        "appear here without restarting the app.",
        tag="No forecast yet", kind="caution",
    )
else:
    ui.note(f"The {currency} model bundle could not be used: {ui.esc(result.message)}",
            tag="Forecast error", kind="caution")

with st.expander("How to read this page"):
    st.markdown(
        "- **Diamonds** are the forecasts the app uses. For each horizon, the method that scored best on a "
        "separate slice of history is chosen; the chip on each card names it.\n"
        "- **The orange range** covers 80% of that method's past errors at that horizon. It describes history; "
        "it is not a guarantee.\n"
        "- **Dates** count weekdays only and do not allow for Malaysian public holidays.\n"
        "- **Horizons** are trading sessions: 5 ≈ one week, 21 ≈ one month, 63 ≈ one quarter."
    )

ui.footer()
