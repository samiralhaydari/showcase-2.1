import streamlit as st

from src.supplylens import ui
from src.supplylens.config import CURRENCY_NAMES
from src.supplylens.database import dashboard_counts, list_exposures
from src.supplylens.ui_helpers import (
    cached_rates, format_rate, get_forecast, priority_currencies, trained_currencies,
)

ui.masthead(
    "Currency & cost intelligence for Malaysian importers",
    "What will your next import really cost in ringgit?",
    "MYR SupplyLens turns Bank Negara reference rates and benchmarked forecasts into three decisions: "
    "what an invoice will cost, which supplier is cheaper once landed, and what it does to your margin.",
)

try:
    rates = cached_rates()
except Exception as exc:
    ui.note(
        f"The exchange-rate file could not be loaded: {ui.esc(exc)}. Check the internet connection, "
        "then rerun the data cell in the Colab notebook.",
        tag="Rates unavailable", kind="caution",
    )
    st.stop()

codes = priority_currencies(rates)
trained = trained_currencies()
latest_date = rates.index.max()

# ---- Rate tape --------------------------------------------------------------
st.subheader(f"Reference rates · {latest_date:%d %b %Y}")
cells = []
for code in codes:
    series = rates[code].dropna()
    current = float(series.iloc[-1])
    month_ago = float(series.iloc[-22]) if len(series) > 22 else float(series.iloc[0])
    change = (current / month_ago - 1) * 100
    tone = ui.cost_tone(change, tolerance=0.005)
    model_chip = ui.chip("MODEL", "on") if code in trained else ui.chip("NO MODEL")
    cells.append(
        '<div class="sl-tape-cell">'
        f'<span class="sl-tape-code">{code}</span><span class="sl-tape-name">{ui.esc(CURRENCY_NAMES.get(code, ""))}</span>'
        f'<div class="sl-tape-rate">{format_rate(current, code)}</div>'
        f'<div class="sl-delta {tone}">{ui.delta_text(change, "%")}</div>'
        f'<div class="sl-tape-foot">{ui.sparkline(series.tail(90).tolist(), width=92, height=26)}{model_chip}</div>'
        '</div>'
    )
ui.html(f'<div class="sl-tape">{"".join(cells)}</div>')
ui.html(
    '<p class="sl-small" style="margin-top:-0.7rem">MYR per 1 unit of foreign currency. Change is over the last '
    '21 trading sessions; the line is the last 90. ▲ means the currency got more expensive in ringgit.</p>'
)

# ---- Your book --------------------------------------------------------------
counts = dashboard_counts()
exposures = list_exposures()
open_value, unpriced = 0.0, 0
for item in exposures:
    if item["currency"] in rates.columns:
        open_value += item["invoice_amount"] * float(rates[item["currency"]].dropna().iloc[-1])
    else:
        unpriced += 1

missing = [c for c in codes if c not in trained]
ui.tiles([
    {"label": "Currencies tracked", "value": len(rates.columns),
     "note": f"{len(codes)} regional priorities shown above"},
    {"label": "Forecast models", "value": f"{len([c for c in trained if c in codes])} of {len(codes)}",
     "kind": "accent" if not missing else "warn",
     "note": ("All priority currencies have a model." if not missing
              else f"Still to train: {', '.join(missing)}. These use today's rate.")},
    {"label": "Saved invoices", "value": counts["exposures"],
     "note": (f"≈ RM {open_value:,.0f} at today's rates" if exposures else "Save one from Invoice cost")},
    {"label": "Supplier directory", "value": counts["suppliers"],
     "note": "Saved from Compare suppliers"},
])

# ---- Where to go --------------------------------------------------------------
st.subheader("Start with a decision")
decisions = [
    ("pages/02_Procurement_Cost_Simulator.py", "Invoice cost", ":material/receipt_long:",
     "I have a foreign-currency invoice to pay. What should I budget in ringgit, and how wrong could that be?"),
    ("pages/03_Supplier_Comparison.py", "Compare suppliers", ":material/balance:",
     "I have quotes in different currencies. Which is cheapest once freight, insurance and duties are added?"),
    ("pages/04_Profitability_Simulator.py", "Product margin", ":material/savings:",
     "My materials are imported. How far can the rate move before my gross margin is in trouble?"),
]
for column, (path, title, icon, question) in zip(st.columns(3), decisions):
    with column, st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(question)
        st.page_link(path, label=f"Open {title.lower()}", icon=icon)

left, right = st.columns(2)
with left, st.container(border=True):
    st.markdown("**FX outlook**")
    st.caption("History, the forecast at four horizons and the range the rate has typically stayed within.")
    st.page_link("pages/01_FX_Forecast_Centre.py", label="Open FX outlook", icon=":material/show_chart:")
with right, st.container(border=True):
    st.markdown("**Model report card**")
    st.caption("How each forecasting method scored on data it never saw, and which one the app uses where.")
    st.page_link("pages/05_Model_Insights.py", label="Open report card", icon=":material/fact_check:")

# ---- One honest line about the forecasts ---------------------------------------
flat = []
for code in [c for c in codes if c in trained]:
    result = get_forecast(rates, code)
    if result.ok and all(cp["method"] == "Random walk" for cp in result.checkpoints()):
        flat.append(code)
if flat:
    ui.note(
        f"For <b>{', '.join(flat)}</b>, testing found no method that reliably beats “the rate stays where it is”. "
        "The app therefore plans with today's rate for those currencies and shows the range it has historically "
        "moved within. That range, not the single number, is what to budget around.",
        tag="How to read the forecasts", kind="insight",
    )

ui.footer()
