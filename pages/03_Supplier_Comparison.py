import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.supplylens import ui
from src.supplylens.business import invoice_cost, landed_cost
from src.supplylens.database import list_suppliers, save_supplier
from src.supplylens.ui_helpers import (
    cached_rates, currency_label, format_rate, get_forecast, horizon_label, priority_currencies,
)

ui.masthead(
    "Compare suppliers",
    "Which quote is really the cheapest?",
    "Put up to three quotes side by side, each in its own currency. Costs are converted at the rate to plan "
    "with for your payment date, then freight, insurance and duties are added.",
)

try:
    rates = cached_rates()
except Exception as exc:
    ui.note(f"Rates could not be loaded: {ui.esc(exc)}", tag="Rates unavailable", kind="caution")
    st.stop()

currencies = ["MYR"] + priority_currencies(rates)
DEFAULTS = [
    ("Supplier A", "Malaysia", "MYR", 60000.0),
    ("Supplier B", "Indonesia", "IDR", 450000000.0),
    ("Supplier C", "China", "CNY", 100000.0),
]
COST_PARTS = [("Invoice", ui.BLUE), ("Freight", ui.ORANGE), ("Insurance", ui.TEAL), ("Duties & taxes", ui.VIOLET)]

horizon = ui.horizon_picker("Pay in", key="cmp_horizon")
st.caption("The starting figures are examples. Compare like with like: same specification, same quantity.")

quotes = []
for i, column in enumerate(st.columns(3, gap="medium")):
    name0, country0, code0, amount0 = DEFAULTS[i]
    with column, st.container(border=True):
        name = st.text_input("Supplier", value=name0, key=f"name{i}")
        a, b = st.columns(2)
        with a:
            country = st.text_input("Country", value=country0, key=f"country{i}")
        with b:
            currency = st.selectbox(
                "Quoted in", currencies, key=f"currency{i}",
                index=currencies.index(code0) if code0 in currencies else 0,
            )
        amount = st.number_input(f"Quote ({currency})", min_value=0.0, value=amount0, step=1000.0,
                                 format="%.2f", key=f"amount{i}")
        with st.expander("Landing costs (MYR)", expanded=False):
            freight = st.number_input("Freight", min_value=0.0, value=1000.0, step=100.0, key=f"freight{i}")
            insurance = st.number_input("Insurance", min_value=0.0, value=200.0, step=50.0, key=f"insurance{i}")
            duties = st.number_input("Duties and other taxes", min_value=0.0, value=500.0, step=50.0,
                                     key=f"duties{i}")

    if currency == "MYR":
        cp = {"rate": 1.0, "method": "Quoted in ringgit", "low": None, "high": None}
        today_rate = 1.0
    else:
        result = get_forecast(rates, currency)
        cp = result.checkpoint(horizon)
        today_rate = result.current_rate
    invoice_myr = invoice_cost(amount, cp["rate"])
    extras = freight + insurance + duties
    quotes.append({
        "name": name.strip() or name0, "country": country.strip(), "currency": currency, "amount": amount,
        "rate": cp["rate"], "today_rate": today_rate, "basis": cp["method"],
        "low": cp["low"], "high": cp["high"],
        "Invoice": invoice_myr, "Freight": freight, "Insurance": insurance, "Duties & taxes": duties,
        "extras": extras, "total": landed_cost(invoice_myr, freight, insurance, duties),
    })

ranked = sorted(quotes, key=lambda q: q["total"])
winner, runner_up = ranked[0], ranked[1]
gap = runner_up["total"] - winner["total"]

st.subheader(f"Landed cost, paying in {horizon_label(horizon)}")
ui.tiles([
    {"label": "Cheapest landed", "value": winner["name"], "small": True, "kind": "accent",
     "note": f"{ui.esc(winner['country'])} · quoted in {winner['currency']}"},
    {"label": "Landed cost", "value": f"RM {winner['total']:,.0f}",
     "note": f"Invoice RM {winner['Invoice']:,.0f} + RM {winner['extras']:,.0f} landing"},
    {"label": f"Saving vs {runner_up['name']}", "value": f"RM {gap:,.0f}",
     "note": f"{gap / runner_up['total'] * 100:.1f}% below the next-cheapest quote" if runner_up["total"] else ""},
])

# ---- Stacked bar: what each landed cost is made of ---------------------------------
order = list(reversed(ranked))  # plotly draws the first category at the bottom
labels = [f"{q['name']} ({q['currency']})" for q in order]
fig = go.Figure()
for part, colour in COST_PARTS:
    fig.add_trace(go.Bar(
        y=labels, x=[q[part] for q in order], name=part, orientation="h",
        marker=dict(color=colour, line=dict(color=ui.CARD, width=2)),
        hovertemplate="%{y}<br>" + part + ": RM %{x:,.2f}<extra></extra>",
    ))
fig.add_trace(go.Scatter(
    y=labels, x=[q["total"] for q in order], mode="text", showlegend=False, hoverinfo="skip",
    text=[f"  RM {q['total']:,.0f}" for q in order], textposition="middle right",
    textfont=dict(family=ui.FONT_MONO, size=13, color=ui.INK),
))
ui.style_figure(fig, height=120 + 62 * len(order))
fig.update_layout(barmode="stack", bargap=0.42)
fig.update_xaxes(showgrid=True, tickprefix="RM ", tickformat=",.0f",
                 range=[0, max(q["total"] for q in order) * 1.16 or 1])
fig.update_yaxes(showgrid=False, tickfont=dict(family=ui.FONT_SANS, size=13, color=ui.INK))
ui.chart(fig, key="cmp_chart")

# ---- How much FX movement would change the answer? ----------------------------------
st.subheader("Would a currency move change the answer?")
findings = []
for rival in ranked[1:]:
    for moving, fixed, flips_when in ((winner, rival, "strengthens"), (rival, winner, "weakens")):
        if moving["currency"] == "MYR" or moving["amount"] <= 0:
            continue
        tie_rate = (fixed["total"] - moving["extras"]) / moving["amount"]
        if tie_rate <= 0:
            continue
        move = (tie_rate / moving["rate"] - 1) * 100
        inside = None
        if moving["low"] is not None:
            inside = moving["low"] <= tie_rate <= moving["high"]
        if inside is True:
            verdict = "That is <b>inside</b> the 80% range, so this ranking is not safe."
        elif inside is False:
            verdict = "That is <b>outside</b> the 80% range, so this ranking looks robust."
        else:
            verdict = "No historical range is available for this currency to judge how likely that is."
        findings.append((abs(move), (
            f"<b>{ui.esc(winner['name'])}</b> stays ahead of <b>{ui.esc(rival['name'])}</b> unless "
            f"{moving['currency']} {flips_when} by more than <b>{abs(move):.1f}%</b> against the ringgit "
            f"(to {format_rate(tie_rate, moving['currency'])}). {verdict}"
        ), inside))
if findings:
    findings.sort(key=lambda item: item[0])
    for _, text, inside in findings[:3]:
        ui.note(text, tag="Break-even rate", kind="caution" if inside else "insight")
else:
    st.caption("All quotes are in ringgit, so exchange rates do not affect this comparison.")

ui.note(
    "This compares cost only. It says nothing about quality, lead time, reliability or payment terms.",
    kind="",
)

with st.expander("Full breakdown"):
    rows = "".join(
        f'<tr class="{"lead" if q is winner else ""}"><td>{ui.esc(q["name"])}'
        f'<span class="sl-verdict">{ui.esc(q["country"])}</span></td>'
        f'<td class="num">{q["currency"]} {q["amount"]:,.2f}</td>'
        f'<td class="num">{format_rate(q["rate"], q["currency"]) if q["currency"] != "MYR" else "—"}'
        f'<span class="sl-verdict">{ui.esc(q["basis"])}</span></td>'
        f'<td class="num">{q["Invoice"]:,.2f}</td><td class="num">{q["Freight"]:,.2f}</td>'
        f'<td class="num">{q["Insurance"]:,.2f}</td><td class="num">{q["Duties & taxes"]:,.2f}</td>'
        f'<td class="num"><b>{q["total"]:,.2f}</b></td></tr>'
        for q in ranked
    )
    ui.html(
        '<div class="sl-scroll"><table class="sl-table"><thead><tr><th>Supplier</th><th class="num">Quote</th>'
        '<th class="num">Rate used</th><th class="num">Invoice (RM)</th><th class="num">Freight</th>'
        '<th class="num">Insurance</th><th class="num">Duties</th><th class="num">Landed (RM)</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )

known = {(s["name"], s["currency"]) for s in list_suppliers()}
new = [q for q in quotes if (q["name"], q["currency"]) not in known]
save_label = ("All three are in your directory" if not new else
              "Save this supplier to my directory" if len(new) == 1 else
              f"Save these {len(new)} suppliers to my directory")
if ui.button(save_label,
             disabled=not new):
    for q in new:
        save_supplier(q["name"], q["country"] or "Unknown", q["currency"])
    st.toast("Suppliers saved.")
    st.rerun()

ui.footer()
