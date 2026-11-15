import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.supplylens import ui
from src.supplylens.business import gross_margin, invoice_cost
from src.supplylens.ui_helpers import (
    cached_rates, currency_label, format_rate, get_forecast, horizon_label, priority_currencies,
)

ui.masthead(
    "Product margin",
    "How much rate movement can my margin absorb?",
    "Describe one finished unit: what the imported material costs in its own currency, your local costs, "
    "and your selling price. See the margin now, at the plan rate, and across a range of rates.",
)

try:
    rates = cached_rates()
except Exception as exc:
    ui.note(f"Rates could not be loaded: {ui.esc(exc)}", tag="Rates unavailable", kind="caution")
    st.stop()

codes = priority_currencies(rates)
inputs, outputs = st.columns([5, 7], gap="large")

with inputs:
    st.subheader("One finished unit")
    currency = st.selectbox("Imported material is priced in", codes, format_func=currency_label)
    foreign_material = st.number_input(f"Imported material ({currency})", min_value=0.0, value=8.0, step=0.5)
    labour = st.number_input("Local labour (MYR)", min_value=0.0, value=15.0, step=1.0)
    overhead = st.number_input("Manufacturing overhead (MYR)", min_value=0.0, value=20.0, step=1.0)
    packaging = st.number_input("Packaging and other local costs (MYR)", min_value=0.0, value=5.0, step=1.0)
    selling_price = st.number_input("Selling price (MYR)", min_value=0.01, value=100.0, step=1.0)
    target = st.slider("Minimum acceptable gross margin", 0, 60, 20, 1, format="%d%%")
    horizon = ui.horizon_picker("Plan for costs in", key="margin_horizon")

result = get_forecast(rates, currency)
cp = result.checkpoint(horizon)
current_rate, plan_rate = result.current_rate, cp["rate"]
local_costs = labour + overhead + packaging


def margin_at(rate):
    return gross_margin(selling_price, invoice_cost(foreign_material, rate), local_costs)


now, plan = margin_at(current_rate), margin_at(plan_rate)
change_pp = plan["gross_margin_pct"] - now["gross_margin_pct"]
# Rate at which the margin falls to the target / to zero.
if foreign_material > 0:
    target_rate = (selling_price * (1 - target / 100) - local_costs) / foreign_material
    breakeven_rate = (selling_price - local_costs) / foreign_material
else:
    target_rate = breakeven_rate = None

with outputs:
    st.subheader("Gross margin")
    headroom_tile = {"label": f"Headroom to {target}% margin", "value": "n/a",
                     "note": "No imported cost entered"}
    if target_rate is not None:
        headroom = (target_rate / current_rate - 1) * 100
        if target_rate <= 0 or headroom < 0:
            headroom_tile = {
                "label": f"Headroom to {target}% margin", "value": "None", "kind": "warn",
                "note": f"Already below {target}% at today's rate",
            }
        else:
            headroom_tile = {
                "label": f"Headroom to {target}% margin", "value": f"+{headroom:.1f}%",
                "kind": "accent",
                "note": f"{currency} can rise to {format_rate(target_rate, currency)} before margin drops below {target}%",
            }
    ui.tiles([
        {"label": "Today", "value": f"{now['gross_margin_pct']:.1f}%",
         "note": f"at {format_rate(current_rate, currency)}"},
        {"label": f"Plan · {horizon_label(horizon)}", "value": f"{plan['gross_margin_pct']:.1f}%",
         "delta": ui.delta_text(change_pp, " pp"), "tone": ui.cost_tone(-change_pp),
         "note": f"at {format_rate(plan_rate, currency)} · {ui.esc(cp['method'])}"},
        headroom_tile,
    ])

    # ---- Where each ringgit of the selling price goes -------------------------------
    imported = invoice_cost(foreign_material, plan_rate)
    parts = [("Imported material", imported, ui.BLUE), ("Labour", labour, ui.ORANGE),
             ("Overhead", overhead, ui.TEAL), ("Packaging & other", packaging, ui.VIOLET)]
    profit = plan["gross_profit"]
    bar = go.Figure()
    for label, value, colour in parts:
        bar.add_trace(go.Bar(
            x=[value], y=["Unit"], orientation="h", name=label,
            marker=dict(color=colour, line=dict(color=ui.CARD, width=2)),
            text=[f"{value:,.2f}"] if value / max(selling_price, 1e-9) > 0.07 else None,
            textposition="inside", insidetextanchor="middle",
            textfont=dict(family=ui.FONT_MONO, color="#FFFFFF", size=12),
            hovertemplate=f"{label}: RM %{{x:,.2f}}<extra></extra>",
        ))
    if profit > 0:
        bar.add_trace(go.Bar(
            x=[profit], y=["Unit"], orientation="h", name="Gross profit",
            marker=dict(color="#DCD5C2", line=dict(color=ui.CARD, width=2)),
            text=[f"{profit:,.2f}"] if profit / selling_price > 0.07 else None,
            textposition="inside", insidetextanchor="middle",
            textfont=dict(family=ui.FONT_MONO, color=ui.INK, size=12),
            hovertemplate="Gross profit: RM %{x:,.2f}<extra></extra>",
        ))
    ui.style_figure(bar, height=170)
    bar.update_layout(barmode="stack", bargap=0.25)
    bar.update_xaxes(tickprefix="RM ", showgrid=False)
    bar.update_yaxes(visible=False)
    if profit > 0:
        bar.add_vline(x=selling_price, line=dict(color=ui.INK, width=1))
    st.markdown(f"**Where RM {selling_price:,.2f} of selling price goes, at the plan rate**")
    ui.chart(bar, key="margin_bar")
    if profit <= 0:
        ui.note(f"At the plan rate this unit loses RM {abs(profit):,.2f}: costs exceed the selling price.",
                tag="Loss-making", kind="caution")

    # ---- Margin across a range of rates ---------------------------------------------
    spread = 12.0
    if cp["low"] is not None:
        spread = max(spread, abs(cp["high"] / current_rate - 1) * 130, abs(cp["low"] / current_rate - 1) * 130)
    moves = np.linspace(-spread, spread, 97)
    margins = [margin_at(current_rate * (1 + m / 100))["gross_margin_pct"] for m in moves]

    fig = go.Figure()
    if cp["low"] is not None:
        fig.add_vrect(
            x0=(cp["low"] / current_rate - 1) * 100, x1=(cp["high"] / current_rate - 1) * 100,
            fillcolor="rgba(200,98,10,0.13)", line_width=0, layer="below",
            annotation_text="80% range", annotation_position="top left",
            annotation_font=dict(family=ui.FONT_MONO, size=11, color=ui.WARN),
        )
    fig.add_trace(go.Scatter(
        x=moves, y=margins, mode="lines", name="Gross margin", line=dict(color=ui.BLUE, width=2),
        hovertemplate=f"{currency} %{{x:+.1f}}% vs today<br>Margin %{{y:.1f}}%<extra></extra>",
    ))
    fig.add_hline(y=target, line=dict(color=ui.BAD, width=1.2, dash="dash"),
                  annotation_text=f"your minimum {target}%", annotation_position="bottom left",
                  annotation_font=dict(family=ui.FONT_MONO, size=11, color=ui.BAD))
    fig.add_trace(go.Scatter(
        x=[0], y=[now["gross_margin_pct"]], mode="markers", name="Today",
        marker=dict(color=ui.BLUE, size=12, line=dict(color=ui.CARD, width=2)),
        hovertemplate="Today<br>Margin %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[(plan_rate / current_rate - 1) * 100], y=[plan["gross_margin_pct"]], mode="markers", name="Plan",
        marker=dict(color=ui.ORANGE, size=11, symbol="diamond", line=dict(color=ui.CARD, width=2)),
        hovertemplate="Plan rate<br>Margin %{y:.1f}%<extra></extra>",
    ))
    st.markdown("**Gross margin as the rate moves**")
    ui.style_figure(fig, height=330)
    fig.update_xaxes(title_text=f"{currency} rate vs today (higher = more ringgit per {currency})",
                     ticksuffix="%", showgrid=False, zeroline=False)
    fig.update_yaxes(title_text="Gross margin", ticksuffix="%")
    ui.chart(fig, key="margin_curve")

    if cp["low"] is not None and foreign_material > 0:
        worst = min(margin_at(cp["low"])["gross_margin_pct"], margin_at(cp["high"])["gross_margin_pct"])
        best = max(margin_at(cp["low"])["gross_margin_pct"], margin_at(cp["high"])["gross_margin_pct"])
        safe = worst >= target
        ui.note(
            f"Across the 80% range for {horizon_label(horizon)} out, margin runs from <b>{worst:.1f}%</b> to "
            f"<b>{best:.1f}%</b>. "
            + (f"Even the weak end stays above your {target}% minimum."
               if safe else f"The weak end falls below your {target}% minimum; consider a price or sourcing buffer."),
            tag="What the range means for you", kind="insight" if safe else "caution",
        )
    elif not result.ok:
        ui.note(f"No trained model for {currency}, so the plan rate is today's rate and no range is drawn.",
                tag="No forecast", kind="caution")
    if breakeven_rate is not None and breakeven_rate > 0:
        st.caption(
            f"Break-even: gross profit reaches zero at {format_rate(breakeven_rate, currency)} MYR per {currency} "
            f"({(breakeven_rate / current_rate - 1) * 100:+.0f}% from today)."
        )

ui.footer("Gross margin here excludes financing, tax, discounts and wastage.")
