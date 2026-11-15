import pandas as pd
import streamlit as st

from src.supplylens import ui
from src.supplylens.business import apply_user_scenario_shock, invoice_cost
from src.supplylens.database import delete_exposure, list_exposures, save_exposure
from src.supplylens.ui_helpers import (
    cached_rates, currency_label, format_rate, get_forecast, horizon_label, priority_currencies,
)

ui.masthead(
    "Invoice cost",
    "What should I budget for this invoice?",
    "Enter a supplier invoice in its own currency. You get the ringgit cost today, the cost to plan for at "
    "your payment date, and how far off that could be.",
)

try:
    rates = cached_rates()
except Exception as exc:
    ui.note(f"Rates could not be loaded: {ui.esc(exc)}", tag="Rates unavailable", kind="caution")
    st.stop()

codes = priority_currencies(rates)
form, slip = st.columns([5, 6], gap="large")

with form:
    st.subheader("The invoice")
    supplier = st.text_input("Supplier", value="", placeholder="e.g. Shenzhen Polymer Co.")
    a, b = st.columns([2, 3])
    with a:
        currency = st.selectbox("Currency", codes, format_func=currency_label)
    with b:
        amount = st.number_input(f"Invoice amount ({currency})", min_value=0.0, value=100000.0,
                                 step=1000.0, format="%.2f")
    category = st.selectbox("What is it for?", ["Raw materials", "Machinery", "Components", "Packaging", "Other"])
    horizon = ui.horizon_picker("Pay in", key="proc_horizon")
    shock = st.slider(
        "Stress test: the rate ends up this much worse than planned", 0.0, 10.0, 2.0, 0.5, format="%.1f%%",
        help="Your own what-if on top of the plan rate. It is not a prediction.",
    )

result = get_forecast(rates, currency)
cp = result.checkpoint(horizon)
current_rate, plan_rate = result.current_rate, cp["rate"]
adverse_rate = apply_user_scenario_shock(plan_rate, shock)
today_cost = invoice_cost(amount, current_rate)
plan_cost = invoice_cost(amount, plan_rate)
adverse_cost = invoice_cost(amount, adverse_rate)
low_cost = invoice_cost(amount, cp["low"]) if cp["low"] is not None else None
high_cost = invoice_cost(amount, cp["high"]) if cp["high"] is not None else None


def rm(value):
    return f"RM {value:,.2f}"


with slip:
    st.subheader("In ringgit")
    pay_date = f"≈ {cp['date']:%d %b %Y}" if cp["date"] is not None else "payment date"
    plan_delta = plan_cost - today_cost
    adverse_delta = adverse_cost - today_cost
    lines = [
        '<div class="sl-receipt">'
        f'<div class="sl-receipt-title">{ui.esc(supplier or "Supplier invoice")} · {currency} {amount:,.2f}</div>',
        '<div class="sl-line"><span class="sl-line-label">If you paid today'
        f'<span class="sl-line-sub">at {format_rate(current_rate, currency)} · BNM rate of '
        f'{result.latest_date:%d %b %Y}</span></span><span class="sl-line-value">{rm(today_cost)}</span></div>',
        '<div class="sl-line total"><span class="sl-line-label">Plan for'
        f'<span class="sl-line-sub">{horizon_label(horizon)} out, {pay_date} · at '
        f'{format_rate(plan_rate, currency)} · {ui.esc(cp["method"])}</span></span>'
        f'<span class="sl-line-value">{rm(plan_cost)}'
        f'<span class="sl-line-delta sl-delta {ui.cost_tone(plan_delta)}">'
        f'{ui.delta_text(plan_delta, prefix="RM ")} vs today</span></span></div>',
    ]
    if low_cost is not None:
        lines.append(
            '<div class="sl-line"><span class="sl-line-label">Likely range'
            '<span class="sl-line-sub">covers 80% of this method\'s past errors</span></span>'
            f'<span class="sl-line-value">{rm(low_cost)} – {rm(high_cost)}</span></div>'
        )
    lines.append(
        f'<div class="sl-line"><span class="sl-line-label">Stress test (+{shock:.1f}%)'
        f'<span class="sl-line-sub">your what-if, at {format_rate(adverse_rate, currency)}</span></span>'
        f'<span class="sl-line-value">{rm(adverse_cost)}'
        f'<span class="sl-line-delta sl-delta {ui.cost_tone(adverse_delta)}">'
        f'{ui.delta_text(adverse_delta, prefix="RM ")} vs today</span></span></div>'
    )

    # Range bar: where today, plan and stress sit relative to the 80% range.
    points = [today_cost, plan_cost, adverse_cost] + ([low_cost, high_cost] if low_cost is not None else [])
    lo, hi = min(points), max(points)
    if amount > 0 and hi > lo:
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad

        def pos(value):
            return (value - lo) / (hi - lo) * 100

        band = ""
        if low_cost is not None:
            band = (f'<div class="sl-range-band" style="left:{pos(low_cost):.2f}%;'
                    f'width:{pos(high_cost) - pos(low_cost):.2f}%"></div>')
        lines.append(
            f'<div class="sl-range"><div class="sl-range-track"></div>{band}'
            f'<div class="sl-range-dot" style="left:{pos(today_cost):.2f}%;background:{ui.BLUE}"></div>'
            f'<div class="sl-range-dot di" style="left:{pos(plan_cost):.2f}%;background:{ui.ORANGE}"></div>'
            f'<div class="sl-range-dot sq" style="left:{pos(adverse_cost):.2f}%;background:{ui.ROSE}"></div></div>'
            '<div class="sl-key">'
            f'<span><i style="background:{ui.BLUE}"></i>Today</span>'
            f'<span><i class="di" style="background:{ui.ORANGE}"></i>Plan</span>'
            f'<span><i class="sq" style="background:{ui.ROSE}"></i>Stress test</span>'
            + ('<span><i class="band"></i>80% range</span>' if low_cost is not None else "")
            + '</div>'
        )
    lines.append('</div>')
    ui.html("".join(lines))

    if not result.ok:
        message = (f"No trained model for {currency} yet, so the plan rate is simply today's rate."
                   if result.status == "no_model"
                   else f"The {currency} model could not be used ({ui.esc(result.message)}); using today's rate.")
        ui.note(message, tag="No forecast", kind="caution")
    elif cp["is_flat"]:
        ui.note(
            f"The plan equals today's cost because, at this horizon, nothing tested beat “the rate stays where "
            f"it is” for {currency}. "
            + ("Budget for the range, not the single figure."
               if low_cost is not None else "Use the stress test to size a buffer."),
            tag="Why there is no change", kind="insight",
        )
    elif low_cost is None:
        ui.note("No historical-error range is available for this method yet. Use the stress test to size a buffer.",
                tag="No range", kind="caution")

    if ui.button("Save this invoice to my exposure book", type="primary", stretch=True,
                 disabled=amount <= 0):
        due = cp["date"].date().isoformat() if cp["date"] is not None else ""
        save_exposure(supplier.strip() or "Unnamed supplier", currency, amount, due, category)
        st.toast("Invoice saved to your exposure book.")

# ---- Exposure book ---------------------------------------------------------------
st.subheader("Exposure book")
exposures = list_exposures()
if not exposures:
    st.caption("Nothing saved yet. Saved invoices appear here, revalued at the latest reference rate.")
else:
    book = pd.DataFrame(exposures)
    book["rate"] = [
        float(rates[c].dropna().iloc[-1]) if c in rates.columns else float("nan") for c in book["currency"]
    ]
    book["myr"] = book["invoice_amount"] * book["rate"]
    totals = book.groupby("currency")["myr"].sum().sort_values(ascending=False)
    ui.tiles(
        [{"label": "Open invoices", "value": len(book)},
         {"label": "Value at today's rates", "value": f"RM {book['myr'].sum():,.0f}", "kind": "accent"}]
        + [{"label": f"{code} share", "value": f"{value / book['myr'].sum() * 100:.0f}%",
            "note": f"RM {value:,.0f}"} for code, value in totals.head(2).items() if book["myr"].sum() > 0]
    )
    view = pd.DataFrame({
        "Supplier": book["supplier_name"],
        "For": book["category"],
        "Currency": book["currency"],
        "Invoice amount": book["invoice_amount"],
        "Pay around": book["due_date"],
        "MYR today": book["myr"],
    })
    ui.table(view, column_config={
        "Invoice amount": st.column_config.NumberColumn(format="%,.2f"),
        "MYR today": st.column_config.NumberColumn(format="RM %,.2f"),
    })
    with st.popover("Remove an invoice"):
        labels = {
            int(row["id"]): f"{row['supplier_name']} · {row['currency']} {row['invoice_amount']:,.0f}"
            for row in exposures
        }
        target = st.selectbox("Invoice", list(labels), format_func=labels.get)
        if st.button("Remove", type="secondary"):
            delete_exposure(target)
            st.rerun()

ui.footer()
