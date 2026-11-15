"""MYR SupplyLens entry point: page chrome and navigation only.

Run with:  streamlit run app.py
Each screen lives in pages/. They are registered here (rather than discovered
from the folder) so the sidebar shows real names, icons and groups.
"""
import streamlit as st

from src.supplylens import ui
from src.supplylens.database import init_db
from src.supplylens.ui_helpers import cached_rates, trained_currencies

st.set_page_config(
    page_title="MYR SupplyLens",
    page_icon=str(ui.ASSETS / "icon.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)
if ui.apply_theme():
    st.rerun()  # the first run after server start picks the theme up immediately
init_db()
ui.inject_css()
st.logo(str(ui.ASSETS / "logo.svg"), size="large", icon_image=str(ui.ASSETS / "icon.svg"))

navigation = st.navigation({
    "": [
        st.Page("pages/00_Overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
    ],
    "Plan a purchase": [
        st.Page("pages/02_Procurement_Cost_Simulator.py", title="Invoice cost", icon=":material/receipt_long:"),
        st.Page("pages/03_Supplier_Comparison.py", title="Compare suppliers", icon=":material/balance:"),
        st.Page("pages/04_Profitability_Simulator.py", title="Product margin", icon=":material/savings:"),
    ],
    "Understand the forecast": [
        st.Page("pages/01_FX_Forecast_Centre.py", title="FX outlook", icon=":material/show_chart:"),
        st.Page("pages/05_Model_Insights.py", title="Model report card", icon=":material/fact_check:"),
    ],
})

try:
    ui.sidebar_status(cached_rates(), trained_currencies())
except Exception:
    ui.sidebar_status(None)

navigation.run()
