"""MYR SupplyLens design system: one place for colour, type, layout and charts.

Pages build their screens from the helpers here instead of styling themselves,
so the app reads as one product. Nothing in this module touches the model.
"""
from html import escape
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"

# ---- Palette -------------------------------------------------------------
INK = "#14213D"        # primary text, sidebar
MUTED = "#6B6557"      # secondary text
FAINT = "#9A9384"      # axis and helper text
RULE = "#E2DBC9"       # hairlines
CANVAS = "#F7F4EC"     # page
CARD = "#FFFDF8"       # raised surfaces and chart background
GOLD = "#E8B04A"       # brand accent (never used for data)
PRIMARY = "#0B6E63"    # interactive

# Categorical series, validated for colour-vision deficiency on the card surface.
BLUE, ORANGE, TEAL, VIOLET, ROSE = "#1F5FA8", "#C8620A", "#0A8F7F", "#8B4FB8", "#B8385A"
SERIES = [BLUE, ORANGE, TEAL, VIOLET, ROSE]
NEUTRAL = "#B9B2A0"

# Status colours: only ever used with a word or an arrow beside them.
GOOD, BAD, WARN = "#1D7A3A", "#C0392B", "#A8650A"

FONT_SANS = "'IBM Plex Sans', 'Segoe UI', system-ui, -apple-system, sans-serif"
FONT_SERIF = "'Fraunces', 'Iowan Old Style', Georgia, 'Times New Roman', serif"
FONT_MONO = "'IBM Plex Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp button, .stApp textarea {{
  font-family: {FONT_SANS};
}}
.stApp h1, .stApp h2, .stApp h3 {{
  font-family: {FONT_SERIF}; color: {INK}; letter-spacing: -0.01em; font-weight: 600;
}}
.stApp h2 {{ font-size: 1.45rem; padding-top: 0.6rem; }}
.stApp h3 {{ font-size: 1.12rem; }}
.block-container {{ padding-top: 2.6rem; padding-bottom: 4rem; max-width: 1240px; }}
.stApp input[type="number"], [data-testid="stMetricValue"] {{ font-family: {FONT_MONO}; font-variant-numeric: tabular-nums; }}

/* Sidebar (colours repeated here in case the theme could not be applied) */
[data-testid="stSidebar"] {{ border-right: 1px solid #0B1322; background-color: #111C30; }}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span,
[data-testid="stSidebar"] [data-testid="stNavSectionHeader"] {{ color: #EDE8DA; }}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span {{ font-size: 0.93rem; }}
[data-testid="stSidebarHeader"] img, [data-testid="stLogo"] {{ height: 3.1rem; max-width: 100%; }}

/* Inputs sit on the card colour so they read as fields on the cream page */
.stApp [data-baseweb="input"], .stApp [data-baseweb="select"] > div, .stApp [data-baseweb="base-input"] {{
  background: {CARD};
}}
[data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {{ gap: 0.7rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {RULE}; }}

/* ---- Masthead ---- */
.sl-mast {{ border-bottom: 3px double {INK}; padding-bottom: 1.05rem; margin-bottom: 1.4rem; }}
.sl-eyebrow {{ font-family: {FONT_MONO}; font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: {PRIMARY}; font-weight: 600; margin-bottom: 0.35rem; }}
.sl-title {{ font-family: {FONT_SERIF}; font-size: 2.55rem; line-height: 1.08; font-weight: 700; color: {INK};
  letter-spacing: -0.02em; margin: 0; }}
.sl-lede {{ font-size: 1.04rem; color: {MUTED}; max-width: 62ch; margin: 0.55rem 0 0 0; line-height: 1.5; }}

/* ---- Stat tiles ---- */
.sl-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.85rem; margin: 0.2rem 0 1.1rem 0; }}
.sl-tile {{ background: {CARD}; border: 1px solid {RULE}; border-top: 3px solid {INK}; border-radius: 4px;
  padding: 0.85rem 1rem 0.9rem 1rem; min-width: 0; }}
.sl-tile.accent {{ border-top-color: {PRIMARY}; }}
.sl-tile.warn {{ border-top-color: {ORANGE}; }}
.sl-tile-label {{ font-size: 0.72rem; letter-spacing: 0.09em; text-transform: uppercase; color: {MUTED}; font-weight: 600; }}
.sl-tile-value {{ font-family: {FONT_MONO}; font-size: 1.62rem; font-weight: 500; color: {INK}; line-height: 1.25;
  margin-top: 0.28rem; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }}
.sl-tile-value.small {{ font-size: 1.22rem; }}
.sl-tile-note {{ font-size: 0.8rem; color: {MUTED}; margin-top: 0.3rem; line-height: 1.35; }}
.sl-delta {{ font-family: {FONT_MONO}; font-size: 0.82rem; font-weight: 600; margin-top: 0.3rem; }}
.sl-delta.good {{ color: {GOOD}; }} .sl-delta.bad {{ color: {BAD}; }} .sl-delta.flat {{ color: {MUTED}; }}

/* ---- Rate tape ---- */
.sl-tape {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(126px, 1fr)); border: 1px solid {RULE};
  background: {CARD}; border-radius: 4px; margin-bottom: 1.3rem; overflow: hidden; }}
.sl-tape-cell {{ padding: 0.75rem 0.9rem 0.6rem 0.9rem; border-right: 1px solid {RULE}; border-bottom: 1px solid {RULE};
  margin: 0 -1px -1px 0; min-width: 0; }}
.sl-tape-code {{ font-family: {FONT_MONO}; font-weight: 600; font-size: 0.86rem; color: {INK}; letter-spacing: 0.06em; }}
.sl-tape-name {{ font-size: 0.7rem; color: {FAINT}; display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.sl-tape-rate {{ font-family: {FONT_MONO}; font-size: 1.08rem; color: {INK}; margin-top: 0.15rem; font-variant-numeric: tabular-nums; }}
.sl-tape-foot {{ display: flex; align-items: center; justify-content: space-between; gap: 0.4rem; margin-top: 0.2rem; }}
.sl-chip {{ display: inline-block; font-family: {FONT_MONO}; font-size: 0.68rem; font-weight: 600; letter-spacing: 0.05em;
  padding: 0.1rem 0.42rem; border-radius: 3px; border: 1px solid {RULE}; color: {MUTED}; background: {CANVAS}; white-space: nowrap; }}
.sl-chip.on {{ color: {PRIMARY}; border-color: {PRIMARY}; background: #E7F2EF; }}
.sl-chip.legacy {{ color: {WARN}; border-color: #D9B777; background: #FBF1DC; }}
.sl-chip.dark {{ color: {CARD}; border-color: {INK}; background: {INK}; }}

/* ---- Notes ---- */
.sl-note {{ border-left: 3px solid {INK}; background: {CARD}; border-radius: 0 4px 4px 0; padding: 0.75rem 1rem;
  color: {INK}; font-size: 0.93rem; line-height: 1.5; margin: 0.4rem 0 1rem 0; border-top: 1px solid {RULE};
  border-right: 1px solid {RULE}; border-bottom: 1px solid {RULE}; }}
.sl-note b {{ font-weight: 600; }}
.sl-note.insight {{ border-left-color: {PRIMARY}; }}
.sl-note.caution {{ border-left-color: {ORANGE}; }}
.sl-note-tag {{ font-family: {FONT_MONO}; font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
  font-weight: 600; display: block; margin-bottom: 0.2rem; color: {MUTED}; }}
.sl-small {{ font-size: 0.8rem; color: {MUTED}; line-height: 1.5; }}

/* ---- Checkpoint cards ---- */
.sl-cps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.85rem; margin-bottom: 1rem; }}
.sl-cp {{ background: {CARD}; border: 1px solid {RULE}; border-radius: 4px; padding: 0.85rem 1rem; }}
.sl-cp-head {{ display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid {RULE};
  padding-bottom: 0.4rem; margin-bottom: 0.5rem; }}
.sl-cp-h {{ font-family: {FONT_SERIF}; font-weight: 600; font-size: 1.05rem; color: {INK}; }}
.sl-cp-date {{ font-family: {FONT_MONO}; font-size: 0.72rem; color: {MUTED}; }}
.sl-cp-rate {{ font-family: {FONT_MONO}; font-size: 1.4rem; color: {INK}; font-variant-numeric: tabular-nums; }}
.sl-cp-range {{ font-family: {FONT_MONO}; font-size: 0.78rem; color: {MUTED}; margin: 0.25rem 0 0.55rem 0; }}

/* ---- Receipt ---- */
.sl-receipt {{ background: {CARD}; border: 1px solid {RULE}; border-radius: 4px; padding: 1.1rem 1.25rem 1rem 1.25rem; margin-bottom: 0.9rem; }}
.sl-receipt-title {{ font-family: {FONT_MONO}; font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase;
  color: {MUTED}; font-weight: 600; border-bottom: 1px dashed {NEUTRAL}; padding-bottom: 0.55rem; margin-bottom: 0.3rem; }}
.sl-line {{ display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; padding: 0.5rem 0;
  border-bottom: 1px dashed {RULE}; }}
.sl-line:last-child {{ border-bottom: none; }}
.sl-line-label {{ color: {INK}; font-size: 0.93rem; }}
.sl-line-sub {{ display: block; color: {MUTED}; font-size: 0.76rem; margin-top: 0.1rem; }}
.sl-line-value {{ font-family: {FONT_MONO}; font-size: 1.02rem; color: {INK}; white-space: nowrap; text-align: right;
  font-variant-numeric: tabular-nums; }}
.sl-line.total .sl-line-label {{ font-weight: 600; }}
.sl-line.total .sl-line-value {{ font-size: 1.5rem; font-weight: 500; }}
.sl-line-delta {{ display: block; font-size: 0.76rem; font-weight: 600; }}

/* ---- Range bar ---- */
.sl-range {{ position: relative; height: 34px; margin: 0.9rem 0.4rem 0.3rem 0.4rem; }}
.sl-range-track {{ position: absolute; left: 0; right: 0; top: 15px; height: 4px; background: {RULE}; border-radius: 2px; }}
.sl-range-band {{ position: absolute; top: 9px; height: 16px; background: rgba(200,98,10,0.20); border-left: 2px solid {ORANGE};
  border-right: 2px solid {ORANGE}; }}
.sl-range-dot {{ position: absolute; top: 9px; width: 16px; height: 16px; margin-left: -8px; border-radius: 50%;
  border: 2px solid {CARD}; box-sizing: border-box; }}
.sl-range-dot.sq {{ border-radius: 2px; }}
.sl-range-dot.di {{ border-radius: 2px; transform: rotate(45deg); }}
.sl-key {{ display: flex; flex-wrap: wrap; gap: 0.35rem 1.1rem; font-size: 0.78rem; color: {MUTED}; margin: 0.2rem 0 0 0; }}
.sl-key i {{ display: inline-block; width: 10px; height: 10px; margin-right: 0.35rem; border-radius: 50%; vertical-align: -1px; }}
.sl-key i.sq {{ border-radius: 2px; }} .sl-key i.di {{ border-radius: 2px; transform: rotate(45deg); }}
.sl-key i.band {{ width: 16px; border-radius: 0; background: rgba(200,98,10,0.20); border-left: 2px solid {ORANGE}; border-right: 2px solid {ORANGE}; }}

/* ---- Tables ---- */
.sl-table {{ width: 100%; border-collapse: collapse; background: {CARD}; border: 1px solid {RULE}; font-size: 0.88rem; }}
.sl-table th {{ font-size: 0.7rem; letter-spacing: 0.09em; text-transform: uppercase; color: {MUTED}; font-weight: 600;
  text-align: left; padding: 0.55rem 0.8rem; border-bottom: 2px solid {INK}; background: {CARD}; }}
.sl-table td {{ padding: 0.55rem 0.8rem; border-bottom: 1px solid {RULE}; color: {INK}; vertical-align: top; }}
.sl-table td.num, .sl-table th.num {{ text-align: right; font-family: {FONT_MONO}; font-variant-numeric: tabular-nums; }}
.sl-table tr:last-child td {{ border-bottom: none; }}
.sl-table tr.lead td {{ background: #EEF6F3; }}
.sl-verdict {{ font-size: 0.74rem; color: {MUTED}; display: block; margin-top: 0.15rem; }}
.sl-scroll {{ overflow-x: auto; }}

/* ---- Pipeline ---- */
.sl-pipe {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: 0.3rem; margin: 0.3rem 0 1rem 0; }}
.sl-pipe-step {{ flex: 1 1 120px; background: {CARD}; border: 1px solid {RULE}; border-bottom: 3px solid {INK};
  border-radius: 4px; padding: 0.6rem 0.7rem; position: relative; }}
.sl-pipe-step b {{ font-family: {FONT_MONO}; font-size: 0.8rem; color: {INK}; display: block; }}
.sl-pipe-step span {{ font-size: 0.76rem; color: {MUTED}; line-height: 1.35; display: block; margin-top: 0.15rem; }}
.sl-pipe-arrow {{ align-self: center; color: {NEUTRAL}; font-family: {FONT_MONO}; }}

.sl-foot {{ border-top: 1px solid {RULE}; margin-top: 2.6rem; padding-top: 0.9rem; font-size: 0.78rem; color: {MUTED}; line-height: 1.55; }}
.sl-foot a {{ color: {PRIMARY}; }}
.sl-side {{ font-size: 0.78rem; line-height: 1.55; color: #9FB0C9; margin-top: -0.6rem; }}
.sl-side b {{ color: #EDE8DA; font-weight: 600; font-family: {FONT_MONO}; }}
.sl-side .k {{ display: block; font-family: {FONT_MONO}; font-size: 0.66rem; letter-spacing: 0.14em; text-transform: uppercase; color: #7C8DA8; margin-top: 0.55rem; }}
</style>
"""


# Streamlit's own widgets (sliders, buttons, the sidebar) take their colours from
# the theme, not from CSS. Setting it here keeps the look in one file and means
# the app is themed even when it is launched without a .streamlit/config.toml,
# which is the case when the Colab notebook overlays only src/ and pages/.
THEME = {
    "theme.base": "light",
    "theme.primaryColor": PRIMARY,
    "theme.backgroundColor": CANVAS,
    "theme.secondaryBackgroundColor": "#EFEADD",
    "theme.textColor": INK,
    "theme.sidebar.backgroundColor": "#111C30",
    "theme.sidebar.secondaryBackgroundColor": "#1B2A45",
    "theme.sidebar.textColor": "#EDE8DA",
}


def apply_theme():
    """Apply THEME once per server process. Returns True if anything changed."""
    from streamlit import config
    changed = False
    for option, value in THEME.items():
        try:
            if config.get_option(option) != value:
                config.set_option(option, value)
                changed = True
        except Exception:
            continue  # option not known to this Streamlit version; CSS covers the essentials
    return changed


def html(markup):
    """Render trusted markup. Always escape user-entered text with ``esc`` first."""
    st.markdown(markup, unsafe_allow_html=True)


def esc(value):
    return escape(str(value), quote=True)


def inject_css():
    html(_CSS)


def masthead(eyebrow, title, lede=""):
    lede_html = f'<p class="sl-lede">{lede}</p>' if lede else ""
    html(
        f'<div class="sl-mast"><div class="sl-eyebrow">{esc(eyebrow)}</div>'
        f'<h1 class="sl-title">{esc(title)}</h1>{lede_html}</div>'
    )


def tiles(items):
    """Row of stat tiles. Each item: label, value, and optional note / delta / tone / kind / small."""
    parts = []
    for item in items:
        delta = ""
        if item.get("delta"):
            delta = f'<div class="sl-delta {item.get("tone", "flat")}">{esc(item["delta"])}</div>'
        note = f'<div class="sl-tile-note">{item["note"]}</div>' if item.get("note") else ""
        small = " small" if item.get("small") else ""
        parts.append(
            f'<div class="sl-tile {item.get("kind", "")}"><div class="sl-tile-label">{esc(item["label"])}</div>'
            f'<div class="sl-tile-value{small}">{esc(item["value"])}</div>{delta}{note}</div>'
        )
    html(f'<div class="sl-tiles">{"".join(parts)}</div>')


def note(body, tag=None, kind=""):
    tag_html = f'<span class="sl-note-tag">{esc(tag)}</span>' if tag else ""
    html(f'<div class="sl-note {kind}">{tag_html}{body}</div>')


def chip(text, kind=""):
    return f'<span class="sl-chip {kind}">{esc(text)}</span>'


def delta_text(value, unit="", decimals=2, prefix=""):
    """Signed number with an arrow, so direction never depends on colour alone."""
    if abs(value) < 0.5 * 10 ** (-decimals):
        return f"● no change"
    arrow = "▲" if value > 0 else "▼"
    return f"{arrow} {prefix}{abs(value):,.{decimals}f}{unit}"


def cost_tone(value, tolerance=0.005):
    """For costs, up is bad. (st.metric colours any increase green by default.)"""
    if abs(value) < tolerance:
        return "flat"
    return "bad" if value > 0 else "good"


def sparkline(values, width=120, height=30, colour=BLUE):
    values = [float(v) for v in values]
    if len(values) < 2:
        return ""
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    step = (width - 4) / (len(values) - 1)
    points = " ".join(
        f"{2 + i * step:.1f},{height - 3 - (v - low) / span * (height - 6):.1f}"
        for i, v in enumerate(values)
    )
    last_x, last_y = points.split(" ")[-1].split(",")
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="trend"><polyline points="{points}" fill="none" stroke="{colour}" stroke-width="1.6" '
        f'stroke-linejoin="round" stroke-linecap="round"/><circle cx="{last_x}" cy="{last_y}" r="2.6" '
        f'fill="{colour}" stroke="{CARD}" stroke-width="1.2"/></svg>'
    )


def footer(extra=""):
    from .config import DATASET_PAGE
    html(
        '<div class="sl-foot">Rates: Bank Negara Malaysia daily middle rates via '
        f'<a href="{DATASET_PAGE}" target="_blank">data.gov.my</a> (CC BY 4.0), quoted as MYR per one unit of '
        'foreign currency. These are reference rates, not the rate your bank will quote. Forecasts and ranges are '
        f'estimates for planning, not financial advice. {extra}</div>'
    )


# ---- Streamlit version shims ----------------------------------------------
def chart(fig, key=None):
    config = {"displayModeBar": False, "responsive": True}
    try:
        st.plotly_chart(fig, width="stretch", config=config, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=config, key=key)


def table(frame, **kwargs):
    kwargs.setdefault("hide_index", True)
    try:
        st.dataframe(frame, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(frame, use_container_width=True, **kwargs)


def button(label, **kwargs):
    stretch = kwargs.pop("stretch", False)
    if not stretch:
        return st.button(label, **kwargs)
    try:
        return st.button(label, width="stretch", **kwargs)
    except TypeError:
        return st.button(label, use_container_width=True, **kwargs)


def choose(label, options, default=None, format_func=str, key=None, help=None):
    """Segmented control where available, radio buttons on older Streamlit."""
    default = options[0] if default is None else default
    if hasattr(st, "segmented_control"):
        picked = st.segmented_control(
            label, options, default=default, format_func=format_func, key=key, help=help,
        )
        return default if picked is None else picked
    return st.radio(
        label, options, index=options.index(default), format_func=format_func,
        key=key, help=help, horizontal=True,
    )


def horizon_picker(label="When will you pay?", key="horizon", default=21):
    from .config import DISPLAY_HORIZONS
    from .ui_helpers import HORIZON_LABELS
    return choose(
        label, list(DISPLAY_HORIZONS), default=default, key=key,
        format_func=lambda h: HORIZON_LABELS.get(h, f"{h} sessions"),
        help="Forecasts are checked at 5, 21, 42 and 63 trading sessions ahead.",
    )


# ---- Charts ----------------------------------------------------------------
def style_figure(fig, height=380, legend=True):
    fig.update_layout(
        height=height,
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family=FONT_SANS, size=13, color=INK),
        margin=dict(l=8, r=16, t=44 if legend else 16, b=8),
        hoverlabel=dict(bgcolor=INK, font=dict(color=CARD, family=FONT_MONO, size=12), bordercolor=INK),
        showlegend=legend,
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left",
                    font=dict(size=12, color=MUTED), bgcolor="rgba(0,0,0,0)", traceorder="normal"),
        bargap=0.35,
    )
    axis = dict(
        gridcolor="#EFE9DB", zeroline=False, linecolor=RULE, ticks="", automargin=True,
        tickfont=dict(family=FONT_MONO, size=11, color=FAINT),
        title_font=dict(size=12, color=MUTED),
    )
    fig.update_xaxes(**axis, showgrid=False)
    fig.update_yaxes(**axis)
    return fig


def sidebar_status(rates=None, trained=None):
    with st.sidebar:
        if rates is None:
            html('<div class="sl-side"><span class="k">Data</span>Reference rates unavailable.</div>')
            return
        latest = rates.index.max().date()
        html(
            '<div class="sl-side">'
            f'<span class="k">Latest BNM rates</span><b>{latest:%d %b %Y}</b>'
            f'<span class="k">Trained models</span><b>{" · ".join(trained) if trained else "none yet"}</b>'
            '<span class="k">Reading the numbers</span>All rates are MYR per 1 unit of foreign currency. '
            'A higher rate means imports cost more.'
            '</div>'
        )
