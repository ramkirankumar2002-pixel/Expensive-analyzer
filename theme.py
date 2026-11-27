# theme.py - the look of the app (colours, emojis, CSS, card HTML, chart styling).
# UI only: no data logic in here. Every chart and table takes its category colour/emoji from
# CATEGORY_STYLE below, so a category looks the same everywhere.

import html

ACCENT = "#2DD4BF"     # the one accent colour (teal)
GREEN, RED, AMBER = "#34D399", "#F87171", "#FBBF24"
MUTED = "#8B95A9"
CARD_BG = "#141A2A"

# category -> (emoji, colour)
CATEGORY_STYLE = {
    "Food":          ("🍔", "#FB923C"),
    "Groceries":     ("🛒", "#4ADE80"),
    "Travel":        ("🚕", "#FACC15"),
    "Bills":         ("💡", "#60A5FA"),
    "Rent":          ("🏠", "#A78BFA"),
    "Shopping":      ("🛍️", "#F472B6"),
    "Entertainment": ("🎬", "#F87171"),
    "Health":        ("💊", "#2DD4BF"),
    "Transfers":     ("🔁", "#94A3B8"),
    "Others":        ("📦", "#64748B"),
    "Income":        ("💰", "#34D399"),
    "Refund":        ("↩️", "#38BDF8"),
    "Self transfer": ("🔄", "#CBD5E1"),
}


def label(category):
    """'Food' -> '🍔 Food' (unknown names are returned as they are)."""
    emoji = CATEGORY_STYLE.get(category, ("", ""))[0]
    return f"{emoji} {category}".strip()


# Plotly colour map keyed by the *labelled* name, e.g. {'🍔 Food': '#FB923C'}
COLOR_MAP = {label(c): colour for c, (_, colour) in CATEGORY_STYLE.items()}


def unlabel(text):
    """'🍔 Food' -> 'Food' (undo label(), used after the user edits a category in the table)."""
    for category in CATEGORY_STYLE:
        if label(category) == text:
            return category
    return text


def label_column(series):
    """Add the emoji to a whole column of category names (for read-only tables and charts)."""
    return series.map(label)


CSS = f"""
<style>
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }}
#MainMenu, footer {{ visibility: hidden; }}
h1 {{ font-weight: 700; letter-spacing: -0.5px; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: #0E1320; border-right: 1px solid #1B2236; }}
.brand {{ display:flex; align-items:center; gap:12px; padding: 4px 4px 18px 4px; }}
.brand .logo {{ font-size: 1.6rem; background: linear-gradient(135deg, {ACCENT}33, {ACCENT}11);
               border: 1px solid {ACCENT}55; border-radius: 14px; padding: 6px 10px; }}
.brand .name {{ font-weight: 700; font-size: 1.1rem; line-height: 1.1; }}
.brand .tag {{ color: {MUTED}; font-size: 0.75rem; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 4px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding: 8px 12px; border-radius: 12px; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ background: {ACCENT}1F; }}
.sidebar-spacer {{ height: max(24px, calc(100vh - 684px)); }}

/* summary cards */
.card {{ background: {CARD_BG}; border: 1px solid #1F2740; border-radius: 18px; padding: 16px 18px;
        height: 100%; }}
.card .top {{ display:flex; align-items:center; gap:10px; color:{MUTED}; font-size:0.82rem; }}
.card .icon {{ background:{ACCENT}1A; border-radius:10px; padding:4px 8px; font-size:1rem; }}
.card .value {{ font-size: 1.7rem; font-weight: 700; margin: 10px 0 4px 0; letter-spacing:-0.5px; }}
.card .delta {{ font-size: 0.78rem; font-weight: 600; }}
.card .delta.up {{ color:{GREEN}; }} .card .delta.down {{ color:{RED}; }} .card .delta.flat {{ color:{MUTED}; }}

/* section headers */
.section {{ margin: 28px 0 8px 0; }}
.section .title {{ font-size: 1.15rem; font-weight: 650; }}
.section .sub {{ color:{MUTED}; font-size: 0.82rem; }}

/* chips (small info pills) */
.chips {{ display:flex; flex-wrap:wrap; gap:8px; margin: 6px 0 14px 0; }}
.chip {{ background:{CARD_BG}; border:1px solid #1F2740; border-radius:999px; padding:5px 12px; font-size:0.82rem; }}
.chip b {{ color:{ACCENT}; }}

/* budget rows */
.budget {{ background:{CARD_BG}; border:1px solid #1F2740; border-radius:14px; padding:12px 16px; margin-bottom:10px; }}
.budget .row {{ display:flex; justify-content:space-between; font-size:0.9rem; margin-bottom:8px; }}
.budget .bar {{ height:8px; border-radius:99px; background:#1F2740; overflow:hidden; }}
.budget .fill {{ height:100%; border-radius:99px; }}

/* goal cards */
.goal .row {{ align-items:center; }}
.goal .badge {{ font-size:0.78rem; font-weight:700; border-radius:99px; padding:3px 12px; border:1px solid; }}
.goal .progress {{ color:{MUTED}; font-size:0.82rem; margin-top:6px; }}
.goal .chips {{ margin: 10px 0 0 0; }}

/* empty state */
.hero {{ text-align:center; background:{CARD_BG}; border:1px dashed {ACCENT}66; border-radius:24px;
        padding: 48px 24px; margin-top: 12px; }}
.hero .big {{ font-size: 3.2rem; }}
.hero h2 {{ margin: 8px 0 4px 0; }}
.hero .sub {{ color:{MUTED}; }}
.steps {{ display:flex; gap:14px; justify-content:center; flex-wrap:wrap; margin-top:26px; }}
.step {{ background:#0B0F19; border:1px solid #1F2740; border-radius:16px; padding:14px 18px; width:230px; text-align:left; }}
.step .n {{ color:{ACCENT}; font-weight:700; }}

/* rounded inputs, buttons, tables */
.stButton > button {{ border-radius: 12px; }}
[data-testid="stExpander"] {{ border-radius: 14px; }}
[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}
</style>
"""


def metric_card(icon, title, value, delta=None, unit="%", invert=False, note="vs last month"):
    """Rounded card: icon + label, big value, small change (green up / red down).
    delta=None -> no change shown. invert=True means 'up is bad' (spending), so up is red."""
    if delta is None:
        change = f'<div class="delta flat">&nbsp;</div>'
    else:
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        good = (delta < 0) if invert else (delta > 0)
        css = "flat" if delta == 0 else ("up" if good else "down")
        text = f"{arrow} {abs(delta):.1f}{unit} {note}"
        change = f'<div class="delta {css}">{text}</div>'
    return (f'<div class="card"><div class="top"><span class="icon">{icon}</span>{html.escape(title)}</div>'
            f'<div class="value">{html.escape(str(value))}</div>{change}</div>')


def section(title, sub=""):
    """A section header with an optional grey sub-line."""
    sub_html = f'<div class="sub">{html.escape(sub)}</div>' if sub else ""
    return f'<div class="section"><div class="title">{html.escape(title)}</div>{sub_html}</div>'


def chips(items):
    """A row of small pills; items = [(label, value), ...]."""
    body = "".join(f'<span class="chip">{html.escape(a)} <b>{html.escape(b)}</b></span>' for a, b in items)
    return f'<div class="chips">{body}</div>'


def budget_row(category, spent_text, budget_text, pct, status):
    """One budget line with a coloured progress bar (green ok / amber warning / red over)."""
    colour = {"ok": GREEN, "warning": AMBER, "over": RED}[status]
    width = min(pct, 100)
    return (f'<div class="budget"><div class="row"><span>{html.escape(label(category))}</span>'
            f'<span>{html.escape(spent_text)} of {html.escape(budget_text)} ({pct:.0f}%)</span></div>'
            f'<div class="bar"><div class="fill" style="width:{width}%;background:{colour}"></div></div></div>')


# goal status -> colour
GOAL_COLOURS = {"On track": GREEN, "Slightly behind": AMBER, "Off track": RED}


def goal_card(name, status, progress, amount_text, facts):
    """One savings goal: name, coloured status badge, progress bar and a row of facts.
    progress is 0 to 1; facts = [(label, value), ...]. The goal name is typed by the user, so it is escaped."""
    colour = GOAL_COLOURS[status]
    width = min(max(progress, 0), 1) * 100
    return (f'<div class="budget goal"><div class="row"><span><b>🎯 {html.escape(name)}</b></span>'
            f'<span class="badge" style="color:{colour};border-color:{colour}">{status}</span></div>'
            f'<div class="bar"><div class="fill" style="width:{width:.1f}%;background:{colour}"></div></div>'
            f'<div class="progress">{html.escape(amount_text)} saved ({width:.0f}%)</div>{chips(facts)}</div>')


def style_fig(fig, height=360):
    """Make a Plotly figure match the theme: transparent, no gridlines, theme font and colours."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
        font=dict(family="Source Sans, sans-serif", color="#E6EAF2"),
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(font=dict(size=16)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1B2236", font_color="#E6EAF2", bordercolor="#2A3350"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False)
    return fig
