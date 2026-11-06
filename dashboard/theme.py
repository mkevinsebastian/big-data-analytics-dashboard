"""Look & feel: CSS (works on both Streamlit light and dark themes) and small HTML component helpers.

Colours use translucent greys + `inherit`, so cards stay readable whichever theme the viewer picks
(Settings menu, top-right of the app, or the OS preference).
"""
from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}
.hero {padding: 1.1rem 1.4rem; border-radius: 16px; margin-bottom: 1rem;
       background: linear-gradient(120deg, rgba(99,102,241,.20), rgba(16,185,129,.12));
       border: 1px solid rgba(127,127,127,.25);}
.hero h1 {margin: 0; font-size: 1.75rem; line-height: 1.25;}
.hero p {margin: .35rem 0 0 0; opacity: .8; font-size: .95rem;}
.kpi {border: 1px solid rgba(127,127,127,.28); border-left: 5px solid var(--accent, #6366F1);
      border-radius: 14px; padding: .9rem 1.05rem; background: rgba(127,127,127,.07); height: 100%;}
.kpi .label {font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; opacity: .7;}
.kpi .value {font-size: 1.85rem; font-weight: 700; line-height: 1.2; margin: .15rem 0;}
.kpi .sub {font-size: .82rem; opacity: .75;}
.insight {border: 1px solid rgba(127,127,127,.28); border-radius: 14px; padding: .95rem 1.1rem; min-height: 190px;
          background: rgba(127,127,127,.06); height: 100%;}
.insight .tag {display: inline-block; font-size: .72rem; font-weight: 700; padding: .1rem .55rem;
               border-radius: 999px; color: #fff; background: var(--accent, #6366F1); margin-bottom: .45rem;}
.insight h4 {margin: .1rem 0 .35rem 0; font-size: 1.02rem;}
.insight p {margin: 0; font-size: .88rem; opacity: .85;}
.callout {border-radius: 12px; padding: .8rem 1rem; margin: .5rem 0;
          border: 1px solid rgba(127,127,127,.28); border-left: 5px solid var(--accent, #6366F1);
          background: rgba(127,127,127,.06); font-size: .92rem;}
.decision {border-radius: 16px; padding: 1.1rem 1.3rem; border: 2px solid var(--accent, #10B981);
           background: rgba(127,127,127,.07);}
.decision .big {font-size: 1.6rem; font-weight: 800; color: var(--accent, #10B981);}
.pill {display: inline-block; padding: .15rem .65rem; margin: .15rem .25rem .15rem 0; border-radius: 999px;
       font-size: .8rem; border: 1px solid rgba(127,127,127,.4); background: rgba(127,127,127,.1);}
div[data-testid="stTabs"] button[role="tab"] {font-size: 1rem; font-weight: 600;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _e(x) -> str:
    return html.escape(str(x))


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{_e(title)}</h1><p>{_e(subtitle)}</p></div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "", accent: str = "#6366F1") -> str:
    return (f'<div class="kpi" style="--accent:{accent}"><div class="label">{_e(label)}</div>'
            f'<div class="value">{_e(value)}</div><div class="sub">{_e(sub)}</div></div>')


def insight_card(tag: str, title: str, body: str, accent: str = "#6366F1") -> str:
    return (f'<div class="insight" style="--accent:{accent}"><span class="tag">{_e(tag)}</span>'
            f'<h4>{_e(title)}</h4><p>{_e(body)}</p></div>')


def callout(text: str, accent: str = "#6366F1") -> None:
    st.markdown(f'<div class="callout" style="--accent:{accent}">{_e(text)}</div>', unsafe_allow_html=True)


def show(html_block: str) -> None:
    st.markdown(html_block, unsafe_allow_html=True)


# ----------------------------- number formatting (Indonesian) ----------------------------- #
def _swap(s: str) -> str:
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def fmt_int(n: float) -> str:
    return _swap(f"{round(n):,}")


def fmt_usd(x: float, decimals: int = 0) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${_swap(f'{abs(x):,.{decimals}f}')}"


def fmt_pct(x: float, decimals: int = 2, already_pct: bool = False) -> str:
    v = x if already_pct else x * 100
    return _swap(f"{v:.{decimals}f}") + "%"


def fmt_num(x: float, decimals: int = 2) -> str:
    return _swap(f"{x:.{decimals}f}")


def plot(fig, container=st) -> None:
    """Render a Plotly figure with its title as a Streamlit heading.

    Streamlit's chart theme re-positions the legend at the top of the plot, where it collides with an in-chart
    title on narrow columns, so titles are drawn outside the figure.
    """
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else ""
    fig.update_layout(title=None)
    if title:
        container.markdown(f"**{title}**")
    container.plotly_chart(fig, theme="streamlit", width="stretch")
