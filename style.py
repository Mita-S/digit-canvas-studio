"""
style.py
--------
Shared CSS + tiny layout helpers so the Draw and Admin pages look like one
app instead of two independently-styled scripts.
"""

from typing import Optional

import streamlit as st

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    h1.app-title { font-size: 2.1rem; font-weight: 800; margin-bottom: 0.1rem; }
    p.app-subtitle { color: #64748b; font-size: 1.02rem; margin-top: 0; margin-bottom: 1.6rem; }

    .stage-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px 14px 10px 14px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .stage-card h4 { margin: 0 0 8px 0; font-size: 0.92rem; color: #334155; font-weight: 700; }

    .final-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }

    .metric-pill {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        border-radius: 999px;
        padding: 3px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0 6px 6px 0;
    }

    .prediction-card {
        background: linear-gradient(180deg, #ecfdf5 0%, #f0fdf4 100%);
        border: 1px solid #a7f3d0;
        border-radius: 16px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .prediction-card .pred-label {
        font-size: 0.78rem; font-weight: 700; color: #047857;
        text-transform: uppercase; letter-spacing: 0.06em; margin: 0;
    }
    .prediction-card .pred-digit {
        font-size: 3.2rem; font-weight: 800; color: #065f46;
        line-height: 1.05; margin: 2px 0 0 0;
    }
    .prediction-card .pred-conf { font-size: 0.85rem; color: #047857; margin: 0; }
    /* Low-confidence variant -- amber instead of green, so a shaky guess
       doesn't read as authoritative. */
    .prediction-card.unsure {
        background: linear-gradient(180deg, #fffbeb 0%, #fefce8 100%);
        border-color: #fde68a;
    }
    .prediction-card.unsure .pred-label,
    .prediction-card.unsure .pred-conf { color: #b45309; }
    .prediction-card.unsure .pred-digit { color: #92400e; }

    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
    section[data-testid="stSidebar"] { border-right: 1px solid #e2e8f0; }
</style>
"""


def inject() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<h1 class="app-title">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="app-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def image_card(
    data_uri: str,
    title: Optional[str] = None,
    css_class: str = "stage-card",
    alt: str = "",
    stretch: bool = True,
    pixelated: bool = False,
) -> str:
    """One self-contained card -- optional heading plus the image -- as a
    single HTML string.

    Streamlit sanitizes every st.markdown call on its own, so an opening
    <div> in one call and a closing </div> in another never wrap anything:
    the div gets auto-closed where it opened and renders as an empty box,
    with the st.image below it sitting outside the card entirely. Emitting
    the heading, the border and the picture together (the image inlined as a
    data: URI) is what makes the card actually enclose its contents.
    """
    heading = f"<h4>{title}</h4>" if title else ""
    width = "width:100%;" if stretch else "max-width:100%;"
    rendering = "image-rendering:pixelated;" if pixelated else ""
    return (
        f'<div class="{css_class}">{heading}'
        f'<img src="{data_uri}" alt="{alt}" '
        f'style="{width}{rendering}display:block;margin:0 auto;border-radius:8px" />'
        f"</div>"
    )


def pills(values: dict) -> str:
    """Render a dict of {label: count} as a row of rounded pill badges."""
    return "".join(f'<span class="metric-pill">{k}: {v}</span>' for k, v in values.items())
