"""Reusable UI building blocks so every page looks consistent."""

from __future__ import annotations

import html

import streamlit as st

from core.config import CLASS_COLOURS

_CSS = """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    .as-hero {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 55%, #558B2F 100%);
        border-radius: 16px;
        padding: 2.2rem 2.4rem;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 24px rgba(27, 94, 32, 0.18);
    }
    .as-hero h1 { color: #FFFFFF; margin: 0 0 0.35rem 0; font-size: 2.3rem; font-weight: 700; }
    .as-hero p  { color: #E8F5E9; margin: 0; font-size: 1.08rem; }
    .as-hero .as-badge {
        display: inline-block; margin-top: 1rem; padding: 0.25rem 0.75rem;
        border-radius: 999px; background: rgba(255,255,255,0.16);
        font-size: 0.82rem; letter-spacing: 0.02em;
    }

    .as-page-header { margin-bottom: 1.25rem; }
    .as-page-header h2 { margin-bottom: 0.15rem; }
    .as-page-header p  { color: #5B6B5E; margin: 0; }

    .as-card {
        background: #FFFFFF; border: 1px solid #E1E8DC; border-radius: 12px;
        padding: 1.1rem 1.2rem; height: 100%;
    }
    .as-card h4 { margin: 0 0 0.35rem 0; }
    .as-card p  { margin: 0; color: #4A5A4D; font-size: 0.95rem; }
    .as-dot {
        display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 50%;
        margin-right: 0.45rem; vertical-align: middle;
    }

    .as-status {
        border-radius: 10px; padding: 0.9rem 1rem; border: 1px dashed #B0BEC5;
        background: #F5F7F5; color: #37474F;
    }

    .as-footer { color: #8A978C; font-size: 0.8rem; text-align: center; margin-top: 3rem; }

    .as-result {
        border-radius: 14px; padding: 1.2rem 1.4rem; color: #FFFFFF; margin-bottom: 0.8rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.10);
    }
    .as-result .as-result-label { font-size: 0.85rem; opacity: 0.9; letter-spacing: 0.04em; }
    .as-result .as-result-class { font-size: 2.1rem; font-weight: 700; line-height: 1.2; }
    .as-result .as-result-conf  { font-size: 1rem; opacity: 0.95; }
</style>
"""

_RESULT_ICONS = {"Good": "✔", "Rotten": "✖", "Damaged": "!"}


def apply_global_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, badge: str | None = None) -> None:
    badge_html = f'<span class="as-badge">{html.escape(badge)}</span>' if badge else ""
    st.markdown(
        f'<div class="as-hero"><h1>{html.escape(title)}</h1>'
        f"<p>{html.escape(subtitle)}</p>{badge_html}</div>",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="as-page-header"><h2>{html.escape(title)}</h2>'
        f"<p>{html.escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str, colour: str | None = None) -> None:
    dot = f'<span class="as-dot" style="background:{colour}"></span>' if colour else ""
    st.markdown(
        f'<div class="as-card"><h4>{dot}{html.escape(title)}</h4>'
        f"<p>{html.escape(body)}</p></div>",
        unsafe_allow_html=True,
    )


def class_card(name: str, description: str) -> None:
    info_card(name, description, CLASS_COLOURS.get(name))


def prediction_card(predicted_class: str, confidence: float) -> None:
    """Large coloured result banner: green for Good, red for Rotten."""
    colour = CLASS_COLOURS.get(predicted_class, "#455A64")
    icon = _RESULT_ICONS.get(predicted_class, "")
    st.markdown(
        f'<div class="as-result" style="background:{colour}">'
        f'<div class="as-result-label">PREDICTED QUALITY</div>'
        f'<div class="as-result-class">{icon} {html.escape(predicted_class)}</div>'
        f'<div class="as-result-conf">Model confidence: {confidence:.1%}</div></div>',
        unsafe_allow_html=True,
    )


def status_box(text: str) -> None:
    st.markdown(f'<div class="as-status">{html.escape(text)}</div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        '<div class="as-footer">AgriSort AI · Academic engineering prototype · '
        "Software-only apple quality analysis</div>",
        unsafe_allow_html=True,
    )
