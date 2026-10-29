"""Small, maintainable styling layer. The one deliberate visual idea: every piece of content carries a provenance
badge (verified by code / LLM-generated / mock or synthetic / reported by the paper)."""

from __future__ import annotations

import html

import streamlit as st

PROVENANCE = {
    "verified": ("Verified by code", "prov-verified"),
    "llm": ("LLM-generated", "prov-llm"),
    "mock": ("Mock stand-in, not an LLM", "prov-mock"),
    "synthetic": ("Synthetic data", "prov-mock"),
    "paper": ("Reported by the paper", "prov-paper"),
    "reference": ("Reference implementation", "prov-verified"),
    "extension": ("Extension, not in the paper", "prov-paper"),
}

CSS = """
<style>
:root { --ink:#1B2430; --muted:#5B6776; --line:#D5DCE4; --verified:#0F6E56; --verified-bg:#E3F3EE; --llm:#5B3FA6; --llm-bg:#EEE9FA;
        --mock:#8A5200; --mock-bg:#FFF1D6; --paper:#1F4E79; --paper-bg:#E4EEF7; --bad:#A31F34; --bad-bg:#FBE7EA; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
h1 { font-size: 1.85rem !important; letter-spacing: -0.01em; margin-bottom: .15rem; }
h2 { font-size: 1.3rem !important; } h3 { font-size: 1.08rem !important; }
.page-sub { color: var(--muted); font-size: 1.02rem; margin: 0 0 1.1rem 0; max-width: 62rem; line-height: 1.5; }
.prov { display:inline-block; padding:1px 9px; border-radius:999px; font-size:.78rem; font-weight:600; border:1px solid; margin: 0 6px 4px 0; white-space:nowrap; }
.prov-verified { color:var(--verified); background:var(--verified-bg); border-color:#9ED3C4; }
.prov-llm { color:var(--llm); background:var(--llm-bg); border-color:#C7B8EE; }
.prov-mock { color:var(--mock); background:var(--mock-bg); border-color:#EBC37A; }
.prov-paper { color:var(--paper); background:var(--paper-bg); border-color:#A9C4DD; }
.mode-banner { border-left:5px solid var(--mock); background:var(--mock-bg); padding:.7rem 1rem; border-radius:6px; margin:0 0 1rem 0; color:#4A2B00; line-height:1.45; }
.mode-banner.live { border-left-color:var(--llm); background:var(--llm-bg); color:#2B1B57; }
.fact-box { border:1px solid var(--line); border-left:5px solid var(--verified); border-radius:6px; padding:.7rem 1rem; background:#fff; }
.llm-box { border:1px solid var(--line); border-left:5px solid var(--llm); border-radius:6px; padding:.7rem 1rem; background:#fff; }
.llm-box.mock { border-left-color:var(--mock); }
.small-note { color:var(--muted); font-size:.88rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:6px; padding:.6rem .8rem; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def badge(kind: str, label: str | None = None) -> str:
    text, cls = PROVENANCE[kind]
    return f'<span class="prov {cls}">{html.escape(label or text)}</span>'


def provenance(*kinds: str) -> None:
    st.markdown("".join(badge(k) for k in kinds), unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.markdown(f'<p class="page-sub">{html.escape(subtitle)}</p>', unsafe_allow_html=True)


def mode_banner(info: dict | None, error: str | None = None) -> None:
    if error:
        st.markdown(f'<div class="mode-banner" style="border-left-color:#A31F34;background:#FBE7EA;color:#5A0D1B"><b>Provider not usable.</b> {html.escape(error)}</div>', unsafe_allow_html=True)
        return
    if info and info.get("is_mock"):
        st.markdown('<div class="mode-banner"><b>Offline mock mode.</b> Metric specifications and plan selections come from a deterministic rule-based '
                    'stand-in, <b>not from a language model</b>. Use it to see how the pipeline works; it says nothing about LLM accuracy.</div>', unsafe_allow_html=True)
    elif info:
        st.markdown(f'<div class="mode-banner live"><b>Live provider:</b> {html.escape(str(info.get("provider")))} / {html.escape(str(info.get("model")))}. '
                    'Runs call the provider API and may incur cost. This is not the paper\'s GPT-4o set-up.</div>', unsafe_allow_html=True)


def llm_box(text: str, is_mock: bool) -> None:
    label = badge("mock") if is_mock else badge("llm")
    cls = "llm-box mock" if is_mock else "llm-box"
    st.markdown(f'<div class="{cls}">{label}<div style="margin-top:.35rem">{html.escape(text)}</div></div>', unsafe_allow_html=True)


def fact_box(markdown_html: str) -> None:
    st.markdown(f'<div class="fact-box">{badge("verified")}<div style="margin-top:.35rem">{markdown_html}</div></div>', unsafe_allow_html=True)
