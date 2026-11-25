"""Home page - project overview."""

import streamlit as st

from components.ui import class_card, footer, hero, info_card
from core.config import APP_NAME, APP_TAGLINE, CLASS_DESCRIPTIONS, QUALITY_CLASSES

hero(APP_NAME, APP_TAGLINE, badge="Academic engineering project · Software-only prototype")

st.markdown("#### Project objective")
st.write(
    "AgriSort AI is a web application that analyses photographs of apples and "
    "classifies their **visible quality** using computer vision and deep learning. "
    "It is purely software-based: images are uploaded through the browser and "
    "analysed on the computer, with no physical sorting hardware involved."
)

st.markdown("#### Quality categories")
cols = st.columns(len(QUALITY_CLASSES))
for col, name in zip(cols, QUALITY_CLASSES):
    with col:
        class_card(name, CLASS_DESCRIPTIONS[name])

st.markdown("#### How it works")
steps = [
    ("1 · Upload", "The user uploads a photo of an apple through the web interface."),
    ("2 · Pre-process", "OpenCV decodes the image and measures brightness, contrast and sharpness."),
    ("3 · Classify", "A YOLO classification model (PyTorch) predicts Good, Damaged or Rotten."),
    ("4 · Record", "Results are stored in SQLite and summarised on the Plotly dashboard."),
]
cols = st.columns(len(steps))
for col, (title, body) in zip(cols, steps):
    with col:
        info_card(title, body)

st.write("")
st.info(
    "**Current stage:** the web interface, image processing and database are working. "
    "The classification model has **not been trained yet**, so step 3 is not active.",
    icon=":material/construction:",
)

c1, c2 = st.columns(2)
with c1:
    st.page_link("views/analyse.py", label="Go to Analyse Apple", icon=":material/arrow_forward:")
with c2:
    st.page_link("views/dashboard.py", label="Open Dashboard", icon=":material/arrow_forward:")

footer()
