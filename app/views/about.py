"""About page - scope, technology stack and roadmap."""

import pandas as pd
import streamlit as st

from components.ui import footer, info_card, page_header
from core.config import APP_VERSION

page_header("About AgriSort AI", f"Project scope, technology and roadmap · v{APP_VERSION}")

st.markdown("#### Scope")
c1, c2 = st.columns(2)
with c1:
    info_card(
        "In scope",
        "A software-only web application that analyses uploaded apple images "
        "and classifies visible quality as Good, Damaged or Rotten.",
    )
with c2:
    info_card(
        "Out of scope",
        "Physical hardware: IoT devices, microcontrollers, sensors, conveyor "
        "belts or mechanical sorting.",
    )

st.markdown("#### Technology stack")
stack = pd.DataFrame(
    [
        ("Web interface", "Streamlit", "Multi-page website and user interaction"),
        ("Language", "Python 3.12", "Application and ML code"),
        ("Deep learning", "PyTorch", "Neural network framework (Apple MPS GPU support)"),
        ("Model", "Ultralytics YOLO11n-cls", "Image classification model v1 (trained: Good / Rotten)"),
        ("Image processing", "OpenCV", "Decoding, measurements, edge detection"),
        ("Database", "SQLite", "Local storage of analysis records"),
        ("Visualisation", "Plotly", "Interactive dashboard charts"),
    ],
    columns=["Layer", "Technology", "Purpose"],
)
st.dataframe(stack, hide_index=True, width="stretch")

st.markdown("#### Development roadmap")
roadmap = [
    ("Phase 1", "Web application, navigation, OpenCV analysis, SQLite history, dashboard", "Done"),
    ("Phase 2", "FruitNet apple dataset (CC BY 4.0): audit and manual relabelling, round 1", "Done"),
    ("Phase 3", "Train YOLO11n-cls (Good vs Rotten) and evaluate on a held-out test set", "Done"),
    ("Phase 4", "Integrate the trained model into Analyse Apple, History and Dashboard", "Done"),
    ("Phase 5", "Report genuine evaluation metrics in the app (Model Information page)", "Done"),
    ("Phase 6", "Add the Damaged class (more labelled data) and test on new photos", "Planned"),
    ("Phase 7", "Detect and grade individual apples in multi-apple photos", "Planned"),
]
for phase, task, status in roadmap:
    icon = ":material/check_circle:" if status == "Done" else ":material/radio_button_unchecked:"
    st.markdown(f"{icon} **{phase}** · {task} — *{status}*")

footer()
