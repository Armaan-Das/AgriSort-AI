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
        ("Model", "Ultralytics YOLO", "Image classification model (to be trained)"),
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
    ("Phase 2", "Collect and label apple image dataset (Good / Damaged / Rotten)", "Not started"),
    ("Phase 3", "Train YOLO classification model and evaluate on a test set", "Not started"),
    ("Phase 4", "Integrate trained model into Analyse Apple page", "Not started"),
    ("Phase 5", "Report real evaluation metrics on the dashboard", "Not started"),
]
for phase, task, status in roadmap:
    icon = ":material/check_circle:" if status == "Done" else ":material/radio_button_unchecked:"
    st.markdown(f"{icon} **{phase}** · {task} — *{status}*")

footer()
