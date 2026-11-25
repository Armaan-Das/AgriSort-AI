"""AgriSort AI - Streamlit entry point.

Run from the project root:
    streamlit run app/main.py

This file sets up the page, the database and the sidebar navigation.
Each page's content lives in app/views/.
"""

import streamlit as st

from components.ui import apply_global_styles
from core.classifier import is_model_available, model_status
from core.config import APP_NAME, APP_VERSION
from core.database import init_db

st.set_page_config(
    page_title=APP_NAME,
    page_icon=":material/nutrition:",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()
apply_global_styles()

pages = {
    "Overview": [
        st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
    ],
    "Quality Analysis": [
        st.Page("views/analyse.py", title="Analyse Apple", icon=":material/image_search:"),
        st.Page("views/dashboard.py", title="Dashboard", icon=":material/monitoring:"),
        st.Page("views/history.py", title="History", icon=":material/history:"),
    ],
    "Project": [
        st.Page("views/about.py", title="About", icon=":material/info:"),
    ],
}

navigation = st.navigation(pages)

with st.sidebar:
    st.markdown(f"### {APP_NAME}")
    st.caption(f"Version {APP_VERSION}")
    st.divider()
    st.markdown("**Model status**")
    if is_model_available():
        st.success(model_status(), icon=":material/check_circle:")
    else:
        st.warning(model_status(), icon=":material/pending:")

navigation.run()
