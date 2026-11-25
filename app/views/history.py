"""History page - table of all saved analyses."""

import streamlit as st

from components.ui import footer, page_header
from core.database import clear_analyses, load_analyses

page_header("History", "Every analysis saved to the SQLite database, newest first.")

df = load_analyses()

if df.empty:
    st.info("No analyses saved yet. Use the Analyse Apple page to add one.",
            icon=":material/inbox:")
    footer()
    st.stop()

table = df.copy()
table["predicted_class"] = table["predicted_class"].fillna("Pending (model not trained)")

st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_order=["id", "created_at", "filename", "width", "height",
                  "brightness", "contrast", "sharpness", "predicted_class", "confidence"],
    column_config={
        "id": st.column_config.NumberColumn("ID", width="small"),
        "created_at": st.column_config.DatetimeColumn("Saved at", format="DD MMM YYYY, HH:mm"),
        "filename": "File",
        "width": "Width (px)",
        "height": "Height (px)",
        "brightness": st.column_config.NumberColumn("Brightness", format="%.1f"),
        "contrast": st.column_config.NumberColumn("Contrast", format="%.1f"),
        "sharpness": st.column_config.NumberColumn("Sharpness", format="%.0f"),
        "predicted_class": "Predicted class",
        "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
    },
)

c1, c2 = st.columns([1, 1])
with c1:
    st.download_button(
        "Download as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="agrisort_history.csv",
        mime="text/csv",
        icon=":material/download:",
    )
with c2:
    with st.popover("Clear history", icon=":material/delete:"):
        st.write("This permanently deletes all saved records.")
        if st.button("Yes, delete everything", type="primary"):
            clear_analyses()
            st.session_state.pop("saved_upload_ids", None)
            st.rerun()

footer()
