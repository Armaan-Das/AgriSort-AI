"""History page - table of all saved inspections."""

import streamlit as st

from components.ui import footer, page_header
from core.database import clear_analyses, load_analyses

page_header("History", "Every inspection saved to the SQLite database, newest first.")

df = load_analyses()

if df.empty:
    st.info("No inspections saved yet. Use the Analyse Apple page to add one.",
            icon=":material/inbox:")
    footer()
    st.stop()

table = df.copy()
table["predicted_class"] = table["predicted_class"].fillna("Not classified (saved before model)")
for col in ("prob_Good", "prob_Rotten"):
    if col not in table:
        table[col] = None

st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_order=["id", "created_at", "filename", "predicted_class", "confidence", "prob_Good",
                  "prob_Rotten", "inference_ms", "brightness", "contrast", "sharpness", "width",
                  "height", "model_version"],
    column_config={
        "id": st.column_config.NumberColumn("ID", width="small"),
        "created_at": st.column_config.DatetimeColumn("Saved at", format="DD MMM YYYY, HH:mm"),
        "filename": "File",
        "predicted_class": "Prediction",
        "confidence": st.column_config.ProgressColumn("Confidence", format="percent", min_value=0,
                                                      max_value=1),
        "prob_Good": st.column_config.NumberColumn("P(Good)", format="percent"),
        "prob_Rotten": st.column_config.NumberColumn("P(Rotten)", format="percent"),
        "inference_ms": st.column_config.NumberColumn("Inference (ms)", format="%.1f"),
        "brightness": st.column_config.NumberColumn("Brightness", format="%.1f"),
        "contrast": st.column_config.NumberColumn("Contrast", format="%.1f"),
        "sharpness": st.column_config.NumberColumn("Sharpness", format="%.0f"),
        "width": "Width (px)",
        "height": "Height (px)",
        "model_version": "Model",
    },
)
st.caption("Confidence is the model's own probability estimate, not a guarantee of correctness.")

export = df.drop(columns=[c for c in df.columns if c.startswith("prob_")])
c1, c2 = st.columns([1, 1])
with c1:
    st.download_button(
        "Download as CSV",
        data=export.to_csv(index=False).encode("utf-8"),
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
