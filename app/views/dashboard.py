"""Dashboard page - summary charts built only from real saved records."""

import plotly.express as px
import streamlit as st

from components.ui import footer, page_header, status_box
from core.classifier import model_status
from core.config import CLASS_COLOURS, QUALITY_CLASSES
from core.database import load_analyses

page_header(
    "Dashboard",
    "Live summary of analyses stored in the SQLite database. "
    "Every chart is generated from real records - nothing is simulated.",
)

df = load_analyses()
classified = df[df["predicted_class"].notna()] if not df.empty else df

k1, k2, k3, k4 = st.columns(4)
k1.metric("Images analysed", len(df), border=True)
k2.metric("Images classified", len(classified), border=True)
k3.metric(
    "Average sharpness",
    f"{df['sharpness'].mean():.0f}" if not df.empty else "-",
    border=True,
)
k4.metric("Model status", model_status(), border=True)

st.write("")
left, right = st.columns(2)

with left:
    st.markdown("##### Quality class distribution")
    if classified.empty:
        status_box(
            "No classified images yet. This chart will show the share of "
            "Good, Damaged and Rotten apples once the model is trained."
        )
    else:
        counts = (
            classified["predicted_class"]
            .value_counts()
            .reindex(QUALITY_CLASSES, fill_value=0)
            .reset_index()
        )
        counts.columns = ["Class", "Count"]
        fig = px.pie(counts, names="Class", values="Count", hole=0.55,
                     color="Class", color_discrete_map=CLASS_COLOURS)
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

with right:
    st.markdown("##### Analyses over time")
    if df.empty:
        status_box("No analyses saved yet. Save an image on the Analyse Apple page.")
    else:
        per_day = df.groupby(df["created_at"].dt.date).size().reset_index(name="Analyses")
        per_day.columns = ["Date", "Analyses"]
        fig = px.bar(per_day, x="Date", y="Analyses", color_discrete_sequence=["#2E7D32"])
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                          plot_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, width="stretch")

st.markdown("##### Image quality: brightness vs sharpness")
if df.empty:
    status_box("Image measurements will be plotted here after images are saved.")
else:
    fig = px.scatter(
        df, x="brightness", y="sharpness", hover_name="filename",
        labels={"brightness": "Brightness (0-255)", "sharpness": "Sharpness (Laplacian variance)"},
        color_discrete_sequence=["#558B2F"],
    )
    fig.update_traces(marker=dict(size=11, opacity=0.8))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

st.markdown("##### Model evaluation")
status_box(
    "Accuracy, precision, recall and the confusion matrix will be reported here "
    "after the model has been trained and evaluated on a held-out test set."
)

footer()
