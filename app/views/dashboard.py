"""Dashboard page - statistics calculated only from inspections saved in SQLite."""

import plotly.express as px
import streamlit as st

from components.ui import footer, page_header, status_box
from core.classifier import model_status
from core.config import CLASS_COLOURS, MODEL_CLASSES
from core.database import load_analyses
from core.model_info import load_report

page_header(
    "Dashboard",
    "Live summary of inspections saved in the SQLite database. Every number and chart is "
    "calculated from real saved records - nothing is simulated.",
)

df = load_analyses()
classified = df[df["predicted_class"].notna()] if not df.empty else df

k1, k2, k3, k4 = st.columns(4)
k1.metric("Inspections saved", len(df), border=True)
k2.metric("Classified by the model", len(classified), border=True)
k3.metric(
    "Predicted Rotten",
    f"{(classified['predicted_class'] == 'Rotten').mean():.0%}" if len(classified) else "-",
    border=True,
    help="Share of classified inspections that the model predicted as Rotten.",
)
k4.metric(
    "Average confidence",
    f"{classified['confidence'].mean():.0%}" if len(classified) else "-",
    border=True,
    help="Mean model confidence of saved predictions. Confidence is not accuracy.",
)
st.caption(f"Model status: {model_status()}. These figures describe the model's *predictions* "
           "on uploaded photos; they are not accuracy measurements, because the true quality "
           "of uploaded apples is not known.")

unclassified = len(df) - len(classified)
if unclassified:
    st.caption(f"{unclassified} record(s) were saved before the model was integrated and have no prediction.")

st.write("")
left, right = st.columns(2)

with left:
    st.markdown("##### Predicted quality distribution")
    if classified.empty:
        status_box("No classified inspections yet. Analyse and save an apple image to populate this chart.")
    else:
        counts = (classified["predicted_class"].value_counts()
                  .reindex(MODEL_CLASSES, fill_value=0).reset_index())
        counts.columns = ["Class", "Count"]
        fig = px.pie(counts, names="Class", values="Count", hole=0.55,
                     color="Class", color_discrete_map=CLASS_COLOURS)
        fig.update_traces(textinfo="label+value+percent")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

with right:
    st.markdown("##### Inspections over time")
    if df.empty:
        status_box("No inspections saved yet. Save an image on the Analyse Apple page.")
    else:
        daily = df.assign(Date=df["created_at"].dt.date.astype(str),
                          Result=df["predicted_class"].fillna("Not classified"))
        per_day = daily.groupby(["Date", "Result"]).size().reset_index(name="Inspections")
        fig = px.bar(per_day, x="Date", y="Inspections", color="Result",
                     color_discrete_map={**CLASS_COLOURS, "Not classified": "#B0BEC5"})
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)",
                          legend_title_text="")
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.markdown("##### Model confidence of saved predictions")
    if classified.empty:
        status_box("Confidence values will appear here after classified inspections are saved.")
    else:
        fig = px.histogram(classified, x="confidence", color="predicted_class", nbins=10,
                           range_x=[0.5, 1.0], color_discrete_map=CLASS_COLOURS,
                           labels={"confidence": "Confidence", "predicted_class": "Prediction"})
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)",
                          bargap=0.05, xaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

with right:
    st.markdown("##### Image quality: brightness vs sharpness")
    if df.empty:
        status_box("Image measurements will be plotted here after images are saved.")
    else:
        plot_df = df.assign(Prediction=df["predicted_class"].fillna("Not classified"))
        fig = px.scatter(
            plot_df, x="brightness", y="sharpness", color="Prediction", hover_name="filename",
            labels={"brightness": "Brightness (0-255)", "sharpness": "Sharpness (Laplacian variance)"},
            color_discrete_map={**CLASS_COLOURS, "Not classified": "#B0BEC5"},
        )
        fig.update_traces(marker=dict(size=11, opacity=0.8))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

st.markdown("##### Model evaluation (held-out test set)")
test = load_report().get("test")
if test is None:
    status_box("Evaluation report not found.")
else:
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Test accuracy", f"{test['accuracy']:.1%}", border=True)
    e2.metric("Balanced accuracy", f"{test['balanced_accuracy']:.1%}", border=True)
    e3.metric("Macro F1", f"{test['macro_f1']:.3f}", border=True)
    e4.metric("Test images", f"{test['n_images']} ({test['n_independent_groups']} groups)", border=True)
    st.page_link("views/model_info.py", label="Full results, learning curves and limitations",
                 icon=":material/arrow_forward:")

footer()
