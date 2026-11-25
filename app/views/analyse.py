"""Analyse Apple page - upload an image and inspect it with OpenCV."""

import plotly.graph_objects as go
import streamlit as st

from components.ui import footer, page_header, status_box
from core.classifier import is_model_available
from core.config import ALLOWED_IMAGE_TYPES, MAX_UPLOAD_MB
from core.database import save_analysis
from core.image_utils import (
    bgr_to_rgb,
    channel_means,
    compute_image_stats,
    decode_image,
    edge_map,
    resize_for_display,
)

page_header(
    "Analyse Apple",
    "Upload a photo of a single apple. The image is processed with OpenCV; "
    "quality classification will be enabled once the model is trained.",
)

uploaded = st.file_uploader(
    f"Upload an apple image ({', '.join(ALLOWED_IMAGE_TYPES).upper()}, max {MAX_UPLOAD_MB} MB)",
    type=ALLOWED_IMAGE_TYPES,
)

if uploaded is None:
    st.info("Upload an image to begin. Clear, well-lit photos of one apple work best.",
            icon=":material/upload:")
    footer()
    st.stop()

try:
    image = decode_image(uploaded.getvalue())
except ValueError as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()

stats = compute_image_stats(image)
display = resize_for_display(image)

col_img, col_edges = st.columns(2)
with col_img:
    st.markdown("**Uploaded image**")
    st.image(bgr_to_rgb(display), width="stretch")
with col_edges:
    st.markdown("**Edge map (OpenCV Canny)**")
    st.image(edge_map(display), width="stretch", clamp=True)

st.markdown("#### Image measurements")
st.caption("These values are measured directly from the image - they are not ML predictions.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Resolution", f"{stats['width']} × {stats['height']}", border=True)
m2.metric("Brightness", f"{stats['brightness']:.1f} / 255", border=True,
          help="Mean grey level. Very low or very high values indicate poor lighting.")
m3.metric("Contrast", f"{stats['contrast']:.1f}", border=True,
          help="Standard deviation of grey levels.")
m4.metric("Sharpness", f"{stats['sharpness']:.0f}", border=True,
          help="Variance of the Laplacian. Low values (< 100) suggest a blurry photo.")

if stats["sharpness"] < 100:
    st.warning("This image looks blurry. A sharper photo will give more reliable results.",
               icon=":material/blur_on:")

means = channel_means(image)
fig = go.Figure(
    go.Bar(
        x=list(means.keys()),
        y=list(means.values()),
        marker_color=["#C62828", "#2E7D32", "#1565C0"],
        text=[f"{v:.0f}" for v in means.values()],
        textposition="outside",
    )
)
fig.update_layout(
    title="Average colour channel intensity",
    yaxis=dict(range=[0, 270], title="Intensity (0-255)"),
    height=320,
    margin=dict(l=10, r=10, t=50, b=10),
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, width="stretch")

st.markdown("#### Quality prediction")
if is_model_available():
    status_box("Trained weights were found, but model integration is not complete yet.")
else:
    status_box(
        "Model not trained yet - no quality prediction is shown. "
        "Good / Damaged / Rotten classification will appear here after the "
        "YOLO model has been trained and evaluated."
    )

st.write("")
saved_ids = st.session_state.setdefault("saved_upload_ids", set())
already_saved = uploaded.file_id in saved_ids
if st.button("Save measurements to history", type="primary",
             icon=":material/save:", disabled=already_saved):
    record_id = save_analysis(uploaded.name, stats)
    saved_ids.add(uploaded.file_id)
    st.success(f"Saved as record #{record_id}. View it on the History page.",
               icon=":material/check_circle:")
elif already_saved:
    st.caption("This upload has already been saved.")

footer()
