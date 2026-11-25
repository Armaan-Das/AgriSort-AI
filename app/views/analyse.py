"""Analyse Apple page - upload an image, classify it with the trained model, save the result."""

import plotly.graph_objects as go
import streamlit as st

from components.ui import footer, page_header, prediction_card
from core.classifier import InferenceError, ModelUnavailableError, predict
from core.config import ALLOWED_IMAGE_TYPES, CLASS_COLOURS, MAX_UPLOAD_MB, MODEL_CLASSES
from core.database import save_prediction
from core.model_info import load_report
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
    "Upload a photo of a single apple. The trained model classifies its visible quality "
    "as Good or Rotten, and OpenCV measures basic image properties.",
)

st.info(
    "**Prototype scope:** model v1 distinguishes **Good** and **Rotten** apples only. "
    "Damaged-apple classification and detection of individual apples in multi-apple photos "
    "are planned for future development.",
    icon=":material/info:",
)

uploaded = st.file_uploader(
    f"Upload an apple image ({', '.join(ALLOWED_IMAGE_TYPES).upper()}, max {MAX_UPLOAD_MB} MB)",
    type=ALLOWED_IMAGE_TYPES,
)

if uploaded is None:
    st.caption("Clear, well-lit photos of one apple work best.")
    footer()
    st.stop()

try:
    image = decode_image(uploaded.getvalue())
except ValueError as exc:
    st.error(f"{exc} Please upload a valid JPG or PNG photo.", icon=":material/error:")
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

# ---------------------------------------------------------------------------
# Quality prediction (genuine model output, computed once per upload)
# ---------------------------------------------------------------------------
st.markdown("#### Quality prediction")
results = st.session_state.setdefault("predictions", {})
if uploaded.file_id not in results:
    try:
        with st.spinner("Classifying..."):
            results[uploaded.file_id] = {"ok": True, "prediction": predict(image)}
    except (ModelUnavailableError, InferenceError) as exc:
        results[uploaded.file_id] = {"ok": False, "error": str(exc),
                                     "kind": type(exc).__name__}
outcome = results[uploaded.file_id]

if not outcome["ok"]:
    title = ("The quality model is unavailable." if outcome["kind"] == "ModelUnavailableError"
             else "This image could not be classified.")
    st.error(f"**{title}** {outcome['error']} No prediction is shown.", icon=":material/error:")
    prediction = None
else:
    prediction = outcome["prediction"]
    left, right = st.columns([1, 1], gap="large")
    with left:
        prediction_card(prediction["predicted_class"], prediction["confidence"])
        m1, m2 = st.columns(2)
        m1.metric("Inference time", f"{prediction['inference_ms']:.1f} ms", border=True,
                  help="Model preprocessing, forward pass and postprocessing on this computer "
                       f"({prediction['device'].upper()}).")
        m2.metric("Model", "v1", border=True, help=prediction["model_version"])
    with right:
        probs = prediction["probabilities"]
        fig = go.Figure(go.Bar(
            x=[probs[c] for c in MODEL_CLASSES], y=MODEL_CLASSES, orientation="h",
            marker_color=[CLASS_COLOURS[c] for c in MODEL_CLASSES],
            text=[f"{probs[c]:.1%}" for c in MODEL_CLASSES], textposition="auto",
        ))
        fig.update_layout(title="Class probabilities", height=220, margin=dict(l=10, r=10, t=40, b=10),
                          xaxis=dict(range=[0, 1], tickformat=".0%"), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    test = load_report().get("test")
    errors = (f" On held-out test data the model was wrong on {test['misclassified']} of "
              f"{test['n_images']} images (see Model Information)." if test else "")
    st.warning(
        "**The confidence score is not a guarantee.** It is the model's own probability estimate "
        "and can be high even when the prediction is wrong. It is **not a food-safety assessment**."
        + errors,
        icon=":material/warning:",
    )

# ---------------------------------------------------------------------------
# Image measurements (OpenCV)
# ---------------------------------------------------------------------------
st.markdown("#### Image measurements")
st.caption("These values are measured directly from the image with OpenCV.")
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

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def _save(filename: str, image_stats: dict, result: dict, file_id: str) -> None:
    """Runs before the page re-renders, so the button is disabled immediately after saving."""
    record_id = save_prediction(filename, image_stats, result)
    st.session_state["saved_upload_ids"][file_id] = record_id


st.write("")
saved_ids = st.session_state.setdefault("saved_upload_ids", {})
if prediction is None:
    st.button("Save inspection to history", type="primary", icon=":material/save:", disabled=True,
              help="Only inspections with a successful model prediction can be saved.")
else:
    st.button("Save inspection to history", type="primary", icon=":material/save:",
              on_click=_save, args=(uploaded.name, stats, prediction, uploaded.file_id),
              disabled=uploaded.file_id in saved_ids)
    if uploaded.file_id in saved_ids:
        st.success(f"Saved as record #{saved_ids[uploaded.file_id]}. It now appears on the History "
                   "and Dashboard pages.", icon=":material/check_circle:")

footer()
