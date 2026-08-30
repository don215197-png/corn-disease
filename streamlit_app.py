import streamlit as st
from PIL import Image

from model_core import friendly, gradcam_views, load_model, predict

st.set_page_config(
    page_title="Crop Disease Detector",
    page_icon=":material/agriculture:",
    layout="wide",
)

st.html(
    """
    <div class="cdd-header">
      <h1>Crop Disease Detector</h1>
      <p>Upload a photo of a crop leaf and the AI model identifies the disease it's most
      likely suffering from — and highlights exactly which region of the leaf convinced it.</p>
      <div class="cdd-metrics">
        <span class="cdd-metric"><b>99.67%</b> test accuracy</span>
        <span class="cdd-metric"><b>38</b> disease / healthy classes</span>
        <span class="cdd-metric"><b>14</b> crop species</span>
        <span class="cdd-metric"><b>100%</b> top-5 accuracy</span>
      </div>
    </div>
    <style>
      .cdd-header{background:linear-gradient(135deg,#31431f 0%,#4c6b2d 55%,#6d8f3c 100%);
        color:#f6f2e2;border-radius:16px;padding:26px 28px 22px;margin-bottom:18px;}
      .cdd-header h1{margin:0 0 6px;font-family:'Lora',Georgia,serif;font-size:34px;letter-spacing:.5px;}
      .cdd-header p{margin:0 0 14px;color:#e8e4cf;font-size:15px;}
      .cdd-metrics{display:flex;gap:10px;flex-wrap:wrap;}
      .cdd-metric{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.25);
        border-radius:999px;padding:5px 14px;font-size:13px;}
      .cdd-metric b{color:#fff;}
      .cdd-stamp{display:inline-block;padding:7px 30px;border-radius:7px;
        font-family:'Lora',Georgia,serif;font-weight:700;font-size:27px;letter-spacing:2.5px;
        transform:rotate(-4deg);border:4px solid;animation:cdd-in .45s ease-out;}
      @keyframes cdd-in{from{transform:rotate(-20deg) scale(1.35);opacity:0}
        to{transform:rotate(-4deg) scale(1);opacity:1}}
      .cdd-verdict{font-family:'Lora',Georgia,serif;font-size:15px;margin:6px 0 14px;}
      .cdd-note{color:#8a8572;font-size:12px;margin-top:14px;line-height:1.6;}
    </style>
    """
)


@st.cache_resource
def get_model():
    return load_model()


uploaded = st.file_uploader(
    "Crop leaf photo",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed",
)

left, right = st.columns([5, 7], gap="medium")

with left:
    diagnose = st.button(
        "Diagnose",
        type="primary",
        icon=":material/science:",
        width="stretch",
    )
    st.caption(
        "Resize to 224×224 · ImageNet normalization · EfficientNet-B0 · "
        "softmax over 38 classes"
    )

if not diagnose:
    with right:
        st.info(
            "Upload a leaf photo above and press **Diagnose** to get the report.",
            icon=":material/upload_file:",
        )

if diagnose and uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        model = get_model()

        with st.spinner("Analyzing this leaf…"):
            top = predict(model, image)
            views = gradcam_views(model, image)

        top1 = top[0]
        crop, disease = friendly(top1["class"])
        is_healthy = disease == "Healthy"
        stamp_class = "healthy" if is_healthy else "disease"
        stamp_text = "HEALTHY" if is_healthy else "DISEASE DETECTED"

        with right:
            st.html(
                f"""
                <div class="cdd-stamp" style="color:{'#2e7d32' if is_healthy else '#b3261e'};
                  border-color:{'#2e7d32' if is_healthy else '#b3261e'};
                  background:{'rgba(46,125,50,.06)' if is_healthy else 'rgba(179,38,30,.06)'};">
                  {stamp_text}
                </div>
                <div class="cdd-verdict">
                  Crop: <b>{crop}</b> · Diagnosis: <b>{disease}</b> ·
                  Confidence: <b>{top1['confidence'] * 100:.2f}%</b>
                </div>
                """
            )

            st.markdown("**Top confidences**")
            for r in top:
                name = " · ".join(friendly(r["class"]))
                st.progress(
                    r["confidence"],
                    text=f"{name} — {r['confidence'] * 100:.1f}%",
                )

            st.html('<div class="cdd-note">AI screening for demonstration — '
                    'always confirm a diagnosis with a plant pathologist or '
                    'agricultural extension agent before applying any treatment.</div>')

        st.markdown("**Grad-CAM explainability — where the model is looking**")
        col1, col2, col3 = st.columns(3)
        col1.image(views["original"], caption="Original", width="stretch")
        col2.image(views["heatmap"], caption="Grad-CAM heatmap", width="stretch")
        col3.image(views["overlay"], caption="Overlay", width="stretch")

    except Exception as e:
        st.error(f"Couldn't analyze this image: {e}", icon=":material/error:")
elif diagnose:
    with right:
        st.warning("Upload a leaf photo first.", icon=":material/upload_file:")

with st.expander(
    "How it works & model card",
    icon=":material/description:",
):
    st.markdown(
        """**Model:** EfficientNet-B0, transfer-learned, fine-tuned on the **PlantVillage** dataset.

- **Data:** 54,304 leaf photos, 38 classes across 14 crops (tomato, potato, apple, grape, corn, pepper, squash, soybean, strawberry, peach, cherry, blueberry, raspberry, orange).
- **Training:** stratified train/val/test split, augmentation (flip, rotation, colour jitter), mixed-precision, cosine scheduling.
- **Evaluation:** 99.67% test accuracy, 99.67% weighted F1, 100% top-5 accuracy; data-leakage check by leaf id; Grad-CAM heatmaps for explainability.
- **Inference pipeline:** image resized to 224×224, normalised to ImageNet stats, softmax probabilities, top-3 ranked, Grad-CAM overlay on the winning class.

This is an educational demo. It is not a replacement for a professional agronomy diagnosis."""
    )

st.caption(
    "Built with PyTorch & Streamlit · EfficientNet-B0 · PlantVillage · a portfolio project"
)