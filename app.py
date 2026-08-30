import os

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from model_core import MEAN, STD, friendly, load_model, make_gradcam, predict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model()


def gradcam_gallery(image_pil, top_index):
    cam, _ = make_gradcam(model, image_pil)

    rgb = val_transform(image_pil).cpu().numpy().transpose(1, 2, 0)
    rgb = np.clip(rgb * STD + MEAN, 0, 1)

    cam_224 = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_224), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(0.5 * np.uint8(255 * rgb) + 0.5 * heatmap_rgb)

    base = np.array(image_pil.convert("RGB"))
    h, w = base.shape[:2]
    max_side = 640
    scale = min(max_side / w, max_side / h, 1.0)
    if scale < 1.0:
        base = cv2.resize(base, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def to_pil(arr):
        return Image.fromarray(arr.astype(np.uint8))

    return [
        (to_pil(base), "Original"),
        (to_pil(np.uint8(255 * cam_224)).convert("L"), "Grad-CAM heatmap"),
        (to_pil(overlay), "Overlay"),
    ]


def analyze(image):
    if image is None:
        return "<div class='err'>No image uploaded.</div>", None

    try:
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        top_results = predict(model, image)
        cam, top_index = make_gradcam(model, image)
        gallery = gradcam_gallery(image, top_index)
    except Exception as e:
        return f"<div class='err'>Couldn't analyse this image: {e}</div>", None

    top1 = top_results[0]
    crop, disease = friendly(top1["class"])
    is_healthy = disease == "Healthy"

    stamp_class = "healthy" if is_healthy else "disease"
    stamp_text = "HEALTHY" if is_healthy else "DISEASE DETECTED"

    bars = "".join(
        f"""<div class="pred-row">
              <div class="pred-name">{friendly(r['class'])[0]} &bull; {friendly(r['class'])[1]}</div>
              <div class="bar-track"><div class="bar-fill" style="width:{r['confidence']*100:.1f}%"></div></div>
              <div class="pred-conf">{r['confidence']*100:.1f}%</div>
            </div>"""
        for r in top_results
    )

    report = f"""
    <div id="stamp-box">
      <div class="stamp {stamp_class}">{stamp_text}</div>
      <div class="verdict-line">Crop: <b>{crop}</b> &mdash; Diagnosis: <b>{disease}</b> &mdash; Confidence: <b>{top1['confidence']*100:.2f}%</b></div>
    </div>
    <div style="margin:16px 0 6px;font-family:Georgia,serif;color:#31431f;font-weight:700;letter-spacing:.5px">TOP CONFIDENCES</div>
    {bars}
    <div style="margin-top:16px;font-size:12px;color:#8a8572">
      AI screening for demonstration &mdash; always confirm a diagnosis with a plant pathologist or
      agricultural extension agent before applying any treatment.
    </div>
    """

    return report, gallery


CSS = """
:root{
  --paper:#f4efe0;
  --card:#fffdf6;
  --leaf:#31431f;
  --leaf-mid:#6d8f3c;
  --ink:#2b2b26;
  --line:#ded5ba;
  --disease:#b3261e;
  --healthy:#2e7d32;
}
.gradio-container{ background:var(--paper); max-width:1180px!important; }
#app-banner{
  background:linear-gradient(135deg,#31431f 0%,#4c6b2d 55%,#6d8f3c 100%);
  color:#f6f2e2; border-radius:16px; padding:26px 28px 22px; margin-bottom:18px;
  box-shadow:0 10px 30px rgba(49,67,31,.25);
}
#app-banner h1{
  margin:0 0 6px; font-family:Georgia,'Times New Roman',serif; font-size:34px;
  letter-spacing:.5px;
}
#app-banner p{ margin:0 0 14px; color:#e8e4cf; font-size:15px; }
.metrics{ display:flex; gap:10px; flex-wrap:wrap; }
.metric{
  background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.25);
  border-radius:999px; padding:5px 14px; font-size:13px;
}
.metric b{ color:#fff; }
.cards{
  border-radius:14px !important; border:1px solid var(--line) !important;
  background:var(--card) !important; box-shadow:0 4px 18px rgba(49,67,31,.08) !important;
  padding:18px !important;
}
#stamp-box{ text-align:center; padding:6px 0 2px; }
.stamp{
  display:inline-block; padding:7px 30px; border-radius:7px;
  font-family:Georgia,serif; font-weight:700; font-size:27px; letter-spacing:2.5px;
  transform:rotate(-4deg); border:4px solid;
  animation:stampin .45s ease-out;
}
@keyframes stampin{ from{transform:rotate(-20deg) scale(1.35); opacity:0} to{transform:rotate(-4deg) scale(1); opacity:1} }
.stamp.healthy{ color:var(--healthy); border-color:var(--healthy); background:rgba(46,125,50,.06); }
.stamp.disease{ color:var(--disease); border-color:var(--disease); background:rgba(179,38,30,.06); }
.verdict-line{ font-family:Georgia,serif; color:#4a5a30; font-size:15px; margin-top:6px; }
.pred-row{ display:flex; align-items:center; gap:12px; margin:10px 0; font-family:'Trebuchet MS',sans-serif; }
.pred-name{ width:44%; font-size:13.5px; line-height:1.25; color:var(--ink); }
.bar-track{ flex:1; height:18px; background:#eee6cc; border-radius:9px; overflow:hidden; }
.bar-fill{ height:100%; border-radius:9px; background:linear-gradient(90deg,#6d8f3c,#31431f);
  transition:width .8s cubic-bezier(.22,1,.36,1); }
.pred-conf{ width:64px; text-align:right; font-weight:700; color:var(--leaf); }
.err{ color:var(--disease); font-family:Georgia,serif; }
#app-footer{ color:#7c7764; font-size:13px; text-align:center; margin-top:18px; line-height:1.6; }
.upload-zone{
  border:2px dashed #b7ab84 !important; border-radius:14px !important;
  background:#faf6e7 !important; transition:border-color .2s, background .2s;
}
.upload-zone:hover,.upload-zone:focus-within{ border-color:var(--leaf-mid) !important; background:#f6f1dc !important; }
"""


def build_demo():
    theme = gr.themes.Soft(
        primary_hue="green",
        secondary_hue="lime",
        neutral_hue="stone",
        font=gr.themes.GoogleFont("Nunito Sans"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    )

    with gr.Blocks(
        css=CSS, theme=theme, title="Crop Disease Detector - AI Leaf Diagnosis"
    ) as demo:
        gr.HTML(
            """<div id="app-banner">
                <h1>Crop Disease Detector</h1>
                <p>Upload a photo of a crop leaf and the AI model identifies the disease it's most likely suffering from &mdash;
                and highlights exactly which region of the leaf convinced it.</p>
                <div class="metrics">
                  <span class="metric"><b>99.67%</b> test accuracy</span>
                  <span class="metric"><b>38</b> disease / healthy classes</span>
                  <span class="metric"><b>14</b> crop species</span>
                  <span class="metric"><b>100%</b> top-5 accuracy</span>
                </div>
              </div>"""
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=320):
                with gr.Group(elem_classes=["cards"]):
                    image_in = gr.Image(
                        label="Crop leaf photo",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=340,
                        elem_classes=["upload-zone"],
                    )
                    diagnose_btn = gr.Button(
                        "Diagnose", variant="primary", size="lg"
                    )

            with gr.Column(scale=7, min_width=360):
                with gr.Group(elem_classes=["cards"]):
                    report = gr.HTML(
                        """<div id="stamp-box">
                            <div class="verdict-line">Upload a leaf photo and press <b>Diagnose</b> to get the report.</div>
                          </div>"""
                    )
                gradcam_gallery = gr.Gallery(
                    label="Grad-CAM explainability &mdash; where the model is looking",
                    columns=3,
                    height=230,
                    object_fit="cover",
                )

        with gr.Accordion("How it works &amp; model card", open=False):
            gr.Markdown(
                """**Model:** EfficientNet-B0, transfer-learned, fine-tuned on the **PlantVillage** dataset.

- **Data:** 54,304 leaf photos, 38 classes across 14 crops (tomato, potato, apple, grape, corn, pepper, squash, soybean, strawberry, peach, cherry, blueberry, raspberry, orange).
- **Training:** stratified train/val/test split, augmentation (flip, rotation, colour jitter), mixed-precision, cosine scheduling.
- **Evaluation:** 99.67% test accuracy, 99.67% weighted F1, 100% top-5 accuracy; data-leakage check by leaf id; Grad-CAM heatmaps for explainability.
- **Inference pipeline:** image resized to 224&times;224, normalised to ImageNet stats, softmax probabilities, top-3 ranked, Grad-CAM overlay on the winning class.

This is an educational demo. It is not a replacement for a professional agronomy diagnosis."""
            )

        gr.HTML(
            """<div id="app-footer">
                Built with PyTorch &amp; Gradio &middot; EfficientNet-B0 &middot; PlantVillage &middot;
                a portfolio project by yours truly.
              </div>"""
        )

        diagnose_btn.click(
            analyze, inputs=image_in, outputs=[report, gradcam_gallery],
            show_progress="full",
        )

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_port=int(os.environ.get("PORT", 7860)))