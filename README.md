# Crop Disease Detector

A deep-learning image classifier that identifies crop and plant leaf diseases from a photo. Built with PyTorch (EfficientNet-B0, transfer learning), fine-tuned on the PlantVillage dataset, and served through a polished web app with Grad-CAM explainability.

**Live demo:** `https://huggingface.co/spaces/<your-username>/crop-disease-detector`

## Results

| Metric | Value |
|---|---|
| Test accuracy | **99.67%** |
| Weighted F1-score | **99.67%** |
| Top-5 accuracy | **100%** |
| Classes | 38 (disease / healthy) |
| Crop species | 14 |
| Training images | 54,304 |

## What it does

- Upload a photo of a crop leaf (drag & drop, or paste from clipboard).
- The model returns a **stamp-style verdict** (HEALTHY / DISEASE DETECTED) with crop, diagnosis, and confidence.
- Top-3 predictions shown as animated confidence bars.
- **Grad-CAM explainability** overlays a heatmap on the leaf, highlighting the exact region that drove the decision.

## Model & training

- **Backbone:** EfficientNet-B0 pretrained on ImageNet; classifier head replaced with a 38-way linear layer.
- **Preprocessing:** resized to 224x224, normalised to ImageNet stats (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
- **Augmentation:** random horizontal flip, random rotation (15°), colour jitter (brightness/contrast/saturation 0.2).
- **Optimisation:** AdamW (lr 1e-3, weight decay 1e-4), CrossEntropyLoss, ReduceLROnPlateau (factor 0.3, patience 2), 10 epochs, mixed-precision (fp16 autocast + GradScaler), best-checkpoint selection by validation accuracy.
- **Explainability:** Grad-CAM targeting `model.features[-1]`, implemented with PyTorch forward/backward hooks.

## Dataset & leakage control

Trained on the [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) dataset (54,304 leaf photos). Splits come from the dataset's own train/test labels:

- 10% of the training images held out as a **stratified validation set** (by class, `random_state=42`).
- An official **test split** kept completely out of training.
- **Leakage audit:** verified that no leaf `leaf_id` / leaf group appears in more than one split — zero overlap between training and test.

## How the app works

Single-file Gradio app (`app.py`):

1. **Image import** — convert to RGB, resize 224x224.
2. **Forward pass** — softmax over the 38 logits; top-3 ranked by confidence.
3. **Grad-CAM** — hooks capture activations + gradients at `model.features[-1]`; class-weighted ReLU map is resized and overlaid (jet colormap) on the original leaf.
4. **Report UI** — verdict stamp, diagnosis line, confidence bars, and the original / heatmap / overlay gallery.

## Project structure

```
.
├── app.py                  # Gradio web app (model load + inference + Grad-CAM + UI)
├── Untitled2.ipynb         # Full training notebook (Colab) — data, training, eval, export
├── crop_disease_model.pth  # Trained EfficientNet-B0 weights (state_dict)
├── class_names.json        # 38 class labels (index = class index)
├── requirements.txt        # Python dependencies
└── .gitattributes          # git-lfs config for the model file
```

## Quickstart

```bash
pip install -r requirements.txt
python app.py               # open http://127.0.0.1:7860
```

Requires Python 3.11+ (tested on 3.11 and 3.14).

## Deploy to Hugging Face Spaces (free, no Docker)

1. Create a Space: SDK **Gradio**, hardware **CPU basic (free)**, Python **3.11**.
2. Upload `app.py`, `requirements.txt`, `class_names.json`, `crop_disease_model.pth`, `.gitattributes` (browser upload works for the 15 MB model).
3. First build installs PyTorch (~5 minutes); the app then runs permanently at a public URL.

## Tech stack

Python, PyTorch, TorchVision, Gradio, NumPy, OpenCV, Hugging Face Spaces.

## Limitations & future work

- Unseen leaf species or images outside PlantVillage's 14 crops may be misclassified.
- Certification for on-farm use requires more out-of-distribution testing — this is a demonstration project.
- Future: add on-device (CoreML / TFLite) export, support for multiple leaves per photo, and confidence-based abstention.