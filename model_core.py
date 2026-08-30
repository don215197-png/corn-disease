import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_disease_model.pth")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

NUM_CLASSES = 38
DEVICE = torch.device("cpu")

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def friendly(label):
    crop, disease = label.split("___", 1)
    if disease == "healthy":
        return crop.replace("_", " "), "Healthy"
    return crop.replace("_", " "), disease.replace("_", " ")


def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def predict(model, image_pil, top_k=3):
    image_tensor = val_transform(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

    confidences, indices = torch.topk(
        probabilities, k=min(top_k, NUM_CLASSES), dim=1
    )

    return [
        {
            "class": class_names[index.item()],
            "confidence": confidence.item(),
            "class_index": index.item(),
        }
        for confidence, index in zip(confidences[0], indices[0])
    ]


def make_gradcam(model, image_pil):
    activations = {}
    gradients = {}
    target_layer = model.features[-1]

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    input_tensor = val_transform(image_pil).unsqueeze(0).to(DEVICE)
    output = model(input_tensor)
    top_index = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, top_index].backward()

    handle_f.remove()
    handle_b.remove()

    act = activations["value"][0].detach().cpu()
    grad = gradients["value"][0].detach().cpu()

    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = F.relu((weights * act).sum(dim=0)).numpy()

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    return cam, top_index


def gradcam_views(model, image_pil):
    cam, top_index = make_gradcam(model, image_pil)

    rgb = val_transform(image_pil).cpu().numpy().transpose(1, 2, 0)
    rgb = np.clip(rgb * STD + MEAN, 0, 1)

    cam_224 = cv2.resize(
        cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
    )

    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_224), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.5 * np.uint8(255 * rgb) + 0.5 * heatmap_rgb)

    base = np.array(image_pil.convert("RGB"))
    h, w = base.shape[:2]
    max_side = 640
    scale = min(max_side / w, max_side / h, 1.0)
    if scale < 1.0:
        base = cv2.resize(
            base, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    return {
        "original": Image.fromarray(base.astype(np.uint8)),
        "heatmap": Image.fromarray(np.uint8(255 * cam_224)).convert("L"),
        "overlay": Image.fromarray(overlay.astype(np.uint8)),
    }