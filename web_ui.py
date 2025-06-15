import gradio as gr
import torch
from PIL import Image
import numpy as np
import pandas as pd
import json

from src.model import DualHeadFaceNet
from src.augmentations import get_val_transforms

# Load identity mapping and model
df = pd.read_csv("utkface_processed.csv")
num_classes = df['identity'].nunique()
with open("id_to_idx.json") as f:
    id_to_idx = json.load(f)
idx_to_id = {int(v): k for k, v in id_to_idx.items()}

model = DualHeadFaceNet(num_classes)
model.load_state_dict(torch.load("model.pt", map_location="cpu"), strict=False)
model.eval()
transform = get_val_transforms()

def predict_face(image):
    img = image.convert("RGB")
    img_tensor = transform(image=np.array(img))["image"].unsqueeze(0)
    with torch.no_grad():
        gender_logits, id_logits = model(img_tensor)
        gender_prob = torch.sigmoid(gender_logits).item()
        predicted_id_idx = torch.argmax(id_logits).item()
    gender = "Male" if gender_prob > 0.5 else "Female"
    identity = idx_to_id.get(int(predicted_id_idx), "Unknown")
    return f"Gender: {gender}\nIdentity: {identity}"

gr.Interface(
    fn=predict_face,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Face Gender & Identity Recognition",
    description="Upload a face image to get gender and identity predictions."
).launch()