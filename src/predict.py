import torch
from PIL import Image
import numpy as np
from src.model import DualHeadFaceNet
from src.augmentations import get_val_transforms
import pandas as pd
import json

def predict(image_path, model_path, csv_path="face_processed.csv"):
    df = pd.read_csv(csv_path)
    num_classes = df['identity'].nunique()
    
    with open("id_to_idx.json") as f:
        id_to_idx = json.load(f)
    idx_to_id = {int(v): k for k, v in id_to_idx.items()}

    model = DualHeadFaceNet(num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.eval()
    
    transform = get_val_transforms()
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(image=np.array(img))["image"].unsqueeze(0)

    with torch.no_grad():
        gender_logits, id_logits = model(img_tensor)
        gender = torch.sigmoid(gender_logits).item()
        predicted_id_idx = torch.argmax(id_logits).item()

    predicted_id = idx_to_id[int(predicted_id_idx)]
    print(f"Predicted Gender: {'Male' if gender < 0.5 else 'Female'}")
    print(f"Predicted identity: {predicted_id}")

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    model_path = "model.pt"
    predict(image_path, model_path)
