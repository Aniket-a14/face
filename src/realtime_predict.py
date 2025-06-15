import cv2
import cv2.data
import torch
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from src.augmentations import get_val_transforms
from src.model import DualHeadFaceNet

class GenderNet(nn.Module):
    def __init__(self):
        super(GenderNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)  # Only gender

    def forward(self, x):
        return self.backbone(x)

def get_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

def realtime_predict(model_path):
    csv_path = "utkface_processed.csv"
    df = pd.read_csv(csv_path)
    num_classes = df['identity'].nunique()
    id_to_idx = {id_: i for i, id_ in enumerate(sorted(df['identity'].unique()))}
    idx_to_id = {v: k for k, v in id_to_idx.items()}

    model = DualHeadFaceNet(num_id_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.eval()

    transform = get_val_transforms()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb).convert("RGB")

            try:
                face_tensor = transform(image=np.array(face_pil))["image"].unsqueeze(0)
            except Exception as e:
                print("Transform error:", e)
                continue

            with torch.no_grad():
                gender_logit, id_logits = model(face_tensor)
                gender = torch.sigmoid(gender_logit).item()

            label = f"{'Male' if gender < 0.5 else 'Female'}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Real-time Gender Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_predict("model.pt")
