import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.dataset import FaceDataset
from src.augmentations import get_train_transforms
from src.train import train_model
import json

df = pd.read_csv("face_processed.csv")
image_paths = df['image_path'].tolist()
genders = df['gender'].tolist()
ids = df['identity'].tolist()
id_to_idx = {id_: i for i, id_ in enumerate(sorted(set(ids)))}
mapped_ids = [id_to_idx[i] for i in ids]

with open("id_to_idx.json", "w") as f:
    json.dump(id_to_idx, f)

dataset = FaceDataset(image_paths, genders, mapped_ids, transform=get_train_transforms())
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model(loader, loader, num_classes=len(id_to_idx), device=device)
