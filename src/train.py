import torch
from torch import nn, optim
from tqdm import tqdm
from src.model import DualHeadFaceNet

def train_model(train_loader, val_loader, num_classes, device):
    model = DualHeadFaceNet(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    gender_criterion = nn.BCEWithLogitsLoss()
    id_criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        for x, gender, identity in tqdm(train_loader):
            x, gender, identity = x.to(device), gender.to(device), identity.to(device)
            gender_logits, id_logits = model(x)
            loss_gender = gender_criterion(gender_logits.squeeze(), gender)
            loss_id = id_criterion(id_logits, identity)
            loss = loss_gender + loss_id
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), "model.pt")
    print("✅ Model saved as model.pt")
