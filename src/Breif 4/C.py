import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "raw" / "Cattle side and back view images"
RAW_DIR = DATA_DIR / "side view"

measurements = pd.read_excel(DATA_DIR / "measurements.xlsx")
measurements = measurements[measurements["Num"].apply(lambda n: (RAW_DIR / f"{n}.png").exists())].reset_index(drop=True)

train_df, test_df = train_test_split(measurements, test_size=0.2, random_state=42)
print(f"Training on {len(train_df)}, testing on {len(test_df)}")


class CattleDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.image_dir / f"{row['Num']}.png").convert("RGB")
        img = self.transform(img)
        weight = torch.tensor(row["Body weight (kg)"], dtype=torch.float32)
        return img, weight


transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dataset = CattleDataset(train_df, RAW_DIR, transform)
test_dataset = CattleDataset(test_df, RAW_DIR, transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# --- Model: pretrained ResNet18, fully frozen except the new final layer ---
model = resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(model.fc.in_features, 1)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Training {trainable:,} of {total:,} parameters ({trainable/total*100:.2f}%)")

# --- Normalize the target weight (critical for training stability) ---
train_mean = train_df["Body weight (kg)"].mean()
train_std = train_df["Body weight (kg)"].std()
print(f"Train weight mean: {train_mean:.1f} kg, std: {train_std:.1f} kg")

# --- Training loop ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.001
)

EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    total_loss_kg = 0
    for images, weights in train_loader:
        images, weights = images.to(device), weights.to(device)

        weights_norm = (weights - train_mean) / train_std

        predictions_norm = model(images).squeeze(1)
        loss = loss_fn(predictions_norm, weights_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_kg += loss.item() * train_std * images.size(0)

    avg_loss_kg = total_loss_kg / len(train_dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train MAE: {avg_loss_kg:.1f} kg")

print("Training loop finished.")

# --- Evaluate on the held-out test set ---
model.eval()
all_preds_kg, all_actuals_kg = [], []

with torch.no_grad():
    for images, weights in test_loader:
        images, weights = images.to(device), weights.to(device)
        predictions_norm = model(images).squeeze(1)
        predictions_kg = predictions_norm * train_std + train_mean  # undo normalization
        all_preds_kg.extend(predictions_kg.cpu().numpy())
        all_actuals_kg.extend(weights.cpu().numpy())

test_mae = mean_absolute_error(all_actuals_kg, all_preds_kg)
print(f"\nTest MAE (CNN, raw photos, last-layer-only fine-tune): {test_mae:.1f} kg")