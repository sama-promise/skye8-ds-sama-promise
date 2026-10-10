import copy
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


# ============================================================
# 1. PATHS
# ============================================================

HERE = Path(__file__).parent

DATA_DIR = HERE / "data" / "raw" / "Cattle side and back view images"
RAW_DIR = DATA_DIR / "side view"

MEASUREMENTS_FILE = DATA_DIR / "measurements.xlsx"


# ============================================================
# 2. LOAD DATA
# ============================================================

measurements = pd.read_excel(MEASUREMENTS_FILE)

# Keep only cattle that actually have a side-view image
measurements = measurements[
    measurements["Num"].apply(
        lambda n: (RAW_DIR / f"{n}.png").exists()
    )
].reset_index(drop=True)

print(f"Total cattle with side images: {len(measurements)}")

print("\nCattle IDs:")
print(measurements["Num"].tolist())


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First: 80% train+validation, 20% test
train_val_df, test_df = train_test_split(
    measurements,
    test_size=0.20,
    random_state=42
)

# Then: split the 80% into training and validation
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.20,
    random_state=42
)

print("\nDataset split:")
print(f"Training:   {len(train_df)} cattle")
print(f"Validation: {len(val_df)} cattle")
print(f"Testing:    {len(test_df)} cattle")


# ============================================================
# 4. DATASET
# ============================================================

class CattleDataset(Dataset):

    def __init__(self, df, image_dir, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = self.image_dir / f"{row['Num']}.png"

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        weight = torch.tensor(
            row["Body weight (kg)"],
            dtype=torch.float32
        )

        return image, weight


# ============================================================
# 5. IMAGE TRANSFORMS
# ============================================================

# Training images get augmentation.
train_transform = T.Compose([

    T.Resize((256, 256)),

    T.RandomResizedCrop(
        224,
        scale=(0.85, 1.0),
        ratio=(0.9, 1.1)
    ),

    T.RandomHorizontalFlip(p=0.5),

    T.RandomRotation(8),

    T.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),

    T.ToTensor(),

    T.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


# Validation/test images should NOT be randomly changed.
eval_transform = T.Compose([

    T.Resize((224, 224)),

    T.ToTensor(),

    T.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


# ============================================================
# 6. DATA LOADERS
# ============================================================

train_dataset = CattleDataset(
    train_df,
    RAW_DIR,
    train_transform
)

val_dataset = CattleDataset(
    val_df,
    RAW_DIR,
    eval_transform
)

test_dataset = CattleDataset(
    test_df,
    RAW_DIR,
    eval_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)


# ============================================================
# 7. MODEL
# ============================================================

print("\nLoading pretrained ResNet18...")

model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)


# Freeze everything first
for param in model.parameters():
    param.requires_grad = False


# Unfreeze the final ResNet block
for param in model.layer4.parameters():
    param.requires_grad = True


# Replace final classifier
model.fc = torch.nn.Sequential(

    torch.nn.Dropout(0.30),

    torch.nn.Linear(
        model.fc.in_features,
        1
    )
)


# The new FC layer must train
for param in model.fc.parameters():
    param.requires_grad = True


# ============================================================
# 8. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

model = model.to(device)


# ============================================================
# 9. TARGET NORMALIZATION
# ============================================================

train_mean = train_df["Body weight (kg)"].mean()
train_std = train_df["Body weight (kg)"].std()

print(
    f"\nTraining weight mean: {train_mean:.1f} kg"
)

print(
    f"Training weight std:  {train_std:.1f} kg"
)


# ============================================================
# 10. LOSS + OPTIMIZER
# ============================================================

loss_fn = torch.nn.SmoothL1Loss()

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.layer4.parameters(),
            "lr": 0.00005
        },
        {
            "params": model.fc.parameters(),
            "lr": 0.0005
        }
    ],
    weight_decay=0.0001
)


# ============================================================
# 11. TRAINING
# ============================================================

EPOCHS = 40

best_val_mae = float("inf")
best_state = None

patience = 8
epochs_without_improvement = 0


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_predictions = []
    train_actuals = []

    for images, weights in train_loader:

        images = images.to(device)
        weights = weights.to(device)

        # Normalize target
        weights_norm = (
            weights - train_mean
        ) / train_std

        predictions_norm = model(
            images
        ).squeeze(1)

        loss = loss_fn(
            predictions_norm,
            weights_norm
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Convert predictions back to kg
        predictions_kg = (
            predictions_norm * train_std
            + train_mean
        )

        train_predictions.extend(
            predictions_kg.detach().cpu().numpy()
        )

        train_actuals.extend(
            weights.cpu().numpy()
        )


    train_mae = mean_absolute_error(
        train_actuals,
        train_predictions
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_predictions = []
    val_actuals = []

    with torch.no_grad():

        for images, weights in val_loader:

            images = images.to(device)
            weights = weights.to(device)

            predictions_norm = model(
                images
            ).squeeze(1)

            predictions_kg = (
                predictions_norm * train_std
                + train_mean
            )

            val_predictions.extend(
                predictions_kg.cpu().numpy()
            )

            val_actuals.extend(
                weights.cpu().numpy()
            )


    val_mae = mean_absolute_error(
        val_actuals,
        val_predictions
    )


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Train MAE: {train_mae:.1f} kg "
        f"| Val MAE: {val_mae:.1f} kg"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_mae < best_val_mae:

        best_val_mae = val_mae

        best_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

        print(
            f"   ✓ New best validation MAE: "
            f"{best_val_mae:.1f} kg"
        )

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= patience:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# 12. RESTORE BEST MODEL
# ============================================================

model.load_state_dict(best_state)

print(
    f"\nBest validation MAE: "
    f"{best_val_mae:.1f} kg"
)


# ============================================================
# 13. FINAL TEST
# ============================================================

model.eval()

test_predictions = []
test_actuals = []
test_ids = []


with torch.no_grad():

    for images, weights in test_loader:

        images = images.to(device)

        predictions_norm = model(
            images
        ).squeeze(1)

        predictions_kg = (
            predictions_norm * train_std
            + train_mean
        )

        test_predictions.extend(
            predictions_kg.cpu().numpy()
        )

        test_actuals.extend(
            weights.numpy()
        )


# Test MAE
test_mae = mean_absolute_error(
    test_actuals,
    test_predictions
)


print("\n======================================")
print("FINAL TEST RESULT")
print("======================================")

print(
    f"Test MAE: {test_mae:.1f} kg"
)


# ============================================================
# 14. SHOW INDIVIDUAL PREDICTIONS
# ============================================================

print("\nIndividual test predictions:")

for actual, predicted in zip(
    test_actuals,
    test_predictions
):

    error = abs(
        actual - predicted
    )

    print(
        f"Actual: {actual:.1f} kg | "
        f"Predicted: {predicted:.1f} kg | "
        f"Error: {error:.1f} kg"
    )