import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "raw" / "Cattle side and back view images"
RAW_DIR = DATA_DIR / "side view"
CUT_DIR = HERE / "outputs" / "segmentation" / "side_view" / "cutouts"
 
FEATURE_COLS = ["Oblique body length (cm)", "Withers height(cm)",
                "Heart girth(cm)", "Hip length (cm)"]
 
# Only animals that have both a raw photo and a cutout
measurements = pd.read_excel(DATA_DIR / "measurements.xlsx")
has_both = [(RAW_DIR / f"{n}.png").exists() and (CUT_DIR / f"{n}.png").exists()
            for n in measurements["Num"]]
data = measurements[has_both].reset_index(drop=True)
print(f"Animals with both raw photo and cutout: {len(data)} of {len(measurements)}")
y = data["Body weight (kg)"].to_numpy(dtype=float)
 
# Frozen ImageNet ResNet-50 as a fixed feature extractor.
# Resize (not centre-crop) so the cow's head and rump are never cut off.
net = resnet50(weights=ResNet50_Weights.DEFAULT)
net.fc = torch.nn.Identity()
net.eval()
prep = T.Compose([
    T.Resize((288, 384)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
 
 
def features(path):
    img = Image.open(path).convert("RGB")
    with torch.no_grad():
        return net(prep(img).unsqueeze(0))[0].numpy()
 
 
print("Extracting features (raw)...")
X_raw = np.stack([features(RAW_DIR / f"{n}.png") for n in data["Num"]])
print("Extracting features (cutout)...")
X_cut = np.stack([features(CUT_DIR / f"{n}.png") for n in data["Num"]])
X_meas = data[FEATURE_COLS].to_numpy(dtype=float)
 
# One shared set of splits so every model sees exactly the same test animals
splits = list(RepeatedKFold(n_splits=5, n_repeats=5, random_state=42).split(y))
 
 
def fold_maes(X):
    maes = []
    for tr, te in splits:
        if X is None:
            pred = y[tr].mean()
        else:
            model = make_pipeline(StandardScaler(),
                                  RidgeCV(alphas=np.logspace(-2, 5, 30)))
            pred = model.fit(X[tr], y[tr]).predict(X[te])
        maes.append(np.abs(y[te] - pred).mean())
    return np.array(maes)
 
 
results = {
    "Baseline (mean weight)": fold_maes(None),
    "Raw photograph": fold_maes(X_raw),
    "Segmented cutout": fold_maes(X_cut),
    "Four body measurements": fold_maes(X_meas),
}
 
print(f"\nMAE in kg, {len(splits)} animal-level train/test splits:")
for name, m in results.items():
    print(f"  {name:<26} {m.mean():6.1f} kg   (sd across splits {m.std():.1f})")
 
diff = results["Raw photograph"] - results["Segmented cutout"]
print(f"\nSegmentation vs raw: cutout is better by {diff.mean():.1f} kg on average,"
      f" and better in {(diff > 0).mean() * 100:.0f}% of splits.")