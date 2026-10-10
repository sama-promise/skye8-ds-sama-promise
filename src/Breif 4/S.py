import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights)
from torchvision.transforms.functional import to_tensor
 
HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "raw" / "Cattle side and back view images"
OUT_DIR = HERE / "outputs" / "seg_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
MAX_SIDE = 1280      # downscale for speed; the originals are very large
MIN_SCORE = 0.5
 
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights).eval()
COW = weights.meta["categories"].index("cow")
 
 
def segment_cow(img):
    """Return (mask, score) for the largest cow instance, or (None, None)."""
    with torch.no_grad():
        out = model([to_tensor(img)])[0]
    best, best_area, best_score = None, 0, None
    for label, score, m in zip(out["labels"], out["scores"], out["masks"]):
        if label.item() != COW or score.item() < MIN_SCORE:
            continue
        mask = m[0].numpy() > 0.5
        if mask.sum() > best_area:
            best, best_area, best_score = mask, mask.sum(), score.item()
    return best, best_score
 
 
animal_nums = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
 
for num in animal_nums:
    img = Image.open(DATA_DIR / "side view" / f"{num}.png").convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    mask, score = segment_cow(img)
 
    if mask is None:
        print(f"Cow #{num}: no cow found")
        continue
 
    arr = np.array(img)
    cutout = np.full_like(arr, 255)
    cutout[mask] = arr[mask]
    ys, xs = np.where(mask)
    w, h = xs.max() - xs.min(), ys.max() - ys.min()
    print(f"Cow #{num}: score {score:.2f}, mask covers "
          f"{mask.mean() * 100:.0f}% of frame, bbox {w}x{h} px")
 
    overlay = arr.copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, im, title in zip(axes, [arr, overlay, cutout],
                             ["Original", "Mask overlay", "Cutout"]):
        ax.imshow(im)
        ax.set_title(f"Cow #{num} - {title}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"cow_{num}.png", dpi=100)
    plt.show()
 