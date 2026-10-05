import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy import ndimage
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights)
from torchvision.transforms.functional import to_tensor

HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "raw" / "Cattle side and back view images"

MAX_SIDE = 1280
MIN_SCORE = 0.5


def keep_largest_component(mask):
    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask  # already one piece, nothing to do
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest_label = np.argmax(sizes) + 1  # labels start at 1, not 0
    return labeled == largest_label


weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights).eval()
COW = weights.meta["categories"].index("cow")

flagged_nums = [9, 15, 25, 54, 60]

fig, axes = plt.subplots(len(flagged_nums), 2, figsize=(10, 4 * len(flagged_nums)))

print("Connected-component check (1 blob = one solid shape, good):")

for row, num in enumerate(flagged_nums):
    img = Image.open(DATA_DIR / "side view" / f"{num}.png").convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))

    with torch.no_grad():
        out = model([to_tensor(img)])[0]

    best, best_area = None, 0
    for label, score, m in zip(out["labels"], out["scores"], out["masks"]):
        if label.item() == COW and score.item() >= MIN_SCORE:
            mask = m[0].numpy() > 0.5
            if mask.sum() > best_area:
                best, best_area = mask, mask.sum()

    mask_before = best
    mask_after = keep_largest_component(best)

    n_before = ndimage.label(mask_before)[1]
    n_after = ndimage.label(mask_after)[1]
    pixels_changed = mask_after.sum() - mask_before.sum()
    print(f"Cow #{num}: {n_before} blob(s) before -> {n_after} blob(s) after, "
          f"{pixels_changed} pixel change (negative = fragments removed)")

    arr = np.array(img)
    for ax, mask, title in zip(axes[row], [mask_before, mask_after], ["Before fix", "After fix"]):
        overlay = arr.copy()
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        ax.imshow(overlay)
        ax.set_title(f"Cow #{num} - {title}")
        ax.axis("off")

plt.tight_layout()
plt.show()