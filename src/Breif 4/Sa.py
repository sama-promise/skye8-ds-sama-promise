import sys
import numpy as np
import pandas as pd
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
VIEW = sys.argv[1] if len(sys.argv) > 1 else "side view"
OUT = HERE / "outputs" / "segmentation" / VIEW.replace(" ", "_")
(OUT / "masks").mkdir(parents=True, exist_ok=True)
(OUT / "cutouts").mkdir(parents=True, exist_ok=True)

MAX_SIDE = 1280
MIN_SCORE = 0.5


def keep_largest_component(mask):
    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest_label = np.argmax(sizes) + 1
    return labeled == largest_label


weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights).eval()
COW = weights.meta["categories"].index("cow")

measurements = pd.read_excel(DATA_DIR / "measurements.xlsx")
rows, flagged = [], []

for num in measurements["Num"]:
    path = DATA_DIR / VIEW / f"{num}.png"
    if not path.exists():
        rows.append({"Num": num, "flags": "missing_file"})
        print(f"#{num}: file missing")
        continue

    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    with torch.no_grad():
        out = model([to_tensor(img)])[0]

    cows = []
    for label, score, m in zip(out["labels"], out["scores"], out["masks"]):
        if label.item() == COW and score.item() >= MIN_SCORE:
            mask = m[0].numpy() > 0.5
            cows.append((mask.sum(), score.item(), mask))
    cows.sort(key=lambda c: c[0], reverse=True)

    if not cows:
        rows.append({"Num": num, "flags": "no_cow"})
        flagged.append((num, np.array(img), None, "no_cow"))
        print(f"#{num}: no cow found")
        continue

    area, score, mask = cows[0]

    # Clean up stray disconnected fragments before anything else uses this mask
    n_blobs_before = ndimage.label(mask)[1]
    mask = keep_largest_component(mask)
    fragments_removed = n_blobs_before > 1

    second_ratio = cows[1][0] / area if len(cows) > 1 else 0.0
    area_frac = mask.mean()
    touches_border = bool(mask[0].any() or mask[-1].any()
                          or mask[:, 0].any() or mask[:, -1].any())
    ys, xs = np.where(mask)

    flags = []
    if score < 0.9:
        flags.append("low_score")
    if second_ratio > 0.3:
        flags.append("multiple_cows")
    if touches_border:
        flags.append("touches_border")
    if area_frac < 0.10 or area_frac > 0.70:
        flags.append("odd_size")
    if fragments_removed:
        flags.append("fragments_removed")

    arr = np.array(img)
    cutout = np.full_like(arr, 255)
    cutout[mask] = arr[mask]
    Image.fromarray((mask * 255).astype(np.uint8)).save(OUT / "masks" / f"{num}.png")
    Image.fromarray(cutout).save(OUT / "cutouts" / f"{num}.png")

    rows.append({
        "Num": num, "score": round(score, 3), "area_frac": round(area_frac, 3),
        "bbox_w": xs.max() - xs.min(), "bbox_h": ys.max() - ys.min(),
        "second_cow_ratio": round(second_ratio, 2),
        "touches_border": touches_border, "flags": ",".join(flags),
    })
    if flags:
        flagged.append((num, arr, mask, ",".join(flags)))
    print(f"#{num}: score {score:.2f}, area {area_frac:.0%}"
          + (f"  FLAGS: {','.join(flags)}" if flags else ""))

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "summary.csv", index=False)
print(f"\nSegmented {summary['score'].notna().sum()} of {len(measurements)} animals; "
      f"{len(flagged)} flagged for review.")

# Contact sheet of flagged animals (up to 24)
if flagged:
    shown = flagged[:24]
    cols = 4
    nrows = int(np.ceil(len(shown) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(5 * cols, 3.5 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (num, arr, mask, why) in zip(axes.ravel(), shown):
        shown_img = arr.copy()
        if mask is not None:
            shown_img[mask] = (0.5 * shown_img[mask]
                               + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        ax.imshow(shown_img)
        ax.set_title(f"#{num}: {why}", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "review_sheet.png", dpi=80)
    plt.show()