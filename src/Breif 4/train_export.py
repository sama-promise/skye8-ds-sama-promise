"""Fit the measurement-based weight models and export them for the app.

Run from the project folder (where data/raw/... lives):

    python train_export.py
    python train_export.py path/to/measurements.xlsx

Writes model.json next to this file. It holds four models so the app can work
with whatever the herder is willing to measure:

    girth only | body length + withers height | girth + body length | all four

Each model carries its own cross-validated error, an 80% and a 90% range, and
its error for light / middle / heavy animals. The app never needs the dataset,
pandas, scikit-learn or torch at run time.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

HERE = Path(__file__).parent
DEFAULT_XLSX = (HERE / "data" / "raw" / "Cattle side and back view images"
                / "measurements.xlsx")

FEATURES = ["Oblique body length (cm)", "Withers height(cm)",
            "Heart girth(cm)", "Hip length (cm)"]
TARGET = "Body weight (kg)"

# key -> (label shown in the app, feature indexes into FEATURES)
MODEL_SETS = {
    "girth": ("Heart girth only", [2]),
    "length_height": ("Body length + withers height", [0, 1]),
    "girth_length": ("Heart girth + body length", [2, 0]),
    "all": ("All four measurements", [2, 0, 1, 3]),
}

# Share of animals each range should cover. 0.80 -> "8 times out of 10".
INTERVAL_LEVELS = [0.80, 0.90]
DEFAULT_LEVEL = 0.80
N_SPLITS = 5

xlsx = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XLSX
data = pd.read_excel(xlsx).dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
X = data[FEATURES].to_numpy(dtype=float)
y = data[TARGET].to_numpy(dtype=float)
n = len(y)
print(f"Animals: {n}  (one row per animal, so every split below is by animal)")
print(f"Weight: mean {y.mean():.1f} kg, min {y.min():.0f}, max {y.max():.0f}")

from sklearn.model_selection import train_test_split

indices = np.arange(len(y))

train_idx, cal_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42
)

X_train, y_train = X[train_idx], y[train_idx]
X_cal, y_cal = X[cal_idx], y[cal_idx]

print(f"Training animals: {len(train_idx)}")
print(f"Calibration animals: {len(cal_idx)}")

splits = list(KFold(n_splits=N_SPLITS, shuffle=True,
                    random_state=42).split(y))
# Baseline: predict the training-fold mean for everyone
baseline_mae = float(np.mean([np.abs(y[te] - y[tr].mean()).mean()
                              for tr, te in splits]))
print(f"\nBaseline (predict the mean) MAE: {baseline_mae:.1f} kg")


def cv_eval(cols, use_log=False):
    """Return out-of-fold residuals, mean fold MAE, and predictions."""
    Xs = X[:, cols]
    Xf, yf = (np.log(Xs), np.log(y)) if use_log else (Xs, y)

    residuals = np.zeros(n)
    maes = []
    oof = np.zeros(n)

    for tr, te in splits:
        model = LinearRegression().fit(Xf[tr], yf[tr])
        p = model.predict(Xf[te])
        p = np.exp(p) if use_log else p

        oof[te] = p
        residuals[te] = y[te] - p
        maes.append(np.abs(y[te] - p).mean())

    return residuals, float(np.mean(maes)), oof


def calibrate_model(cols):
    """Fit on training animals and calculate errors on unseen calibration animals."""
    model = LinearRegression().fit(X_train[:, cols], y_train)
    predictions = model.predict(X_cal[:, cols])

    residuals = np.abs(y_cal - predictions)

    return model, residuals


def conformal_quantile(residuals, coverage=0.90):
    """Calculate the finite-sample conformal error threshold."""
    residuals = np.sort(np.asarray(residuals))
    n_cal = len(residuals)

    rank = int(np.ceil((n_cal + 1) * coverage))

    if rank > n_cal:
        return float("inf")

    return float(residuals[rank - 1])

def band_errors(avg_pred):
    """Error for the lightest, middle and heaviest third of animals."""
    out = {}
    for name, idx in zip(["light", "middle", "heavy"],
                         np.array_split(np.argsort(y), 3)):
        out[name] = {"from_kg": float(y[idx].min()), "to_kg": float(y[idx].max()),
                     "mae_kg": float(np.abs(y[idx] - avg_pred[idx]).mean()),
                     "bias_kg": float((avg_pred[idx] - y[idx]).mean())}
    return out



models = {}

for key, (label, cols) in MODEL_SETS.items():
    _, mae, avg_pred = cv_eval(cols)
    final, calibration_residuals = calibrate_model(cols)

    intervals = {}
    for level in INTERVAL_LEVELS:
        q = conformal_quantile(calibration_residuals, level)
        intervals[f"{level:.2f}"] = {
            "low_offset": float(-q),
            "high_offset": float(q),
        }

    bands = band_errors(avg_pred)
    names = [FEATURES[c] for c in cols]

    models[key] = {
        "label": label,
        "features": names,
        "coef": [float(c) for c in final.coef_],
        "intercept": float(final.intercept_),
        "feature_range": {
            nm: [float(X_train[:, c].min()), float(X_train[:, c].max())]
            for nm, c in zip(names, cols)
        },
        "mae_kg": mae,
        "intervals": intervals,
        "error_by_band": bands,
    }

    print(f"\n== {label}")
    print(f"   Error (CV): {mae:.1f} kg")
    for lv, iv in intervals.items():
        print(f"   {float(lv):.0%} range: estimate "
              f"{iv['low_offset']:+.0f} / {iv['high_offset']:+.0f} kg")


out = {
    "n_animals": int(n),
    "weight_range_kg": [float(y.min()), float(y.max())],
    "baseline_mae_kg": baseline_mae,
    "interval_levels": INTERVAL_LEVELS,
    "default_interval_level": DEFAULT_LEVEL,
    "models": models,
}

path = HERE / "model.json"
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"\nSaved {path}")        