import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold

HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "raw" / "Cattle side and back view images"

# --- Stage A: load data, describe distribution, baseline ---
measurements = pd.read_excel(DATA_DIR / "measurements.xlsx")

print("Number of animals:", len(measurements))
print()
print(measurements["Body weight (kg)"].describe())

plt.hist(measurements["Body weight (kg)"], bins=10, edgecolor="black")
plt.xlabel("Body weight (kg)")
plt.ylabel("Number of animals")
plt.title("Weight distribution across 72 cattle")
plt.show()

baseline_prediction = measurements["Body weight (kg)"].mean()
actual_weights = measurements["Body weight (kg)"]
baseline_errors = (actual_weights - baseline_prediction).abs()
baseline_mae = baseline_errors.mean()

print(f"Baseline prediction (mean weight): {baseline_prediction:.1f} kg")
print(f"Baseline MAE: {baseline_mae:.1f} kg")

# --- Stage B + D (robust version): cross-validated error, by weight band ---
feature_cols = ["Oblique body length (cm)", "Withers height(cm)", "Heart girth(cm)", "Hip length (cm)"]
X = measurements[feature_cols].to_numpy(dtype=float)
y = measurements["Body weight (kg)"].to_numpy(dtype=float)
n = len(y)

N_SPLITS, N_REPEATS = 5, 10
splits = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42).split(y))
print(f"\nRunning {len(splits)} animal-level train/test splits ({N_SPLITS}-fold, repeated {N_REPEATS}x)")

residuals = []
oof_preds = np.zeros((N_REPEATS, n))

for i, (train_idx, test_idx) in enumerate(splits):
    m = LinearRegression().fit(X[train_idx], y[train_idx])
    pred = m.predict(X[test_idx])
    residuals.extend(y[test_idx] - pred)
    oof_preds[i // N_SPLITS, test_idx] = pred

residuals = np.array(residuals)
cv_mae = np.abs(residuals).mean()
avg_pred = oof_preds.mean(axis=0)

print(f"Cross-validated MAE: {cv_mae:.1f} kg")

INTERVAL_LEVEL = 0.80
tail = (1 - INTERVAL_LEVEL) / 2
low_offset = np.quantile(residuals, tail)
high_offset = np.quantile(residuals, 1 - tail)
print(f"\n{INTERVAL_LEVEL:.0%} empirical interval: estimate {low_offset:+.1f} / {high_offset:+.1f} kg")

print("\nError by weight band:")
for name, idx in zip(["light", "middle", "heavy"], np.array_split(np.argsort(y), 3)):
    band_mae = np.abs(y[idx] - avg_pred[idx]).mean()
    band_bias = (avg_pred[idx] - y[idx]).mean()
    print(f"  {name:<6} {y[idx].min():4.0f}-{y[idx].max():4.0f} kg   "
          f"MAE {band_mae:5.1f}   bias {band_bias:+5.1f} kg (+ = over-estimate)")

final_model = LinearRegression().fit(X, y)
print(f"\nFinal shipped model intercept: {final_model.intercept_:.2f}")
for name, coef in zip(feature_cols, final_model.coef_):
    print(f"  {name}: {coef:.2f}")

# --- Check: does a log-log model fix the heavy-animal underestimation? ---
residuals_log = []
oof_preds_log = np.zeros((N_REPEATS, n))

for i, (train_idx, test_idx) in enumerate(splits):
    m = LinearRegression().fit(np.log(X[train_idx]), np.log(y[train_idx]))
    pred_log = np.exp(m.predict(np.log(X[test_idx])))
    residuals_log.extend(y[test_idx] - pred_log)
    oof_preds_log[i // N_SPLITS, test_idx] = pred_log

residuals_log = np.array(residuals_log)
cv_mae_log = np.abs(residuals_log).mean()
avg_pred_log = oof_preds_log.mean(axis=0)

print(f"\nLog-log model cross-validated MAE: {cv_mae_log:.1f} kg (straight-line was {cv_mae:.1f} kg)")
print("Log-log error by weight band:")
for name, idx in zip(["light", "middle", "heavy"], np.array_split(np.argsort(y), 3)):
    band_mae = np.abs(y[idx] - avg_pred_log[idx]).mean()
    band_bias = (avg_pred_log[idx] - y[idx]).mean()
    print(f"  {name:<6} MAE {band_mae:5.1f}   bias {band_bias:+5.1f} kg")    