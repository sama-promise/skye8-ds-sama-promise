"""Loads model.json (written by train_export.py) and makes predictions.

Only needs the standard library, so it stays light enough to run offline.
"""
import json
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model.json"


def load_model(path=MODEL_PATH):
    """Return the model dict, or None if model.json has not been created yet."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def predict(model, key, measurements, level=None):
    """key: which model set ("girth", "length_height", "girth_length", "all").
    measurements: dict {feature name: value in cm}.
    level: share of animals the range should cover (0.80 or 0.90).

    Returns (estimate_kg, low_kg, high_kg). The range is the estimate plus the
    lower/upper error quantiles measured on animals the model had never seen.
    """
    m = model["models"][key]
    if level is None:
        level = model["default_interval_level"]
    interval = m["intervals"][f"{level:.2f}"]
    estimate = m["intercept"]
    for name, coef in zip(m["features"], m["coef"]):
        estimate += coef * float(measurements[name])
    return (estimate,
            estimate + interval["low_offset"],
            estimate + interval["high_offset"])


def band_note(model, key, estimate):
    """A caution if the estimate sits in a weight band where testing showed a
    consistent bias, or None. bias_kg > 0 means the model over-estimated."""
    bands = model["models"][key]["error_by_band"]
    heavy, middle, light = bands["heavy"], bands["middle"], bands["light"]
    if estimate > middle["to_kg"] and heavy["bias_kg"] < 0:
        return (f"Heavier animals ({heavy['from_kg']:.0f}-{heavy['to_kg']:.0f} kg) "
                f"were under-estimated by about {abs(heavy['bias_kg']):.0f} kg on "
                f"average in testing. The true weight is more likely to be above "
                f"this estimate than below it.")
    if estimate < middle["from_kg"] and light["bias_kg"] > 0:
        return (f"Lighter animals ({light['from_kg']:.0f}-{light['to_kg']:.0f} kg) "
                f"were over-estimated by about {light['bias_kg']:.0f} kg on "
                f"average in testing. The true weight is more likely to be below "
                f"this estimate than above it.")
    return None


def out_of_range(model, key, measurements):
    """Messages for inputs outside the range seen in training."""
    m = model["models"][key]
    problems = []
    for name in m["features"]:
        lo, hi = m["feature_range"][name]
        v = float(measurements[name])
        if v < lo or v > hi:
            problems.append(f"{name}: {v:g} is outside the training range {lo:g}-{hi:g}")
    return problems
