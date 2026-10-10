import streamlit as st
from PIL import Image, ImageOps

from weight_model import band_note, load_model, out_of_range, predict

# Result of the photo-only experiment (Google Colab, 5-fold cross-validation on
# the 71 animals that have a side-view photo). Average error in kg.
PHOTO_TEST = {"guess the average": 72.7, "raw photo": 74.7, "cutout": 69.7}

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="PictureWeight Estimator",
    page_icon="⚖️",
    layout="centered",
)

model = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("🐄 PictureWeight Estimator")
st.write("Get an estimated live weight, with a range, for a cow.")

if model is None:
    st.error(
        "No trained model found. Run `python train_export.py` once in this "
        "folder to create `model.json`, then restart the app."
    )
    st.stop()

if "models" not in model or "interval_levels" not in model:
    st.error(
        "`model.json` is from an older version. Run `python train_export.py` "
        "again, then restart the app."
    )
    st.stop()

st.info(
    "ℹ️ **The estimate comes from body measurements. The photo is not used.** "
    f"In cross-validated tests on {model['n_animals'] - 1} photographed cattle, "
    f"a photo-only model was off by {PHOTO_TEST['raw photo']:.0f} kg (raw photo) "
    f"or {PHOTO_TEST['cutout']:.0f} kg (background removed), against "
    f"{PHOTO_TEST['guess the average']:.0f} kg for simply guessing the average. "
    "A photo does not show how far away the camera was, so it cannot show size "
    "without something of known length in the frame."
)

# -----------------------------
# Animal selection
# -----------------------------
animal_type = st.selectbox(
    "Select animal type",
    ["Cow", "Goat", "Sheep", "Pig", "Other"],
)

if animal_type != "Cow":
    st.warning(
        f"The model was trained on {model['n_animals']} cattle only, so it "
        f"cannot estimate the weight of a {animal_type.lower()}."
    )
    st.stop()

# -----------------------------
# Photo guidance + optional photo (for your records only)
# -----------------------------
with st.expander("📸 How to take a good photo", expanded=False):
    st.markdown(
        "- Stand **directly side-on** to the animal, at the height of its back.\n"
        "- Stand about **5–8 metres** away, so the whole animal fills most of the frame.\n"
        "- Keep the **whole animal in frame**: nose to tail, hooves to back.\n"
        "- Animal **standing square** on level ground, head up, not walking.\n"
        "- Even daylight, no strong shadows, nobody or nothing in front of it.\n"
        "- Phone held **landscape**, not tilted."
    )

uploaded_file = st.file_uploader(
    "Add a side-view photo for your records (optional, not used in the estimate)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    st.subheader("Uploaded Image")
    st.image(image, caption=f"Selected animal: {animal_type}")

st.divider()

# -----------------------------
# Measurements
# -----------------------------
st.subheader("Body measurements (cm)")

# Most accurate set first, so it is the default
keys = sorted(model["models"], key=lambda k: model["models"][k]["mae_kg"])
key = st.radio(
    "What can you measure?",
    keys,
    format_func=lambda k: (
        f"{model['models'][k]['label']}  "
        f"(typical error about {model['models'][k]['mae_kg']:.0f} kg)"
    ),
)
chosen = model["models"][key]
st.caption("Measure with a tape on the standing animal. "
           "More measurements give a more accurate estimate.")

values = {}
cols = st.columns(2)
for i, name in enumerate(chosen["features"]):
    lo, hi = chosen["feature_range"][name]
    with cols[i % 2]:
        values[name] = st.number_input(
            name,
            min_value=1.0,
            max_value=400.0,
            value=float(round((lo + hi) / 2)),
            step=1.0,
            key=f"{key}-{name}",
            help=f"Animals used for training ranged from {lo:g} to {hi:g} cm.",
        )

level = st.radio(
    "How sure should the range be?",
    model["interval_levels"],
    index=model["interval_levels"].index(model["default_interval_level"]),
    format_func=lambda lv: (
        f"{lv:.0%} - true weight inside the range about {round(lv * 10)} "
        f"times out of 10" + (" (narrower)" if lv == min(model["interval_levels"])
                              else " (wider)")
    ),
)

# -----------------------------
# Estimate
# -----------------------------
if st.button("Estimate Weight", type="primary"):
    estimated, low, high = predict(model, key, values, level)
    problems = out_of_range(model, key, values)
    note = band_note(model, key, estimated)

    st.success("Weight estimation completed!")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Estimated Weight", f"{estimated:.0f} kg")
    with col2:
        st.metric("Estimated Range", f"{low:.0f}–{high:.0f} kg")

    st.write(f"### 🐄 {animal_type}")
    st.write(
        f"The estimated live weight is approximately **{estimated:.0f} kg**. "
        f"Across many similar, previously unseen cattle, ranges built this way "
        f"are designed to contain the true weight about {level:.0%} of the time. "
        f"This is a statistical target, not a guarantee for this individual animal."

    )

    if note:
        st.warning(note)

    if problems:
        st.warning(
            "Some measurements are outside what the model was trained on, so "
            "this estimate is less reliable:\n\n- " + "\n- ".join(problems)
        )

    st.caption(
        f"Typical error on unseen animals: about {chosen['mae_kg']:.0f} kg. "
        f"Guessing the average weight every time would be off by about "
        f"{model['baseline_mae_kg']:.0f} kg. Trained on {model['n_animals']} "
        f"cattle weighing {model['weight_range_kg'][0]:.0f}–"
        f"{model['weight_range_kg'][1]:.0f} kg."
    )

# -----------------------------
# Limits of use
# -----------------------------
st.divider()
st.error(
    "**Do not use this estimate to calculate a drug dose.** It is suitable "
    "for negotiating a sale price only. Medicine must be dosed from a "
    "weighed animal or a veterinarian's advice."
)
