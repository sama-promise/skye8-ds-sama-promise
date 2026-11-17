
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image


# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Cattle Weight Estimator",
    page_icon="🐄",
    layout="centered",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "cnn_conformal_90.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. LOAD THE TRAINED MODEL
# ==========================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at:\n{MODEL_PATH}\n\n"
            "Create a 'models' folder beside app_2.py and place "
            "cnn_conformal_90.pt inside it."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 1),
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    train_mean = checkpoint.get(
        "train_mean", checkpoint.get("mean")
    )
    train_std = checkpoint.get(
        "train_std", checkpoint.get("std")
    )
    q90 = checkpoint.get(
        "q90", checkpoint.get("q90_threshold")
    )

    if train_mean is None or train_std is None or q90 is None:
        raise KeyError(
            "The model file does not contain all required values. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    if float(train_std) <= 0 or float(q90) < 0:
        raise ValueError("Invalid target scaling or conformal threshold.")

    return model, float(train_mean), float(train_std), float(q90)


# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================
image_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ==========================================
# 4. PREDICTION FUNCTION
# ==========================================
def predict_weight(image, model, train_mean, train_std, q90):
    image = image.convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        scaled_prediction = model(image_tensor).item()

    predicted_kg = scaled_prediction * train_std + train_mean

    lower_kg = predicted_kg - q90
    upper_kg = predicted_kg + q90

    return predicted_kg, lower_kg, upper_kg


# ==========================================
# 5. STREAMLIT INTERFACE
# ==========================================
st.title("🐄 Cattle Weight Estimator")
st.write(
    "Upload a clear side-view photo of a cow to estimate its live weight."
)

st.info(
    "Experimental model: the CNN's held-out test MAE was 94.31 kg. "
    "Use these results for testing, not as a verified market weight."
)

uploaded_file = st.file_uploader(
    "Choose a cattle image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded cattle image",
            use_container_width=True,
        )

        if st.button("Estimate Weight", type="primary"):
            with st.spinner("Loading model and estimating weight..."):
                model, train_mean, train_std, q90 = load_model()

                predicted_kg, lower_kg, upper_kg = predict_weight(
                    image,
                    model,
                    train_mean,
                    train_std,
                    q90,
                )

            st.subheader("Estimated result")

            st.metric(
                "Predicted weight",
                f"{predicted_kg:.1f} kg",
            )

            left, right = st.columns(2)
            left.metric("Lower interval bound", f"{lower_kg:.1f} kg")
            right.metric("Upper interval bound", f"{upper_kg:.1f} kg")

            st.caption(
                f"90% conformal interval using a calibration threshold "
                f"of {q90:.1f} kg. Nominal coverage is a population-level "
                "target under the calibration assumptions, not a guarantee "
                "for an individual animal."
            )

            st.warning(
                "Confirm estimates with a scale or validated measurement "
                "method when possible."
            )

    except Exception as error:
        st.error(f"Unable to process the image: {error}")
        st.exception(error)

st.divider()
st.caption(f"Device: {DEVICE}")