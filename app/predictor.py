# ============================================================
# DEEPFAKE AUDIO DETECTION - PREDICTION ENGINE
# ============================================================

from pathlib import Path
import joblib
import pandas as pd

from app.feature_extraction import extract_features


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"


# ------------------------------------------------------------
# FEATURE NAMES
# ------------------------------------------------------------

feature_names = [
    "MFCC_1",
    "MFCC_2",
    "MFCC_3",
    "MFCC_4",
    "MFCC_5",
    "MFCC_6",
    "MFCC_7",
    "MFCC_8",
    "MFCC_9",
    "MFCC_10",
    "MFCC_11",
    "MFCC_12",
    "MFCC_13",
    "Zero_Crossing_Rate",
    "RMS_Energy",
    "Spectral_Centroid",
    "Spectral_Bandwidth",
    "Spectral_Rolloff",
    "Tempo",
    "Beat_Frames"
]


# ------------------------------------------------------------
# LOAD COMMON PREPROCESSING ARTIFACTS
# ------------------------------------------------------------

scaler = joblib.load(
    MODELS_DIR / "scaler.pkl"
)

pca = joblib.load(
    MODELS_DIR / "pca.pkl"
)

label_encoder = joblib.load(
    MODELS_DIR / "label_encoder.pkl"
)


# ------------------------------------------------------------
# AVAILABLE MODELS
# ------------------------------------------------------------

AVAILABLE_MODELS = {
    "LightGBM (Final Model)": MODELS_DIR / "final_lightgbm_model.pkl"
}


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

def load_model(model_name):
    """
    Load the selected trained model.
    """

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not available."
        )

    model_path = AVAILABLE_MODELS[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path.name}"
        )

    return joblib.load(model_path)


# ------------------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_audio(file_path, model_name="LightGBM (Final Model)"):
    """
    Perform end-to-end prediction on an audio file.

    Pipeline:
    Audio File
        ↓
    Feature Extraction
        ↓
    StandardScaler
        ↓
    PCA
        ↓
    Selected ML Model
        ↓
    Bonafide / Fake Prediction
    """

    # Step 1: Load selected model
    model = load_model(model_name)

    # Step 2: Extract audio features
    features = extract_features(file_path)

    # Step 3: Create DataFrame using exact training feature order
    features_df = pd.DataFrame(
        [features],
        columns=feature_names
    )

    # Step 4: Apply saved scaler
    features_scaled = scaler.transform(features_df)

    # Step 5: Apply saved PCA
    features_pca = pca.transform(features_scaled)

    # Step 6: Make prediction
    prediction_encoded = model.predict(
        features_pca
    )[0]

    # Step 7: Get probabilities
    probabilities = model.predict_proba(
        features_pca
    )[0]

    # Step 8: Convert encoded prediction to original label
    prediction_label = label_encoder.inverse_transform(
        [prediction_encoded]
    )[0]

    # Step 9: Calculate confidence
    confidence = float(
        probabilities[prediction_encoded] * 100
    )

    # Step 10: Return result
    return {
        "prediction": prediction_label,
        "confidence": round(confidence, 2),
        "model": model_name
    }