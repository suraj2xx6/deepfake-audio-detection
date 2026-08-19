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

# This file is inside: deepfake_audio_app/app/
APP_DIR = Path(__file__).resolve().parent

# Project root: deepfake_audio_app/
PROJECT_ROOT = APP_DIR.parent

# Models directory
MODELS_DIR = PROJECT_ROOT / "models"


# ------------------------------------------------------------
# LOAD TRAINED ARTIFACTS
# ------------------------------------------------------------

model = joblib.load(
    MODELS_DIR / "final_lightgbm_model.pkl"
)

scaler = joblib.load(
    MODELS_DIR / "scaler.pkl"
)

pca = joblib.load(
    MODELS_DIR / "pca.pkl"
)

label_encoder = joblib.load(
    MODELS_DIR / "label_encoder.pkl"
)

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
# PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_audio(file_path):
    """
    Perform end-to-end prediction on an uploaded audio file.

    Pipeline:
    Audio File
        ↓
    Feature Extraction
        ↓
    StandardScaler
        ↓
    PCA
        ↓
    LightGBM
        ↓
    Bonafide / Fake Prediction
    """

    # Step 1: Extract the 20 audio features
    features = extract_features(file_path)

    # Step 2: Create DataFrame using exact training feature order
    features_df = pd.DataFrame(
        [features],
        columns=feature_names
    )

    # Step 3: Apply the saved scaler
    features_scaled = scaler.transform(features_df)

    # Step 4: Apply the saved PCA transformation
    features_pca = pca.transform(features_scaled)

    # Step 5: Make prediction
    prediction_encoded = model.predict(features_pca)[0]

    # Step 6: Get prediction probabilities
    probabilities = model.predict_proba(features_pca)[0]

    # Step 7: Convert encoded prediction to original label
    prediction_label = label_encoder.inverse_transform(
        [prediction_encoded]
    )[0]

    # Step 8: Calculate confidence
    confidence = float(
        probabilities[prediction_encoded] * 100
    )

    # Return result in a clean dictionary
    return {
        "prediction": prediction_label,
        "confidence": round(confidence, 2)
    }