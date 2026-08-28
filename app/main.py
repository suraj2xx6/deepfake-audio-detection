# ============================================================
# DEEPFAKE AUDIO DETECTION - FASTAPI BACKEND
# ============================================================

from pathlib import Path
import shutil
import uuid

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from app.predictor import (
    predict_audio,
    AVAILABLE_MODELS
)


# ------------------------------------------------------------
# CREATE FASTAPI APPLICATION
# ------------------------------------------------------------

app = FastAPI(
    title="Deepfake Audio Detection API",
    description="API for detecting whether uploaded audio is Bonafide or Deepfake.",
    version="1.0.0"
)


# ------------------------------------------------------------
# ENABLE CORS
# ------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# HEALTH CHECK ENDPOINT
# ------------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Deepfake Audio Detection API is running successfully!"
    }


# ------------------------------------------------------------
# AVAILABLE MODELS ENDPOINT
# ------------------------------------------------------------

@app.get("/models")
def get_models():
    return {
        "models": list(AVAILABLE_MODELS.keys())
    }


# ------------------------------------------------------------
# PREDICTION ENDPOINT
# ------------------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("LightGBM (Final Model)")
):

    # --------------------------------------------------------
    # ALLOWED AUDIO FORMATS
    # --------------------------------------------------------

    allowed_extensions = {
        ".mp3",
        ".wav",
        ".flac",
        ".ogg"
    }

    # Get uploaded file extension
    file_extension = Path(
        file.filename
    ).suffix.lower()

    # Validate file type
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file format. "
                "Please upload MP3, WAV, FLAC, or OGG audio."
            )
        )

    # --------------------------------------------------------
    # VALIDATE MODEL
    # --------------------------------------------------------

    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Selected model '{model_name}' "
                "is not available."
            )
        )

    # --------------------------------------------------------
    # CREATE UNIQUE TEMPORARY FILE
    # --------------------------------------------------------

    unique_filename = (
        f"{uuid.uuid4()}{file_extension}"
    )

    file_path = UPLOAD_DIR / unique_filename

    try:

        # ----------------------------------------------------
        # SAVE UPLOADED AUDIO TEMPORARILY
        # ----------------------------------------------------

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        # ----------------------------------------------------
        # RUN PREDICTION
        # ----------------------------------------------------

        result = predict_audio(
            str(file_path),
            model_name
        )

        # ----------------------------------------------------
        # RETURN JSON RESULT
        # ----------------------------------------------------

        return {
            "filename": file.filename,
            "model": result["model"],
            "prediction": result["prediction"],
            "confidence": result["confidence"]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:

        # ----------------------------------------------------
        # DELETE TEMPORARY AUDIO FILE
        # ----------------------------------------------------

        if file_path.exists():
            file_path.unlink()


# ------------------------------------------------------------
# RUN FASTAPI SERVER
# ------------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )