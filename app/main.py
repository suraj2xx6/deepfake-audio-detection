# ============================================================
# DEEPFAKE AUDIO DETECTION - FASTAPI BACKEND
# ============================================================

from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.predictor import predict_audio


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
# PREDICTION ENDPOINT
# ------------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Allow common audio formats
    allowed_extensions = {".mp3", ".wav", ".flac", ".ogg"}

    # Get uploaded file extension
    file_extension = Path(file.filename).suffix.lower()

    # Validate file type
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file format. "
                "Please upload MP3, WAV, FLAC, or OGG audio."
            )
        )

    # Create a unique filename to avoid conflicts
    unique_filename = (
        f"{uuid.uuid4()}{file_extension}"
    )

    file_path = UPLOAD_DIR / unique_filename

    try:
        # Save uploaded audio temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run complete prediction pipeline
        result = predict_audio(str(file_path))

        # Return prediction as JSON
        return {
            "filename": file.filename,
            "prediction": result["prediction"],
            "confidence": result["confidence"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:
        # Delete temporary uploaded file
        if file_path.exists():
            file_path.unlink()
            