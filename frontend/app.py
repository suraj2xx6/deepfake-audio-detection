# ============================================================
# DEEPFAKE AUDIO DETECTION - GRADIO FRONTEND
# ============================================================

import os
import gradio as gr
import requests


# ------------------------------------------------------------
# FASTAPI BACKEND URL
# ------------------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)


# ------------------------------------------------------------
# SEND AUDIO TO FASTAPI FOR PREDICTION
# ------------------------------------------------------------

def detect_audio(audio_file):

    if audio_file is None:
        return (
            "⚠️ No audio file uploaded",
            "Please upload an MP3, WAV, FLAC, or OGG file."
        )

    try:
        filename = os.path.basename(audio_file)

        with open(audio_file, "rb") as file:

            response = requests.post(
                API_URL,
                files={
                    "file": (
                        filename,
                        file,
                        "audio/mpeg"
                    )
                },
                timeout=120
            )

        response.raise_for_status()

        result = response.json()

        prediction = result["prediction"]
        confidence = float(result["confidence"])

        if prediction.lower() == "fake":
            prediction_result = "⚠️ FAKE AUDIO DETECTED"
        else:
            prediction_result = "✅ BONAFIDE AUDIO DETECTED"

        confidence_result = (
            f"{confidence:.2f}% confidence"
        )

        return prediction_result, confidence_result

    except requests.exceptions.ConnectionError:
        return (
            "❌ BACKEND CONNECTION ERROR",
            "Please make sure the FastAPI server is running."
        )

    except requests.exceptions.Timeout:
        return (
            "⏳ REQUEST TIMEOUT",
            "Audio processing took too long. Please try again."
        )

    except Exception as e:
        return (
            "❌ PREDICTION FAILED",
            str(e)
        )


# ------------------------------------------------------------
# CUSTOM CSS FOR PROFESSIONAL DASHBOARD
# ------------------------------------------------------------

custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

.main-title {
    text-align: center;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.footer-text {
    text-align: center;
    margin-top: 25px;
    font-size: 14px;
}
"""


# ------------------------------------------------------------
# CREATE GRADIO DASHBOARD
# ------------------------------------------------------------

with gr.Blocks(
    title="Deepfake Audio Detection System",
    css=custom_css
) as demo:

    # Header
    gr.HTML("""
    <div class="main-title">
        <h1>🎙️ Deepfake Audio Detection System</h1>
        <h3>AI-Based Detection of Bonafide and Synthetic Audio</h3>
    </div>
    """)

    gr.Markdown("""
    <div class="subtitle">
    Upload a raw audio file and the trained machine learning pipeline will
    analyze its acoustic characteristics and classify it as
    <b>Bonafide</b> or <b>Fake</b>.
    </div>
    """)


    with gr.Row():

        # ----------------------------------------------------
        # LEFT SIDE - INPUT
        # ----------------------------------------------------

        with gr.Column(scale=1):

            gr.Markdown("## 📁 Upload Audio")

            audio_input = gr.File(
                label="Supported formats: MP3, WAV, FLAC, OGG",
                file_types=[
                    ".mp3",
                    ".wav",
                    ".flac",
                    ".ogg"
                ],
                type="filepath"
            )

            detect_button = gr.Button(
                "🔍 ANALYZE AUDIO",
                variant="primary",
                size="lg"
            )


        # ----------------------------------------------------
        # RIGHT SIDE - RESULTS
        # ----------------------------------------------------

        with gr.Column(scale=1):

            gr.Markdown("## 🤖 Detection Results")

            prediction_output = gr.Textbox(
                label="Classification",
                placeholder="Prediction will appear here",
                interactive=False
            )

            confidence_output = gr.Textbox(
                label="Model Confidence",
                placeholder="Confidence score will appear here",
                interactive=False
            )


    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    gr.Markdown("""
    ---
    ### 🧠 Model Pipeline

    **Audio Input → Feature Extraction → StandardScaler → PCA → LightGBM → Prediction**

    The system extracts engineered acoustic features from the uploaded audio
    and processes them through the trained classification pipeline.
    """)


    # --------------------------------------------------------
    # APPLICATION USAGE INFORMATION
    # --------------------------------------------------------

    gr.Markdown("""
    ### 📌 How to Use

    1. Upload one audio file in a supported format.
    2. Click **ANALYZE AUDIO**.
    3. Wait for the system to process the audio.
    4. View the predicted class and model confidence.
    """)


    # --------------------------------------------------------
    # CONNECT BUTTON TO FUNCTION
    # --------------------------------------------------------

    detect_button.click(
        fn=detect_audio,
        inputs=audio_input,
        outputs=[
            prediction_output,
            confidence_output
        ]
    )


    # Footer
    gr.Markdown("""
    <div class="footer-text">
    Deepfake Audio Detection | Machine Learning Deployment - Phase 5
    </div>
    """)


# ------------------------------------------------------------
# RUN APPLICATION
# ------------------------------------------------------------

if __name__ == "__main__":
    demo.launch()