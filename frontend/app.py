# ============================================================
# DEEPFAKE AUDIO DETECTION - GRADIO FRONTEND
# ============================================================

import os
import requests
import gradio as gr
from mimetypes import guess_type


# ------------------------------------------------------------
# FASTAPI BACKEND URL
# ------------------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "https://deepfake-audio-detection-8.onrender.com/predict"
)


# ------------------------------------------------------------
# AVAILABLE MODELS
# ------------------------------------------------------------

MODEL_OPTIONS = [
    "LightGBM (Final Model)"
]


# ------------------------------------------------------------
# SEND AUDIO TO FASTAPI FOR PREDICTION
# ------------------------------------------------------------

def get_audio_path(audio_file):

    if isinstance(audio_file, dict):
        audio_path = audio_file.get("path") or audio_file.get("name")
    else:
        audio_path = audio_file

    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(
            "The uploaded audio file could not be found. Please upload it again."
        )

    return audio_path


def detect_audio(audio_file, model_name):

    if audio_file is None:
        return (
            "⚠️ No audio provided",
            "Please upload an audio file or record audio using the microphone."
        )

    if model_name is None:
        return (
            "⚠️ No model selected",
            "Please select a model before analysis."
        )

    try:

        # ----------------------------------------------------
        # GET FILE NAME
        # ----------------------------------------------------

        audio_path = get_audio_path(audio_file)
        filename = os.path.basename(audio_path)
        content_type = guess_type(filename)[0] or "application/octet-stream"

        # ----------------------------------------------------
        # SEND AUDIO + MODEL TO FASTAPI
        # ----------------------------------------------------

        with open(audio_path, "rb") as file:

            response = requests.post(
                API_URL,
                files={
                    "file": (
                        filename,
                        file,
                        content_type
                    )
                },
                data={
                    "model_name": model_name
                },
                timeout=120
            )

        response.raise_for_status()

        result = response.json()

        # ----------------------------------------------------
        # GET RESULTS
        # ----------------------------------------------------

        prediction = result["prediction"]
        confidence = float(
            result["confidence"]
        )

        selected_model = result.get(
            "model",
            model_name
        )

        # ----------------------------------------------------
        # FORMAT PREDICTION
        # ----------------------------------------------------

        if prediction.lower() == "fake":

            prediction_result = (
                "⚠️ FAKE AUDIO DETECTED"
            )

        else:

            prediction_result = (
                "✅ BONAFIDE AUDIO DETECTED"
            )

        confidence_result = (
            f"{confidence:.2f}% confidence\n"
            f"Model: {selected_model}"
        )

        return (
            prediction_result,
            confidence_result
        )

    except requests.exceptions.ConnectionError:

        return (
            "❌ BACKEND CONNECTION ERROR",
            "Please check the FastAPI backend."
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
# CUSTOM CSS
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

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.HTML("""
    <div class="main-title">
        <h1>🎙️ Deepfake Audio Detection System</h1>
        <h3>AI-Based Detection of Bonafide and Synthetic Audio</h3>
    </div>
    """)

    gr.Markdown("""
    <div class="subtitle">
    Upload an audio file or record your voice using the microphone.
    The trained machine learning pipeline will analyze its acoustic
    characteristics and classify it as <b>Bonafide</b> or <b>Fake</b>.
    </div>
    """)


    # --------------------------------------------------------
    # INPUT + RESULTS
    # --------------------------------------------------------

    with gr.Row():

        # ----------------------------------------------------
        # LEFT SIDE
        # ----------------------------------------------------

        with gr.Column(scale=1):

            gr.Markdown(
                "## 🎙️ Audio Input"
            )

            audio_input = gr.Audio(
                sources=[
                    "upload",
                    "microphone"
                ],
                type="filepath",
                label="Upload Audio or Record from Microphone"
            )

            gr.Markdown(
                "## 🤖 Model Selection"
            )

            model_dropdown = gr.Dropdown(
                choices=MODEL_OPTIONS,
                value=MODEL_OPTIONS[0],
                label="Select Detection Model",
                info="Choose the trained model for prediction."
            )

            detect_button = gr.Button(
                "🔍 ANALYZE AUDIO",
                variant="primary",
                size="lg"
            )


        # ----------------------------------------------------
        # RIGHT SIDE
        # ----------------------------------------------------

        with gr.Column(scale=1):

            gr.Markdown(
                "## 🤖 Detection Results"
            )

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
    # MODEL PIPELINE INFORMATION
    # --------------------------------------------------------

    gr.Markdown("""
    ---
    ### 🧠 Model Pipeline

    **Audio Input → Feature Extraction → StandardScaler → PCA → LightGBM → Prediction**

    The system extracts engineered acoustic features from the audio
    and processes them through the selected trained classification model.
    """)


    # --------------------------------------------------------
    # HOW TO USE
    # --------------------------------------------------------

    gr.Markdown("""
    ### 📌 How to Use

    1. Upload an audio file **or record audio using the microphone**.
    2. Select the required detection model.
    3. Click **ANALYZE AUDIO**.
    4. Wait for the system to process the audio.
    5. View the predicted class and model confidence.
    """)


    # --------------------------------------------------------
    # CONNECT BUTTON
    # --------------------------------------------------------

    detect_button.click(
        fn=detect_audio,
        inputs=[
            audio_input,
            model_dropdown
        ],
        outputs=[
            prediction_output,
            confidence_output
        ]
    )


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    gr.Markdown("""
    <div class="footer-text">
    Deepfake Audio Detection | Machine Learning Deployment - Phase 5
    </div>
    """)


# ------------------------------------------------------------
# RUN APPLICATION
# ------------------------------------------------------------

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", 7860)
        )
    )