from app.predictor import predict_audio

audio_path = "common_voice_en_34925858.mp3"

try:
    result = predict_audio(audio_path)

    print("\n" + "=" * 50)
    print("DEEPFAKE AUDIO DETECTION TEST")
    print("=" * 50)
    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']}%")
    print("=" * 50)

except Exception as e:
    print("\nERROR:")
    print(e)