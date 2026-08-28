import librosa
import numpy as np


def extract_features(file_path):
    """
    Extract the exact 20 features used during model training.

    Feature order:
    13 MFCCs
    Zero Crossing Rate
    RMS Energy
    Spectral Centroid
    Spectral Bandwidth
    Spectral Rolloff
    Tempo
    Beat Frames
    """

    # Load audio at the training sampling rate
    signal, sr = librosa.load(
        file_path,
        sr=22050,
        mono=True
    )

    # 1–13. MFCC
    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(
        mfccs,
        axis=1
    )

    # 14. Zero Crossing Rate
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(
            y=signal
        )
    )

    # 15. RMS Energy
    rms = np.mean(
        librosa.feature.rms(
            y=signal
        )
    )

    # 16. Spectral Centroid
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=signal,
            sr=sr
        )
    )

    # 17. Spectral Bandwidth
    spectral_bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(
            y=signal,
            sr=sr
        )
    )

    # 18. Spectral Rolloff
    spectral_rolloff = np.mean(
        librosa.feature.spectral_rolloff(
            y=signal,
            sr=sr
        )
    )

    # 19. Tempo
    tempo = librosa.beat.beat_track(
        y=signal,
        sr=sr
    )[0]

    # Convert tempo to scalar
    tempo = float(
        np.asarray(tempo).reshape(-1)[0]
    )

    # 20. Beat Frames
    _, beat_frames = librosa.beat.beat_track(
        y=signal,
        sr=sr
    )

    # Training data stored Beat_Frames as a numerical value.
    # Use the mean beat-frame position when beats are available.
    if len(beat_frames) > 0:
        beat_frames_value = float(
            np.mean(beat_frames)
        )
    else:
        beat_frames_value = 0.0

    # Combine exactly 20 features
    features = np.hstack([
        mfcc_mean,
        zcr,
        rms,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        tempo,
        beat_frames_value
    ])

    return features