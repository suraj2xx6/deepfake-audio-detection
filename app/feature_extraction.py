import librosa
import numpy as np
import pandas as pd


def extract_features(file_path):
    """
    Extract the 20 audio features required by the trained
    Deepfake Audio Detection model.

    The feature order matches the Phase 1 training dataset.
    """

    # Load audio at the same sampling rate used during training
    signal, sr = librosa.load(file_path, sr=22050)

    # 1. MFCC features (13)
    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13
    )
    mfcc_mean = np.mean(mfccs, axis=1)

    # 2. Zero Crossing Rate
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(signal)
    )

    # 3. RMS Energy
    rms = np.mean(
        librosa.feature.rms(y=signal)
    )

    # 4. Spectral Centroid
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=signal,
            sr=sr
        )
    )

    # 5. Spectral Bandwidth
    spectral_bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(
            y=signal,
            sr=sr
        )
    )

    # 6. Spectral Rolloff
    spectral_rolloff = np.mean(
        librosa.feature.spectral_rolloff(
            y=signal,
            sr=sr
        )
    )

    # 7. Chroma
    chroma = np.mean(
        librosa.feature.chroma_stft(
            y=signal,
            sr=sr
        )
    )

    # 8. Beat Tempo
    tempo = librosa.beat.beat_track(
        y=signal,
        sr=sr
    )[0]

    # Combine all values in the same positional order
    # used by the training dataset
    features = np.hstack([
        mfcc_mean,
        zcr,
        rms,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        chroma,
        tempo
    ])

    return features