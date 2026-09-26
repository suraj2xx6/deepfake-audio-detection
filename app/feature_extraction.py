import librosa
import numpy as np


def extract_features(file_path):
    """Extract the 20 features used to train the saved model."""
    signal, sr = librosa.load(
        file_path,
        sr=22050,
        mono=True
    )

    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13
    )
    mfcc_mean = np.mean(mfccs, axis=1)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal))
    rms = np.mean(librosa.feature.rms(y=signal))
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(y=signal, sr=sr)
    )
    spectral_bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    )
    spectral_rolloff = np.mean(
        librosa.feature.spectral_rolloff(y=signal, sr=sr)
    )
    chroma = np.mean(
        librosa.feature.chroma_stft(y=signal, sr=sr)
    )
    tempo = librosa.beat.beat_track(y=signal, sr=sr)[0]
    tempo = float(np.asarray(tempo).reshape(-1)[0])

    return np.hstack([
        mfcc_mean,
        zcr,
        rms,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        chroma,
        tempo
    ])