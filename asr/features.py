import numpy as np
import librosa
from config import *


def mfcc(audio: np.ndarray) -> np.ndarray:
    """
    Computing mel-frequency cepstral coefficients

    audio: raw audio samples
    return: mfcc
    """

    features = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=num_feature_filters,
        n_mels=128,
        fmin=0,
        fmax=None,
        hop_length=int(step_len * sample_rate),
        n_fft=int(window_len * sample_rate),
        center=True,
    )
    
    # Computing root-mean-square (RMS) value for each frame
    rmse = librosa.feature.rms(
        y=audio,
        hop_length=int(step_len * sample_rate),
        frame_length=int(window_len * sample_rate),
        center=True,
    )
    features[0] = rmse
    features = features.transpose().astype("float32")
    return features
