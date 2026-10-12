from __future__ import annotations
import numpy as np
from scipy import stats
import librosa

def _safe(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0

def _entropy(x: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(x, bins=bins)
    prob = hist.astype(np.float64)
    prob = prob / max(prob.sum(), 1.0)
    prob = prob[prob > 0]
    return _safe(-np.sum(prob * np.log2(prob)))

def extract_audio_features(
    segment: np.ndarray,
    sample_rate: int = 16000,
    bands: tuple[tuple[int, int], ...] = (
        (0, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 4000),
        (4000, 8000),
    ),
) -> dict[str, float]:
    x = np.asarray(segment, dtype=np.float32)
    if x.size == 0:
        x = np.zeros(1, dtype=np.float32)

    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(x**2) + 1e-12))
    peak = float(np.max(abs_x))
    features: dict[str, float] = {
        "audio_mean": _safe(np.mean(x)),
        "audio_std": _safe(np.std(x)),
        "audio_rms": _safe(rms),
        "audio_peak": _safe(peak),
        "audio_ptp": _safe(np.ptp(x)),
        "audio_zcr": _safe(np.mean(librosa.feature.zero_crossing_rate(x)[0])),
        "audio_skewness": _safe(stats.skew(x)),
        "audio_kurtosis": _safe(stats.kurtosis(x)),
        "audio_crest_factor": _safe(peak / (rms + 1e-12)),
        "audio_temporal_entropy": _safe(_entropy(x)),
    }

    n_fft = min(2048, max(256, int(2 ** np.ceil(np.log2(max(len(x), 2))))))
    magnitude = np.abs(np.fft.rfft(x, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    power = magnitude**2
    total_power = float(np.sum(power) + 1e-12)
    centroid = float(np.sum(freqs * power) / total_power)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
    cumulative = np.cumsum(power)
    rolloff_idx = min(int(np.searchsorted(cumulative, 0.85 * total_power)), len(freqs) - 1)
    prob = power / total_power
    onset_env = librosa.onset.onset_strength(y=x, sr=sample_rate)

    features.update(
        {
            "audio_spectral_centroid": _safe(centroid),
            "audio_spectral_bandwidth": _safe(bandwidth),
            "audio_spectral_rolloff": _safe(freqs[rolloff_idx]),
            "audio_spectral_flatness": _safe(
                np.exp(np.mean(np.log(magnitude + 1e-12))) / (np.mean(magnitude) + 1e-12)
            ),
            "audio_spectral_flux": _safe(np.mean(onset_env) if onset_env.size else 0.0),
            "audio_spectral_entropy": _safe(-np.sum(prob[prob > 0] * np.log2(prob[prob > 0]))),
            "audio_dominant_frequency": _safe(freqs[int(np.argmax(power))]),
        }
    )

    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        energy = float(np.sum(power[mask]))
        features[f"audio_band_energy_{low}_{high}"] = _safe(energy)
        features[f"audio_band_ratio_{low}_{high}"] = _safe(energy / total_power)

    mfcc = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    for name, mat in [("mfcc", mfcc), ("delta_mfcc", delta), ("delta2_mfcc", delta2)]:
        for i in range(mat.shape[0]):
            features[f"audio_{name}_{i + 1:02d}_mean"] = _safe(np.mean(mat[i]))
            features[f"audio_{name}_{i + 1:02d}_std"] = _safe(np.std(mat[i]))

    return features
