from __future__ import annotations
import numpy as np
from scipy import signal, stats

def _safe(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0

def _entropy(x: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(x, bins=bins)
    prob = hist.astype(np.float64)
    prob = prob / max(prob.sum(), 1.0)
    prob = prob[prob > 0]
    return _safe(-np.sum(prob * np.log2(prob)))

def _time_features(x: np.ndarray, prefix: str) -> dict[str, float]:
    q75, q25 = np.percentile(x, [75, 25])
    return {
        f"{prefix}_mean": _safe(np.mean(x)),
        f"{prefix}_std": _safe(np.std(x)),
        f"{prefix}_var": _safe(np.var(x)),
        f"{prefix}_rms": _safe(np.sqrt(np.mean(x**2) + 1e-12)),
        f"{prefix}_max": _safe(np.max(x)),
        f"{prefix}_min": _safe(np.min(x)),
        f"{prefix}_median": _safe(np.median(x)),
        f"{prefix}_iqr": _safe(q75 - q25),
        f"{prefix}_ptp": _safe(np.ptp(x)),
        f"{prefix}_skewness": _safe(stats.skew(x)),
        f"{prefix}_kurtosis": _safe(stats.kurtosis(x)),
        f"{prefix}_energy": _safe(np.sum(x**2)),
        f"{prefix}_temporal_entropy": _safe(_entropy(x)),
    }

def _freq_features(
    x: np.ndarray,
    sample_rate: int,
    prefix: str,
    bands: tuple[tuple[float, float | None], ...],
) -> dict[str, float]:
    if len(x) < 4:
        x = np.pad(x, (0, 4 - len(x)))
    freqs, psd = signal.welch(x, fs=sample_rate, nperseg=min(256, len(x)))
    total = float(np.sum(psd) + 1e-12)
    centroid = float(np.sum(freqs * psd) / total)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / total))
    prob = psd / total
    cumulative = np.cumsum(psd)
    rolloff_idx = min(int(np.searchsorted(cumulative, 0.85 * total)), len(freqs) - 1)

    features = {
        f"{prefix}_dominant_freq": _safe(freqs[int(np.argmax(psd))]),
        f"{prefix}_spectral_centroid": _safe(centroid),
        f"{prefix}_spectral_bandwidth": _safe(bandwidth),
        f"{prefix}_spectral_entropy": _safe(-np.sum(prob[prob > 0] * np.log2(prob[prob > 0]))),
        f"{prefix}_spectral_rolloff": _safe(freqs[rolloff_idx]),
        f"{prefix}_spectral_energy": _safe(total),
    }
    low_energy = 0.0
    high_energy = 0.0
    for low, high in bands:
        if high is None:
            mask = freqs >= low
            name = f"{int(low)}_inf"
        else:
            mask = (freqs >= low) & (freqs < high)
            name = f"{int(low)}_{int(high)}"
        energy = float(np.sum(psd[mask]))
        features[f"{prefix}_band_power_{name}"] = _safe(energy)
        if high is not None and high <= 100:
            low_energy += energy
        if low >= 100:
            high_energy += energy
    features[f"{prefix}_high_low_ratio"] = _safe(high_energy / (low_energy + 1e-12))
    return features

def _jerk_features(x: np.ndarray, prefix: str) -> dict[str, float]:
    jerk = np.diff(x)
    if jerk.size == 0:
        jerk = np.zeros(1, dtype=np.float32)
    return {
        f"{prefix}_jerk_mean": _safe(np.mean(jerk)),
        f"{prefix}_jerk_std": _safe(np.std(jerk)),
        f"{prefix}_jerk_rms": _safe(np.sqrt(np.mean(jerk**2) + 1e-12)),
        f"{prefix}_jerk_max": _safe(np.max(jerk)),
        f"{prefix}_jerk_energy": _safe(np.sum(jerk**2)),
    }

def extract_accel_features(accel_t3: np.ndarray, sample_rate: int = 1000) -> dict[str, float]:
    x = np.asarray(accel_t3, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("accel_t3 must be [T, 3]，column order is x、y、z")
    if x.shape[0] == 0:
        x = np.zeros((1, 3), dtype=np.float32)

    bands = ((0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, None))
    features: dict[str, float] = {}
    magnitude = np.linalg.norm(x, axis=1, keepdims=True)
    signals = np.concatenate([x, magnitude], axis=1)
    for i, name in enumerate(["ax", "ay", "az", "mag"]):
        prefix = f"accel_{name}"
        features.update(_time_features(signals[:, i], prefix))
        features.update(_freq_features(signals[:, i], sample_rate, prefix, bands))
        features.update(_jerk_features(signals[:, i], prefix))

    ax, ay, az = x[:, 0], x[:, 1], x[:, 2]
    features["accel_corr_ax_ay"] = _safe(np.corrcoef(ax, ay)[0, 1])
    features["accel_corr_ax_az"] = _safe(np.corrcoef(ax, az)[0, 1])
    features["accel_corr_ay_az"] = _safe(np.corrcoef(ay, az)[0, 1])
    cov = np.cov(np.stack([ax, ay, az], axis=0))
    for i in range(3):
        for j in range(3):
            features[f"accel_cov_{i}_{j}"] = _safe(cov[i, j])
    features["accel_sma"] = _safe(np.mean(np.abs(ax) + np.abs(ay) + np.abs(az)))
    energies = np.array([np.sum(ax**2), np.sum(ay**2), np.sum(az**2)], dtype=np.float64)
    total = float(np.sum(energies) + 1e-12)
    for name, energy in zip(["x", "y", "z"], energies):
        features[f"accel_axis_energy_ratio_{name}"] = _safe(energy / total)
    return features
