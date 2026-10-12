from __future__ import annotations
import cv2
import numpy as np
from scipy import stats
from skimage.feature import hog, local_binary_pattern
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix
    from skimage.feature import greycoprops as graycoprops

def _safe(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0

def _hist(values: np.ndarray, bins: int, prefix: str) -> dict[str, float]:
    hist, _ = np.histogram(values, bins=bins, range=(0, 255))
    hist = hist.astype(np.float64)
    hist = hist / max(hist.sum(), 1.0)
    return {f"{prefix}_hist_{i:03d}": _safe(v) for i, v in enumerate(hist)}

def _basic(rgb: np.ndarray, prefix: str) -> dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    features: dict[str, float] = {}
    for i, name in enumerate(["r", "g", "b"]):
        features[f"{prefix}_{name}_mean"] = _safe(np.mean(rgb[:, :, i]))
        features[f"{prefix}_{name}_std"] = _safe(np.std(rgb[:, :, i]))
    features[f"{prefix}_gray_mean"] = _safe(np.mean(gray))
    features[f"{prefix}_gray_std"] = _safe(np.std(gray))
    features[f"{prefix}_gray_skewness"] = _safe(stats.skew(gray.reshape(-1)))
    features[f"{prefix}_gray_kurtosis"] = _safe(stats.kurtosis(gray.reshape(-1)))
    return features

def _lbp(gray: np.ndarray, prefix: str) -> dict[str, float]:
    features: dict[str, float] = {}
    for radius, points in [(1, 8), (2, 16), (3, 24)]:
        lbp = local_binary_pattern(gray, points, radius, method="uniform")
        bins = points + 2
        hist, _ = np.histogram(lbp, bins=bins, range=(0, bins))
        hist = hist.astype(np.float64) / max(hist.sum(), 1.0)
        for i, v in enumerate(hist):
            features[f"{prefix}_lbp_r{radius}_p{points}_{i:02d}"] = _safe(v)
    return features

def _glcm(gray: np.ndarray, prefix: str) -> dict[str, float]:
    gray8 = np.clip(gray, 0, 255).astype(np.uint8)
    distances = [1, 2, 4]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    matrix = graycomatrix(gray8, distances, angles, levels=256, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    names = ["0", "45", "90", "135"]
    features: dict[str, float] = {}
    for prop in props:
        values = graycoprops(matrix, prop)
        for d_i, d in enumerate(distances):
            for a_i, a in enumerate(names):
                features[f"{prefix}_glcm_{prop}_d{d}_a{a}"] = _safe(values[d_i, a_i])
    return features

def _hog(gray: np.ndarray, prefix: str, max_dim: int = 512) -> dict[str, float]:
    vec = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    vec = vec[:max_dim]
    return {f"{prefix}_hog_{i:04d}": _safe(v) for i, v in enumerate(vec)}

def _gabor(gray: np.ndarray, prefix: str) -> dict[str, float]:
    features: dict[str, float] = {}
    for sigma in [2.0, 4.0, 8.0]:
        for angle in [0, 30, 60, 90, 120, 150]:
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=sigma,
                theta=np.deg2rad(angle),
                lambd=10.0,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F,
            )
            resp = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
            base = f"{prefix}_gabor_s{int(sigma)}_a{angle}"
            features[f"{base}_mean"] = _safe(np.mean(resp))
            features[f"{base}_std"] = _safe(np.std(resp))
            features[f"{base}_energy"] = _safe(np.mean(resp**2))
    return features

def _color(rgb: np.ndarray, prefix: str) -> dict[str, float]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    features: dict[str, float] = {}
    for i, name in enumerate(["r", "g", "b"]):
        features.update(_hist(rgb[:, :, i], 32, f"{prefix}_{name}"))
    for i, name in enumerate(["h", "s", "v"]):
        features[f"{prefix}_hsv_{name}_mean"] = _safe(np.mean(hsv[:, :, i]))
        features[f"{prefix}_hsv_{name}_std"] = _safe(np.std(hsv[:, :, i]))
        features.update(_hist(hsv[:, :, i], 32, f"{prefix}_hsv_{name}"))
    features.update(_hist(gray, 32, f"{prefix}_gray"))
    features[f"{prefix}_brightness"] = _safe(np.mean(gray))
    features[f"{prefix}_contrast"] = _safe(np.std(gray))
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    features[f"{prefix}_edge_density"] = _safe(np.mean(edges > 0))
    return features

def extract_gelsight_features(rgb: np.ndarray) -> dict[str, float]:
    rgb = np.asarray(rgb, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    features: dict[str, float] = {}
    features.update(_basic(rgb, "gelsight"))
    features.update(_hist(gray, 64, "gelsight_intensity"))
    features.update(_lbp(gray, "gelsight"))
    features.update(_glcm(gray, "gelsight"))
    features.update(_hog(gray, "gelsight"))
    features.update(_gabor(gray, "gelsight"))
    return features

def extract_visual_features(rgb: np.ndarray) -> dict[str, float]:
    rgb = np.asarray(rgb, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    features: dict[str, float] = {}
    features.update(_basic(rgb, "visual"))
    features.update(_color(rgb, "visual"))
    features.update(_lbp(gray, "visual"))
    features.update(_glcm(gray, "visual"))
    features.update(_hog(gray, "visual"))
    features.update(_gabor(gray, "visual"))
    return features
