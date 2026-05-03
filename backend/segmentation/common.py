"""Shared helpers for segmentation: loading images, encoding, clustering utilities."""

from __future__ import annotations

import base64
import io
from typing import Tuple

import numpy as np
from PIL import Image


def load_image_from_file(image_file) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Load an uploaded image file.

    Returns:
        rgb_uint8: (H, W, 3) for display
        gray_uint8: (H, W)
        features: (N, C) float in [0, 1], C is 1 (gray) or 3 (RGB)
        is_grayscale: whether source was single-channel
    """
    img = Image.open(image_file)
    if img.mode == "L":
        gray = np.asarray(img, dtype=np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
        features = (gray.reshape(-1, 1).astype(np.float64)) / 255.0
        return rgb, gray, features, True

    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    gray = (
        0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    ).astype(np.uint8)
    features = (rgb.reshape(-1, 3).astype(np.float64)) / 255.0
    return rgb, gray, features, False


def array_to_base64_png(arr: np.ndarray) -> str:
    if arr.ndim == 2:
        pil_img = Image.fromarray(arr.astype(np.uint8), mode="L")
    else:
        pil_img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def reconstruction_from_centers(
    labels: np.ndarray,
    centers: np.ndarray,
    shape: Tuple[int, int],
    is_grayscale: bool,
) -> np.ndarray:
    """Map each pixel to its cluster center color (uint8 image)."""
    flat = centers[labels.astype(np.int64)]
    h, w = shape
    if is_grayscale:
        return (flat[:, 0] * 255.0).clip(0, 255).astype(np.uint8).reshape(h, w)
    return (flat * 255.0).clip(0, 255).astype(np.uint8).reshape(h, w, 3)


def subsample_rows(
    features: np.ndarray, max_samples: int, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a row subset of features and the chosen indices."""
    n = features.shape[0]
    if n <= max_samples:
        return features, np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(random_state)
    idx = rng.choice(n, size=max_samples, replace=False)
    return features[idx], idx


def assign_nearest_center(
    features: np.ndarray, centers: np.ndarray, chunk: int = 65536
) -> np.ndarray:
    """Assign each row to the nearest center (Euclidean), memory-friendly."""
    n = features.shape[0]
    k = centers.shape[0]
    labels = np.empty(n, dtype=np.int32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = features[start:end]
        dist = np.sum((block[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels[start:end] = np.argmin(dist, axis=1)
    return labels


def cluster_centroids_from_labels(
    features: np.ndarray, labels: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Mean feature vector per cluster; re-seed empty clusters from global mean."""
    dim = features.shape[1]
    centers = np.zeros((n_clusters, dim), dtype=np.float64)
    overall = features.mean(axis=0)
    for j in range(n_clusters):
        mask = labels == j
        if np.any(mask):
            centers[j] = features[mask].mean(axis=0)
        else:
            centers[j] = overall
    return centers
