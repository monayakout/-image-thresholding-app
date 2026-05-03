"""Mean shift mode seeking from scratch (NumPy only)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .common import (
    array_to_base64_png,
    assign_nearest_center,
    cluster_centroids_from_labels,
    load_image_from_file,
    reconstruction_from_centers,
    subsample_rows,
)

MEAN_SHIFT_FIT_CAP = 4000


def estimate_bandwidth_numpy(
    X: np.ndarray,
    quantile: float = 0.2,
    n_anchor: int = 64,
    random_state: int = 42,
) -> float:
    """Rough bandwidth: quantile of distances from random anchors to all points."""
    n = X.shape[0]
    rng = np.random.RandomState(random_state)
    na = min(n_anchor, n)
    anchors = rng.choice(n, size=na, replace=False)
    dists = []
    for a in anchors[: min(32, na)]:
        d = np.linalg.norm(X - X[a], axis=1)
        dists.append(d)
    flat = np.concatenate(dists) if dists else np.array([0.1])
    bw = float(np.quantile(flat, quantile))
    return max(bw, 1e-4)


def mean_shift_centers(
    X: np.ndarray,
    bandwidth: float,
    max_iter: int = 100,
    tol_ratio: float = 1e-3,
    max_seeds: int = 256,
    random_state: int = 42,
) -> np.ndarray:
    """
    Flat kernel mean shift: shift each seed to mean of points within bandwidth.
    Merge modes closer than bandwidth/2.
    """
    n = X.shape[0]
    rng = np.random.RandomState(random_state)
    if n <= max_seeds:
        seed_idx = np.arange(n)
    else:
        seed_idx = rng.choice(n, size=max_seeds, replace=False)

    converged: list[np.ndarray] = []
    bw = max(float(bandwidth), 1e-6)

    for si in seed_idx:
        x = X[int(si)].astype(np.float64).copy()
        for _ in range(max_iter):
            diff = X - x
            dist = np.linalg.norm(diff, axis=1)
            mask = dist <= bw
            if not np.any(mask):
                break
            x_new = X[mask].mean(axis=0)
            if np.linalg.norm(x_new - x) <= tol_ratio * bw:
                x = x_new
                break
            x = x_new
        converged.append(x.copy())

    if not converged:
        return X.mean(axis=0, keepdims=True)

    peaks = [converged[0]]
    merge_dist = 0.5 * bw
    for c in converged[1:]:
        dmin = min(np.linalg.norm(np.asarray(peaks) - c, axis=1))
        if dmin > merge_dist:
            peaks.append(c)

    return np.stack(peaks, axis=0)


def apply_mean_shift(
    features: np.ndarray,
    shape: tuple[int, int],
    is_grayscale: bool,
    bandwidth: float | None = None,
    quantile: float = 0.2,
    max_samples: int = 8000,
    random_state: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    fit_cap = min(int(max_samples), MEAN_SHIFT_FIT_CAP)
    sub, _ = subsample_rows(features, fit_cap, random_state=random_state)

    if bandwidth is None or bandwidth <= 0:
        bw = estimate_bandwidth_numpy(
            sub, quantile=float(quantile), random_state=random_state
        )
    else:
        bw = float(bandwidth)

    centers = mean_shift_centers(sub, bw, random_state=random_state)
    if centers.shape[0] == 0:
        centers = sub.mean(axis=0, keepdims=True)

    labels = assign_nearest_center(features, centers, chunk=131072)
    centers = cluster_centroids_from_labels(features, labels, centers.shape[0])
    seg = reconstruction_from_centers(labels, centers, shape, is_grayscale)

    meta = {
        "bandwidth": float(bw),
        "discovered_clusters": int(centers.shape[0]),
        "quantile": float(quantile),
        "max_samples_requested": max_samples,
        "fit_sample_size": int(sub.shape[0]),
    }
    return seg, meta


def run_mean_shift_segmentation(image_file, **params) -> Dict[str, Any]:
    rgb, _gray, features, is_grayscale = load_image_from_file(image_file)
    h, w = rgb.shape[0], rgb.shape[1]

    bandwidth = params.get("bandwidth")
    if bandwidth is not None and bandwidth != "":
        bandwidth = float(bandwidth)
    else:
        bandwidth = None

    quantile = float(params.get("quantile", 0.2))
    max_samples = int(params.get("max_samples", 8000))

    seg, meta = apply_mean_shift(
        features,
        (h, w),
        is_grayscale,
        bandwidth=bandwidth,
        quantile=quantile,
        max_samples=max_samples,
    )

    return {
        "method": f"Mean Shift (~{meta['discovered_clusters']} regions)",
        "description": (
            f"From-scratch flat-kernel mean shift (bandwidth ≈ {meta['bandwidth']:.4f}); "
            f"{meta['discovered_clusters']} modes on {meta['fit_sample_size']} samples."
        ),
        "parameters": {**meta, "is_grayscale": is_grayscale},
        "original_image": array_to_base64_png(rgb),
        "result_image": array_to_base64_png(seg),
    }
