"""Hierarchical agglomerative clustering from scratch (NumPy only)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .common import (
    array_to_base64_png,
    assign_nearest_center,
    cluster_centroids_from_labels,
    load_image_from_file,
    reconstruction_from_centers,
    subsample_rows,
)

# Full matrix merges are O(n³); cap keeps interactive use reasonable.
AGGLOMERATIVE_FIT_CAP = 1200


def _initial_distance_matrix(
    X: np.ndarray, linkage: str, sizes: np.ndarray
) -> np.ndarray:
    sum_x = np.sum(X.astype(np.float64) ** 2, axis=1)
    se = sum_x[:, None] + sum_x[None, :] - 2.0 * (X.astype(np.float64) @ X.astype(np.float64).T)
    se = np.maximum(se, 0.0)
    if linkage == "ward":
        ni = sizes[:, None].astype(np.float64)
        nj = sizes[None, :].astype(np.float64)
        D = (ni * nj / (ni + nj)) * se
    else:
        D = np.sqrt(se, out=np.empty_like(se))
    np.fill_diagonal(D, np.inf)
    return D.astype(np.float64)


def _merge_distance(
    linkage: str,
    D: np.ndarray,
    sizes: np.ndarray,
    i: int,
    j: int,
    k: int,
) -> float:
    ni, nj, nk = float(sizes[i]), float(sizes[j]), float(sizes[k])
    if linkage == "single":
        return float(min(D[i, k], D[j, k]))
    if linkage == "complete":
        return float(max(D[i, k], D[j, k]))
    if linkage == "average":
        return float((ni * D[i, k] + nj * D[j, k]) / (ni + nj))
    if linkage == "ward":
        return float(
            ((nk + ni) * D[i, k] + (nk + nj) * D[j, k] - nk * D[i, j]) / (ni + nj + nk)
        )
    raise ValueError(f"Unknown linkage: {linkage}")


def agglomerative_labels(
    X: np.ndarray, n_clusters: int, linkage: str
) -> np.ndarray:
    """
    Return cluster index 0..n_clusters-1 for each row of X.
    """
    n = X.shape[0]
    n_clusters = max(2, min(int(n_clusters), n))
    if n_clusters >= n:
        return np.arange(n, dtype=np.int32)

    sizes = np.ones(n, dtype=np.float64)
    clusters: List[List[int]] = [[i] for i in range(n)]
    D = _initial_distance_matrix(X, linkage, sizes)
    m = n

    while m > n_clusters:
        idx_flat = int(np.argmin(D))
        i, j = divmod(idx_flat, m)
        if i > j:
            i, j = j, i

        for k in range(m):
            if k == i or k == j:
                continue
            nd = _merge_distance(linkage, D, sizes, i, j, k)
            D[i, k] = D[k, i] = nd

        clusters[i].extend(clusters[j])
        sizes[i] = sizes[i] + sizes[j]

        D = np.delete(np.delete(D, j, axis=0), j, axis=1)
        sizes = np.delete(sizes, j)
        del clusters[j]
        m -= 1

    labels = np.empty(n, dtype=np.int32)
    for cid, members in enumerate(clusters):
        for idx in members:
            labels[idx] = cid
    return labels


def apply_agglomerative(
    features: np.ndarray,
    shape: tuple[int, int],
    is_grayscale: bool,
    n_clusters: int = 4,
    linkage: str = "ward",
    max_samples: int = 8000,
    random_state: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_clusters = max(2, int(n_clusters))
    fit_cap = min(int(max_samples), AGGLOMERATIVE_FIT_CAP)
    sub, _ = subsample_rows(features, fit_cap, random_state=random_state)

    if linkage == "ward" and sub.shape[1] == 1:
        pass

    sub_labels = agglomerative_labels(sub, n_clusters, linkage)
    centers = cluster_centroids_from_labels(sub, sub_labels, n_clusters)
    labels = assign_nearest_center(features, centers, chunk=131072)
    centers = cluster_centroids_from_labels(features, labels, n_clusters)
    seg = reconstruction_from_centers(labels, centers, shape, is_grayscale)
    meta = {
        "n_clusters": n_clusters,
        "linkage": linkage,
        "max_samples_requested": max_samples,
        "fit_sample_size": int(sub.shape[0]),
    }
    return seg, meta


def run_agglomerative_segmentation(image_file, **params) -> Dict[str, Any]:
    rgb, _gray, features, is_grayscale = load_image_from_file(image_file)
    h, w = rgb.shape[0], rgb.shape[1]

    n_clusters = int(params.get("n_clusters", 4))
    linkage = str(params.get("linkage", "ward"))
    max_samples = int(params.get("max_samples", 8000))

    seg, meta = apply_agglomerative(
        features,
        (h, w),
        is_grayscale,
        n_clusters=n_clusters,
        linkage=linkage,
        max_samples=max_samples,
    )

    return {
        "method": f"Agglomerative ({meta['linkage']}, k={meta['n_clusters']})",
        "description": (
            f"From-scratch hierarchical clustering on {meta['fit_sample_size']} sample pixels "
            f"(cap {AGGLOMERATIVE_FIT_CAP}); labels extended by nearest center."
        ),
        "parameters": {**meta, "is_grayscale": is_grayscale},
        "original_image": array_to_base64_png(rgb),
        "result_image": array_to_base64_png(seg),
    }
