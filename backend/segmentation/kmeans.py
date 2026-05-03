"""
K-means image segmentation — implemented from scratch (NumPy + PIL only).

Clusters pixels in unsupervised fashion using intensity (grayscale) or RGB
feature vectors. No scikit-learn or other ML libraries.
"""

from __future__ import annotations

import base64
import io
import random

import numpy as np
from PIL import Image


class KMeansSegmentation:
    """Lloyd's k-means on pixel features (1-D gray or 3-D RGB, normalized to [0, 1])."""

    def __init__(
        self,
        k: int = 3,
        max_iterations: int = 100,
        convergence_threshold: float = 5.0,
        random_seed: int = 42,
    ):
        self.k = k
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.random_seed = random_seed
        self.centers = None
        self.labels = None
        self.is_grayscale: bool | None = None
        self.original_shape: tuple[int, ...] | None = None
        self.total_distances: list[float] = []

    def _load_image(self, image_file):
        img = Image.open(image_file)
        if img.mode == "L":
            self.is_grayscale = True
            arr = np.asarray(img, dtype=np.float32).reshape(-1, 1) / 255.0
        else:
            self.is_grayscale = False
            arr = (
                np.asarray(img.convert("RGB"), dtype=np.float32).reshape(-1, 3) / 255.0
            )
        self.original_shape = (img.size[1], img.size[0])
        return arr

    def _kmeans_plus_plus_init(self, data: np.ndarray) -> np.ndarray:
        rng = random.Random(self.random_seed)
        n, _ = data.shape
        centers = [data[rng.randrange(n)].copy()]
        for _ in range(1, self.k):
            dist_sq = np.zeros(n, dtype=np.float64)
            cstack = np.stack(centers, axis=0)
            d = np.sum((data[:, None, :] - cstack[None, :, :]) ** 2, axis=2)
            dist_sq = np.min(d, axis=1)
            s = float(dist_sq.sum())
            if s <= 1e-20:
                centers.append(data[rng.randrange(n)].copy())
                continue
            probs = dist_sq / s
            cdf = np.cumsum(probs)
            r = rng.random()
            pick = int(np.searchsorted(cdf, r))
            centers.append(data[pick].copy())
        return np.stack(centers, axis=0)

    def _assign(self, data: np.ndarray) -> tuple[np.ndarray, float]:
        d2 = np.sum(
            (data[:, None, :] - self.centers[None, :, :]) ** 2,
            axis=2,
        )
        labels = np.argmin(d2, axis=1).astype(np.int32)
        total = float(np.sum(np.min(d2, axis=1)))
        return labels, total

    def _update_centers(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        _, dim = data.shape
        new_c = np.zeros((self.k, dim), dtype=np.float64)
        rng = random.Random(self.random_seed)
        for j in range(self.k):
            mask = labels == j
            if np.any(mask):
                new_c[j] = data[mask].mean(axis=0)
            else:
                new_c[j] = data[rng.randrange(data.shape[0])]
        return new_c.astype(np.float64)

    def _movement(self, old: np.ndarray, new: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(old - new, axis=1)))

    def fit(self, image_file):
        data = self._load_image(image_file).astype(np.float64)
        self.centers = self._kmeans_plus_plus_init(data.astype(np.float32))
        self.total_distances = []

        for _ in range(self.max_iterations):
            labels, td = self._assign(data)
            self.labels = labels
            self.total_distances.append(td)
            new_c = self._update_centers(data, labels)
            move = self._movement(self.centers, new_c)
            if move < self.convergence_threshold:
                self.centers = new_c
                break
            self.centers = new_c

        self.labels, _ = self._assign(data)
        return self._build_output(data)

    def _build_output(self, data: np.ndarray):
        h, w = self.original_shape
        flat = self.centers[self.labels]
        seg_u8 = (flat * 255.0).clip(0, 255).astype(np.uint8)
        if self.is_grayscale:
            segmented = seg_u8.reshape(h, w)
        else:
            segmented = seg_u8.reshape(h, w, 3)

        labels_2d = self.labels.reshape(h, w)
        cluster_viz = self._label_colors(labels_2d, h, w)

        return {
            "segmented_image": segmented,
            "cluster_visualization": cluster_viz,
            "labels": self.labels,
            "centers": self.centers,
            "iterations": len(self.total_distances),
            "final_total_distance": float(self.total_distances[-1]),
        }

    def _label_colors(self, labels_2d: np.ndarray, h: int, w: int) -> np.ndarray:
        if self.is_grayscale:
            out = np.zeros((h, w), dtype=np.uint8)
            for i in range(self.k):
                g = int(255 * i / (self.k - 1)) if self.k > 1 else 128
                out[labels_2d == i] = g
            return out
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(self.k):
            hue = (i * 137.508) % 360
            s = 0.7 + (i % 3) * 0.1
            v = 0.8 + (i % 2) * 0.1
            hh = hue / 360.0
            if s == 0.0:
                rgb = (v, v, v)
            else:
                hh = hh * 6.0
                i_h = int(hh)
                f = hh - i_h
                p = v * (1.0 - s)
                q = v * (1.0 - s * f)
                t = v * (1.0 - s * (1.0 - f))
                if i_h == 0:
                    rgb = (v, t, p)
                elif i_h == 1:
                    rgb = (q, v, p)
                elif i_h == 2:
                    rgb = (p, v, t)
                elif i_h == 3:
                    rgb = (p, q, v)
                elif i_h == 4:
                    rgb = (t, p, v)
                else:
                    rgb = (v, p, q)
            col = tuple(int(c * 255) for c in rgb)
            out[labels_2d == i] = col
        return out


def _rewind(file_obj) -> None:
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)


def _to_b64(arr: np.ndarray) -> str:
    if arr.ndim == 2:
        pil_img = Image.fromarray(arr.astype(np.uint8), mode="L")
    else:
        pil_img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def segment_with_kmeans_from_scratch(
    image_file,
    k: int = 3,
    max_iterations: int = 100,
    convergence_threshold: float = 5.0,
) -> dict:
    """Run from-scratch k-means; returns API-style dict with base64 images."""
    _rewind(image_file)
    km = KMeansSegmentation(
        k=k,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
    )
    results = km.fit(image_file)

    _rewind(image_file)
    orig = Image.open(image_file)
    if orig.mode != "RGB":
        orig = orig.convert("RGB")
    original_array = np.asarray(orig, dtype=np.uint8)

    space = "grayscale intensity" if km.is_grayscale else "RGB color"
    return {
        "method": f"K-Means from scratch (k={k}, {space})",
        "description": (
            f"Unsupervised k-means in {space} space; "
            f"{results['iterations']} iteration(s)."
        ),
        "parameters": {
            "k": k,
            "iterations": results["iterations"],
            "final_total_distance": results["final_total_distance"],
            "is_grayscale": km.is_grayscale,
            "feature_dim": 1 if km.is_grayscale else 3,
            "convergence_threshold": convergence_threshold,
        },
        "original_image": _to_b64(original_array),
        "segmented_image": _to_b64(results["segmented_image"]),
        "cluster_visualization": _to_b64(results["cluster_visualization"]),
        "n_clusters": k,
    }


def run_kmeans_segmentation(image_file, **params) -> dict:
    """Entry point for `segmentation_controller` (adds `result_image`)."""
    k = max(2, int(params.get("k", 3)))
    max_iterations = int(params.get("max_iterations", 100))
    convergence_threshold = float(params.get("convergence_threshold", 5.0))
    out = segment_with_kmeans_from_scratch(
        image_file,
        k=k,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
    )
    out["result_image"] = out["segmented_image"]
    return out
