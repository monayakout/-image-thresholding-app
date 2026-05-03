"""Dispatch segmentation requests to algorithm modules and normalize parameters."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .agglomerative import run_agglomerative_segmentation
from .kmeans import run_kmeans_segmentation
from .mean_shift import run_mean_shift_segmentation
from .region_growing import run_region_growing_segmentation

METHOD_SLUGS: Dict[str, str] = {
    "kmeans": "kmeans",
    "region-growing": "region_growing",
    "region_growing": "region_growing",
    "agglomerative": "agglomerative",
    "mean-shift": "mean_shift",
    "mean_shift": "mean_shift",
}

RUNNERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "kmeans": run_kmeans_segmentation,
    "region_growing": run_region_growing_segmentation,
    "agglomerative": run_agglomerative_segmentation,
    "mean_shift": run_mean_shift_segmentation,
}


def normalize_method_slug(slug: str) -> str:
    key = (slug or "kmeans").strip().lower().replace(" ", "-")
    if key in METHOD_SLUGS:
        return METHOD_SLUGS[key]
    raise ValueError(f"Unknown segmentation method: {slug!r}")


def parse_params(method_key: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract algorithm parameters from POST data (QueryDict or dict-like)."""

    def g(name, default=None, cast=None):
        if name not in data or data.get(name) in (None, ""):
            return default
        val = data.get(name)
        if cast is not None:
            try:
                return cast(val)
            except (TypeError, ValueError):
                return default
        return val

    if method_key == "kmeans":
        return {
            "k": g("k", 3, int),
            "max_iterations": g("max_iterations", 100, int),
            "convergence_threshold": g("convergence_threshold", 5.0, float),
        }

    if method_key == "region_growing":
        return {
            "seed_x": g("seed_x", None, int),
            "seed_y": g("seed_y", None, int),
            "threshold": g("threshold", 15.0, float),
            "connectivity": g("connectivity", 8, int),
        }

    if method_key == "agglomerative":
        return {
            "n_clusters": g("n_clusters", 4, int),
            "linkage": g("linkage", "ward", str) or "ward",
            "max_samples": g("max_samples", 8000, int),
        }

    if method_key == "mean_shift":
        bw = g("bandwidth", None, float)
        return {
            "bandwidth": bw,
            "quantile": g("quantile", 0.2, float),
            "max_samples": g("max_samples", 8000, int),
        }

    raise ValueError(f"No parameter parser for {method_key}")


def run_segmentation(method_slug: str, image_file, data: Dict[str, Any]) -> Dict[str, Any]:
    method_key = normalize_method_slug(method_slug)
    params = parse_params(method_key, data)
    runner = RUNNERS[method_key]
    return runner(image_file, **params)
