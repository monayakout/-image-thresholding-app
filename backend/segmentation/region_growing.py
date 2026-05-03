"""Region growing from a seed pixel — grayscale (intensity) or color (RGB distance)."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Tuple

import numpy as np

from .common import array_to_base64_png, load_image_from_file


def apply_region_growing_grayscale(
    gray: np.ndarray,
    seed_y: int,
    seed_x: int,
    threshold: float,
    connectivity: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    h, w = gray.shape
    seed_y = int(np.clip(seed_y, 0, h - 1))
    seed_x = int(np.clip(seed_x, 0, w - 1))
    seed_val = float(gray[seed_y, seed_x])

    visited = np.zeros((h, w), dtype=bool)
    region = np.zeros((h, w), dtype=bool)
    q = deque([(seed_y, seed_x)])
    visited[seed_y, seed_x] = True

    neigh = (
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        if connectivity == 8
        else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )

    while q:
        y, x = q.popleft()
        if abs(float(gray[y, x]) - seed_val) <= threshold:
            region[y, x] = True
            for dy, dx in neigh:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))

    seg = np.zeros((h, w), dtype=np.uint8)
    seg[region] = 255
    meta = {
        "seed_y": seed_y,
        "seed_x": seed_x,
        "threshold": threshold,
        "region_pixels": int(region.sum()),
        "space": "grayscale",
    }
    return seg, meta


def apply_region_growing_color(
    rgb: np.ndarray,
    seed_y: int,
    seed_x: int,
    threshold: float,
    connectivity: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Region growing where similarity is Euclidean distance in RGB (0–255 per channel)."""
    h, w, _ = rgb.shape
    seed_y = int(np.clip(seed_y, 0, h - 1))
    seed_x = int(np.clip(seed_x, 0, w - 1))
    seed_color = rgb[seed_y, seed_x].astype(np.float64)

    visited = np.zeros((h, w), dtype=bool)
    region = np.zeros((h, w), dtype=bool)
    q = deque([(seed_y, seed_x)])
    visited[seed_y, seed_x] = True

    neigh = (
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        if connectivity == 8
        else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )

    thr2 = float(threshold) ** 2

    while q:
        y, x = q.popleft()
        pix = rgb[y, x].astype(np.float64)
        if np.sum((pix - seed_color) ** 2) <= thr2:
            region[y, x] = True
            for dy, dx in neigh:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))

    seg = np.zeros((h, w), dtype=np.uint8)
    seg[region] = 255
    meta = {
        "seed_y": seed_y,
        "seed_x": seed_x,
        "threshold": threshold,
        "region_pixels": int(region.sum()),
        "space": "RGB",
    }
    return seg, meta


def run_region_growing_segmentation(image_file, **params) -> Dict[str, Any]:
    rgb, gray, _features, is_grayscale = load_image_from_file(image_file)
    h, w = gray.shape

    seed_y = int(params.get("seed_y", h // 2))
    seed_x = int(params.get("seed_x", w // 2))
    threshold = float(params.get("threshold", 15.0))
    connectivity = int(params.get("connectivity", 8))

    if is_grayscale:
        seg, meta = apply_region_growing_grayscale(
            gray, seed_y, seed_x, threshold, connectivity=connectivity
        )
        desc = (
            f"Grow from seed ({meta['seed_x']}, {meta['seed_y']}) while "
            f"|I − I_seed| ≤ {threshold:.1f} (grayscale)."
        )
    else:
        seg, meta = apply_region_growing_color(
            rgb, seed_y, seed_x, threshold, connectivity=connectivity
        )
        desc = (
            f"Grow from seed ({meta['seed_x']}, {meta['seed_y']}) while "
            f"‖RGB − RGB_seed‖ ≤ {threshold:.1f} (Euclidean in 0–255 RGB)."
        )

    return {
        "method": "Region Growing",
        "description": desc,
        "parameters": {
            **meta,
            "connectivity": connectivity,
            "is_grayscale": is_grayscale,
        },
        "original_image": array_to_base64_png(rgb),
        "result_image": array_to_base64_png(seg),
    }
