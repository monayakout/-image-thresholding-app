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

# Cap the number of pixels used to find the centers to prevent freezing
MEAN_SHIFT_FIT_CAP = 4000


def estimate_bandwidth_numpy(
    pixels: np.ndarray,
    quantile: float = 0.2,
    n_anchor: int = 64,
    random_state: int = 42,
) -> float:
    """
    Estimates a good 'search radius' (bandwidth) for the algorithm.
    It does this by measuring distances between random points and finding a 
    distance that covers a certain percentage (quantile) of the neighbors.
    """
    num_pixels = pixels.shape[0]
    rng = np.random.RandomState(random_state)
    
    # Pick a small number of random "anchor" pixels to test
    num_anchors = min(n_anchor, num_pixels)
    anchor_indices = rng.choice(num_pixels, size=num_anchors, replace=False)
    
    all_distances = []
    
    # Measure the distance from each anchor to every other pixel
    for anchor_idx in anchor_indices[: min(32, num_anchors)]:
        distances = np.linalg.norm(pixels - pixels[anchor_idx], axis=1)
        all_distances.append(distances)
        
    # Flatten all distances into one giant list
    flat_distances = np.concatenate(all_distances) if all_distances else np.array([0.1])
    
    # Find the distance value at the requested quantile (e.g., 20% of the way up the list)
    estimated_bandwidth = float(np.quantile(flat_distances, quantile))
    
    # Return at least a tiny number to prevent divide-by-zero errors later
    return max(estimated_bandwidth, 1e-4)


def mean_shift_centers(
    pixels: np.ndarray,
    bandwidth: float,
    max_iter: int = 100,
    tol_ratio: float = 1e-3,
    max_seeds: int = 256,
    random_state: int = 42,
) -> np.ndarray:
    """
    The Core Mean Shift Algorithm.
    Imagine dropping a bunch of circles (seeds) on the image. Each circle looks at 
    the colors inside of it, calculates the average color, and moves itself to that 
    new average. It repeats this until it stops moving.
    """
    num_pixels = pixels.shape[0]
    rng = np.random.RandomState(random_state)
    
    # 1. INITIALIZATION: Pick starting points (seeds)
    if num_pixels <= max_seeds:
        seed_indices = np.arange(num_pixels)
    else:
        seed_indices = rng.choice(num_pixels, size=max_seeds, replace=False)

    converged_centers = []
    search_radius = max(float(bandwidth), 1e-6)

    # 2. SHIFTING: Move each seed toward the densest nearby color
    for seed_idx in seed_indices:
        current_center = pixels[int(seed_idx)].astype(np.float64).copy()
        
        for _ in range(max_iter):
            # Find all pixels within the search radius of our current center
            distances = np.linalg.norm(pixels - current_center, axis=1)
            neighbors_mask = distances <= search_radius
            
            # If no pixels are nearby, stop moving
            if not np.any(neighbors_mask):
                break
                
            # Calculate the average color of those nearby pixels
            new_center = pixels[neighbors_mask].mean(axis=0)
            
            # If the center barely moved, we have found the final peak (convergence)
            movement = np.linalg.norm(new_center - current_center)
            if movement <= tol_ratio * search_radius:
                current_center = new_center
                break
                
            current_center = new_center
            
        converged_centers.append(current_center.copy())

    if not converged_centers:
        return pixels.mean(axis=0, keepdims=True)

    # 3. MERGING: If multiple seeds climbed the same "hill" and ended up 
    # in the exact same spot, merge them into a single center.
    unique_peaks = [converged_centers[0]]
    merge_distance = 0.5 * search_radius
    
    for converged_center in converged_centers[1:]:
        # Find distance to the closest peak we've already saved
        distances_to_peaks = np.linalg.norm(np.asarray(unique_peaks) - converged_center, axis=1)
        closest_peak_distance = min(distances_to_peaks)
        
        # If it's far enough away from existing peaks, save it as a new distinct color peak
        if closest_peak_distance > merge_distance:
            unique_peaks.append(converged_center)

    return np.stack(unique_peaks, axis=0)


def apply_mean_shift(
    features: np.ndarray,
    shape: tuple[int, int],
    is_grayscale: bool,
    bandwidth: float | None = None,
    quantile: float = 0.2,
    max_samples: int = 8000,
    random_state: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Handles the preprocessing, running the algorithm, and reconstructing the image."""
    
    # Step 1: Subsample the image. Mean shift is slow, so we only run it on a fraction of pixels.
    fit_cap = min(int(max_samples), MEAN_SHIFT_FIT_CAP)
    subsampled_pixels, _ = subsample_rows(features, fit_cap, random_state=random_state)

    # Step 2: Estimate bandwidth if the user didn't provide one
    if bandwidth is None or bandwidth <= 0:
        actual_bandwidth = estimate_bandwidth_numpy(
            subsampled_pixels, quantile=float(quantile), random_state=random_state
        )
    else:
        actual_bandwidth = float(bandwidth)

    # Step 3: Run Mean Shift to find the dominant color centers
    centers = mean_shift_centers(subsampled_pixels, actual_bandwidth, random_state=random_state)
    
    if centers.shape[0] == 0:
        centers = subsampled_pixels.mean(axis=0, keepdims=True)

    # Step 4: Map every pixel in the full image to the nearest discovered center
    labels = assign_nearest_center(features, centers, chunk=131072)
    
    # Refine centers and rebuild the final image array
    centers = cluster_centroids_from_labels(features, labels, centers.shape[0])
    segmented_image = reconstruction_from_centers(labels, centers, shape, is_grayscale)

    meta = {
        "bandwidth": float(actual_bandwidth),
        "discovered_clusters": int(centers.shape[0]),
        "quantile": float(quantile),
        "max_samples_requested": max_samples,
        "fit_sample_size": int(subsampled_pixels.shape[0]),
    }
    return segmented_image, meta


def run_mean_shift_segmentation(image_file, **params) -> Dict[str, Any]:
    """Entry point for the frontend controller."""
    rgb, _gray, features, is_grayscale = load_image_from_file(image_file)
    height, width = rgb.shape[0], rgb.shape[1]

    # Parse inputs from the frontend
    bandwidth = params.get("bandwidth")
    if bandwidth is not None and bandwidth != "":
        bandwidth = float(bandwidth)
    else:
        bandwidth = None

    quantile = float(params.get("quantile", 0.2))
    max_samples = int(params.get("max_samples", 8000))

    # Run the processing pipeline
    segmented_array, meta_info = apply_mean_shift(
        features,
        (height, width),
        is_grayscale,
        bandwidth=bandwidth,
        quantile=quantile,
        max_samples=max_samples,
    )

    # Return exactly what the frontend expects
    return {
        "method": f"Mean Shift (~{meta_info['discovered_clusters']} regions)",
        "description": (
            f"From-scratch flat-kernel mean shift (bandwidth ≈ {meta_info['bandwidth']:.4f}); "
            f"{meta_info['discovered_clusters']} modes on {meta_info['fit_sample_size']} samples."
        ),
        "parameters": {**meta_info, "is_grayscale": is_grayscale},
        "original_image": array_to_base64_png(rgb),
        "result_image": array_to_base64_png(segmented_array),
    }