"""
K-means image segmentation — implemented from scratch (NumPy + PIL only).
Simplified for readability and learning.
"""

from __future__ import annotations

import base64
import io
import random

import numpy as np
from PIL import Image


class KMeansSegmentation:
    """Simple K-Means on pixel features (1-D gray or 3-D RGB, normalized to [0, 1])."""

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
        """Loads image and converts pixels to a 2D array of floats between 0 and 1."""
        img = Image.open(image_file)
        if img.mode == "L":
            self.is_grayscale = True
            arr = np.asarray(img, dtype=np.float32).reshape(-1, 1) / 255.0
        else:
            self.is_grayscale = False
            arr = np.asarray(img.convert("RGB"), dtype=np.float32).reshape(-1, 3) / 255.0
            
        self.original_shape = (img.size[1], img.size[0])
        return arr

    def _initialize_centers(self, data: np.ndarray) -> np.ndarray:
        """Simple Initialization: Pick K random pixels from the image to act as starting centers."""
        rng = np.random.default_rng(self.random_seed)
        num_pixels = data.shape[0]
        random_indices = rng.choice(num_pixels, self.k, replace=False)
        return data[random_indices].copy()

    def _assign_clusters(self, data: np.ndarray) -> tuple[np.ndarray, float]:
        """Calculates the distance from every pixel to every center instantly."""
        # NumPy Broadcasting: Expands data and centers to calculate all distances at once
        squared_distances = np.sum((data[:, None, :] - self.centers[None, :, :]) ** 2, axis=2)
        
        # Find the index of the closest center for each pixel
        labels = np.argmin(squared_distances, axis=1).astype(np.int32)
        
        # Calculate the total error (sum of distances to closest centers)
        total_distance = float(np.sum(np.min(squared_distances, axis=1)))
        
        return labels, total_distance

    def _update_centers(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculates new centers by finding the average color of all pixels in each cluster."""
        _, dim = data.shape
        new_centers = np.zeros((self.k, dim), dtype=np.float64)
        
        for cluster_idx in range(self.k):
            # Grab only the pixels that belong to this specific cluster
            pixels_in_cluster = data[labels == cluster_idx]
            
            if len(pixels_in_cluster) > 0:
                new_centers[cluster_idx] = pixels_in_cluster.mean(axis=0)
            else:
                # If a cluster is empty, reset it with a random pixel
                new_centers[cluster_idx] = data[random.randrange(data.shape[0])]
                
        return new_centers

    def fit(self, image_file):
        """The main K-Means loop."""
        data = self._load_image(image_file).astype(np.float64)
        self.centers = self._initialize_centers(data)
        self.total_distances = []

        for _ in range(self.max_iterations):
            # Step 1: Assign pixels to the closest center
            labels, total_dist = self._assign_clusters(data)
            self.labels = labels
            self.total_distances.append(total_dist)
            
            # Step 2: Calculate new centers based on the assignments
            new_centers = self._update_centers(data, labels)
            
            # Step 3: Check if the centers have stopped moving
            movement = float(np.sum(np.linalg.norm(self.centers - new_centers, axis=1)))
            if movement < self.convergence_threshold:
                self.centers = new_centers
                break
                
            self.centers = new_centers

        self.labels, _ = self._assign_clusters(data)
        return self._build_output(data)

    def _build_output(self, data: np.ndarray):
        """Formats the final arrays back into standard image shapes."""
        h, w = self.original_shape
        
        # Reconstruct the image using only the K center colors
        flat_segmented = self.centers[self.labels]
        seg_u8 = (flat_segmented * 255.0).clip(0, 255).astype(np.uint8)
        
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
        """Generates bright, distinct colors for the visualization map."""
        if self.is_grayscale:
            out = np.zeros((h, w), dtype=np.uint8)
            for i in range(self.k):
                shade = int(255 * i / (self.k - 1)) if self.k > 1 else 128
                out[labels_2d == i] = shade
            return out
            
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(self.k):
            # Seed the random generator so cluster 1 is always the same color
            np.random.seed(i * 100) 
            color = np.random.randint(0, 255, size=3)
            out[labels_2d == i] = color
            
        return out


# --- API / Frontend Wrapper Functions Below ---

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