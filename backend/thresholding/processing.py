import numpy as np
from PIL import Image
import io, base64
from skimage.filters import threshold_otsu, threshold_multiotsu, threshold_local, threshold_isodata


def _load_gray(image_file) -> np.ndarray:
    img = Image.open(image_file).convert('L')
    return np.array(img, dtype=np.uint8)


def _to_base64(array: np.ndarray) -> str:
    pil_img = Image.fromarray(array.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _histogram(array: np.ndarray):
    counts, _ = np.histogram(array.flatten(), bins=256, range=(0, 255))
    return list(range(256)), counts.tolist()


def optimal_threshold(gray: np.ndarray) -> dict:
    t = threshold_isodata(gray)
    binary = (gray > t).astype(np.uint8) * 255
    return {
        'method': 'Optimal (Isodata)',
        'threshold_value': float(round(t, 2)),
        'description': f'Iterative isodata method. Converged at t = {t:.1f}',
        'result_image': _to_base64(binary),
    }


def otsu_threshold(gray: np.ndarray) -> dict:
    t = threshold_otsu(gray)
    binary = (gray > t).astype(np.uint8) * 255
    return {
        'method': "Otsu's Method",
        'threshold_value': float(round(t, 2)),
        'description': f'Maximises inter-class variance. Best for bimodal histograms. t = {t:.1f}',
        'result_image': _to_base64(binary),
    }


def spectral_threshold(gray: np.ndarray, classes: int = 3) -> dict:
    classes = max(classes, 3)
    thresholds = threshold_multiotsu(gray, classes=classes)
    regions = np.digitize(gray, bins=thresholds)
    levels = np.linspace(0, 255, classes).astype(np.uint8)
    segmented = levels[regions]
    return {
        'method': f'Spectral / Multi-Otsu ({classes} classes)',
        'threshold_values': [float(round(t, 2)) for t in thresholds],
        'threshold_value': float(round(thresholds[0], 2)),
        'description': (
            f'Multi-Otsu with {classes} classes, thresholds: '
            + ', '.join(f'{t:.1f}' for t in thresholds)
        ),
        'result_image': _to_base64(segmented),
    }


def local_threshold(gray: np.ndarray, block_size: int = 35, offset: float = 10) -> dict:
    if block_size % 2 == 0:
        block_size += 1
    local_thresh = threshold_local(gray, block_size=block_size, offset=offset)
    binary = (gray > local_thresh).astype(np.uint8) * 255
    return {
        'method': 'Local (Adaptive)',
        'threshold_value': None,
        'description': f'Adaptive threshold with {block_size}x{block_size} neighbourhood. Handles uneven lighting.',
        'result_image': _to_base64(binary),
    }


def run_all_thresholds(image_file, spectral_classes=3, local_block=35, local_offset=10) -> dict:
    gray = _load_gray(image_file)
    values, counts = _histogram(gray)
    results = [
        optimal_threshold(gray),
        otsu_threshold(gray),
        spectral_threshold(gray, classes=spectral_classes),
        local_threshold(gray, block_size=local_block, offset=local_offset),
    ]
    return {
        'width': gray.shape[1],
        'height': gray.shape[0],
        'original_image': _to_base64(gray),
        'histogram': {'values': values, 'counts': counts},
        'results': results,
    }