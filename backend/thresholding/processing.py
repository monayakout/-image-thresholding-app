import numpy as np
from PIL import Image
import io
import base64


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_gray(image_file) -> np.ndarray:
    img = Image.open(image_file).convert('L')
    return np.array(img, dtype=np.uint8)


def _to_base64(array: np.ndarray) -> str:
    pil_img = Image.fromarray(array.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _histogram(gray: np.ndarray):
    counts = np.zeros(256, dtype=np.int64)
    for val in gray.flatten():
        counts[val] += 1
    return list(range(256)), counts.tolist()


def _compute_histogram(gray: np.ndarray) -> np.ndarray:
    hist = np.zeros(256, dtype=np.float64)
    for val in gray.flatten():
        hist[val] += 1
    return hist


# ── 1. Optimal Thresholding (Isodata / Iterative) ────────────────────────────

def optimal_threshold(gray: np.ndarray) -> dict:
    t = float(gray.mean())

    for _ in range(1000):
        low  = gray[gray <= t]
        high = gray[gray >  t]

        if len(low) == 0 or len(high) == 0:
            break

        new_t = (low.mean() + high.mean()) / 2.0

        if abs(new_t - t) < 0.5:
            t = new_t
            break
        t = new_t

    t = round(t)
    binary = np.where(gray > t, 255, 0).astype(np.uint8)

    return {
        'method': 'Optimal (Isodata)',
        'threshold_value': float(t),
        'description': f'Iterative isodata method. Splits image into two groups, averages their means. Converged at t = {t}',
        'result_image': _to_base64(binary),
    }


# ── 2. Otsu's Thresholding ───────────────────────────────────────────────────

def otsu_threshold(gray: np.ndarray) -> dict:
    hist  = _compute_histogram(gray)
    total = gray.size
    prob  = hist / total

    best_t   = 0
    best_var = 0.0

    w_bg       = 0.0
    sum_bg     = 0.0
    total_mean = float(np.sum(np.arange(256) * prob))

    for t in range(256):
        w_bg   += prob[t]
        sum_bg += t * prob[t]
        w_fg    = 1.0 - w_bg

        if w_bg == 0 or w_fg == 0:
            continue

        mean_bg = sum_bg / w_bg
        mean_fg = (total_mean - sum_bg) / w_fg

        var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2

        if var_between > best_var:
            best_var = var_between
            best_t   = t

    binary = np.where(gray > best_t, 255, 0).astype(np.uint8)

    return {
        'method': "Otsu's Method",
        'threshold_value': float(best_t),
        'description': f'Maximises inter-class variance between background and foreground. Best t = {best_t}',
        'result_image': _to_base64(binary),
    }


# ── 3. Spectral (Multi-level Otsu) ───────────────────────────────────────────

def spectral_threshold(gray: np.ndarray, classes: int = 3) -> dict:
    classes = max(classes, 3)
    hist    = _compute_histogram(gray)
    total   = gray.size
    prob    = hist / total

    P  = np.cumsum(prob)
    PS = np.cumsum(np.arange(256) * prob)

    def class_variance(i_start, i_end):
        w = P[i_end] - (P[i_start - 1] if i_start > 0 else 0)
        if w < 1e-10:
            return 0.0
        s    = PS[i_end] - (PS[i_start - 1] if i_start > 0 else 0)
        mean = s / w
        return w * mean * mean

    best_var        = -1.0
    best_thresholds = [85, 170]

    if classes == 3:
        for t1 in range(1, 254):
            for t2 in range(t1 + 1, 255):
                v = (class_variance(0, t1) +
                     class_variance(t1 + 1, t2) +
                     class_variance(t2 + 1, 255))
                if v > best_var:
                    best_var        = v
                    best_thresholds = [t1, t2]
    else:
        best_thresholds = [
            int(255 * i / classes)
            for i in range(1, classes)
        ]

    levels         = np.linspace(0, 255, classes).astype(np.uint8)
    thresholds_arr = np.array(best_thresholds)
    regions        = np.digitize(gray, bins=thresholds_arr)
    segmented      = levels[regions]

    return {
        'method': f'Spectral / Multi-Otsu ({classes} classes)',
        'threshold_values': [float(t) for t in best_thresholds],
        'threshold_value':  float(best_thresholds[0]),
        'description': (
            f'Multi-level Otsu with {classes} classes. '
            f'Thresholds: {", ".join(str(t) for t in best_thresholds)}'
        ),
        'result_image': _to_base64(segmented),
    }


# ── 4. Local (Adaptive) Thresholding ─────────────────────────────────────────

def local_threshold(gray: np.ndarray, block_size: int = 35, offset: float = 10) -> dict:
    if block_size % 2 == 0:
        block_size += 1

    half  = block_size // 2
    h, w  = gray.shape
    gray_f = gray.astype(np.float64)

    # Pad image with reflection
    padded = np.pad(gray_f, half, mode='reflect')

    # Build integral image then add zero row/col for clean indexing
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode='constant', constant_values=0)

    r1 = np.arange(h)
    r2 = r1 + block_size
    c1 = np.arange(w)
    c2 = c1 + block_size

    R1 = r1[:, None]
    R2 = r2[:, None]
    C1 = c1[None, :]
    C2 = c2[None, :]

    box_sum      = integral[R2, C2] - integral[R1, C2] - integral[R2, C1] + integral[R1, C1]
    local_mean   = box_sum / (block_size * block_size)
    local_thresh = local_mean - offset

    binary = np.where(gray_f > local_thresh, 255, 0).astype(np.uint8)

    return {
        'method': 'Local (Adaptive)',
        'threshold_value': None,
        'description': (
            f'Adaptive threshold: each pixel compared to mean of its '
            f'{block_size}×{block_size} neighbourhood minus offset={offset:.0f}. '
            f'Handles uneven lighting.'
        ),
        'result_image': _to_base64(binary),
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run_all_thresholds(image_file, spectral_classes=3,
                       local_block=35, local_offset=10) -> dict:
    gray = _load_gray(image_file)
    values, counts = _histogram(gray)

    results = [
        optimal_threshold(gray),
        otsu_threshold(gray),
        spectral_threshold(gray, classes=spectral_classes),
        local_threshold(gray, block_size=local_block, offset=local_offset),
    ]

    return {
        'width':          gray.shape[1],
        'height':         gray.shape[0],
        'original_image': _to_base64(gray),
        'histogram':      {'values': values, 'counts': counts},
        'results':        results,
    }