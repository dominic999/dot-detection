import cv2
import numpy as np

from helpers import save  # pyright: ignore


def find_bubbles(img, index, n_cols=5, n_rows=20):
    """
    Detecteaza matricea de bule prin:
    1. constructia unei masti care scoate in evidenta inelele
    2. gasirea centrelor de coloana/rand din proiectii 1D
    3. regularizarea grilei la dimensiunea fixa n_cols x n_rows

    Pastreaza aceeasi interfata si aceleasi nume de output ca bubbles.py.
    """
    if img is None or img.size == 0:
        return []

    gray = to_gray(img)
    mask = build_grid_mask(gray)
    save(f"debug_gray_{index}.png", mask)

    cols = detect_grid_columns(mask, expected=n_cols)
    rows = detect_grid_rows(mask, cols, expected=n_rows)
    if not cols or not rows:
        save(f"bubbles{index}.png", draw_grid_overlay(img, [], [], None))
        return []

    cols = regularize_centers(cols, n_cols)
    rows = regularize_centers(rows, n_rows)
    grid_box = make_grid_box(cols, rows, gray.shape)
    radius = estimate_bubble_radius(cols, rows)

    bubbles = build_bubble_matrix(cols, rows, radius)
    vis = draw_grid_overlay(img, cols, rows, grid_box, radius=radius)
    save(f"bubbles{index}.png", vis)

    out = []
    for bubble in bubbles:
        out.append(
            {
                "cx": int(round(bubble["cx"])),
                "cy": int(round(bubble["cy"])),
                "r": int(round(bubble["r"])),
            }
        )

    return out


def to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def build_grid_mask(gray):
    """
    Scoate in evidenta contururile elipselor.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(blur)

    mask = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        8,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def detect_grid_columns(mask, expected=5):
    h, w = mask.shape
    y0 = int(0.08 * h)
    y1 = int(0.50 * h)
    roi = mask[y0:y1, :]
    if roi.size == 0:
        roi = mask

    proj = roi.sum(axis=0).astype(np.float32)
    proj = smooth_projection(proj, max(9, (w // 18) | 1))
    peaks = find_local_maxima(proj, rel_thr=0.42, min_distance=max(10, w // 14))
    if len(peaks) < expected:
        peaks = find_local_maxima(proj, rel_thr=0.28, min_distance=max(8, w // 18))

    return select_regular_peaks(peaks, proj, expected)


def detect_grid_rows(mask, cols, expected=20):
    if not cols:
        return []

    h, w = mask.shape
    col_spacing = median_spacing(cols)
    if col_spacing <= 0:
        col_spacing = max(10.0, w / 7.0)

    x0 = int(max(0, min(cols) - 0.55 * col_spacing))
    x1 = int(min(w, max(cols) + 0.55 * col_spacing))
    roi = mask[:, x0:x1]
    if roi.size == 0:
        roi = mask

    proj = roi.sum(axis=1).astype(np.float32)
    proj = smooth_projection(proj, max(9, (h // 35) | 1))
    peaks = find_local_maxima(proj, rel_thr=0.28, min_distance=max(10, h // 34))
    if len(peaks) < expected:
        peaks = find_local_maxima(proj, rel_thr=0.18, min_distance=max(8, h // 40))

    rows = select_regular_peaks(peaks, proj, expected)
    if len(rows) >= 2:
        return rows

    return fallback_uniform_rows(mask.shape[0], expected)


def smooth_projection(arr, kernel):
    if arr.size == 0:
        return arr

    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(arr.reshape(-1, 1), (1, k), 0).ravel()


def find_local_maxima(arr, rel_thr=0.3, min_distance=10):
    if arr.size < 3:
        return []

    maxv = float(arr.max())
    if maxv <= 0:
        return []

    thr = maxv * rel_thr
    candidates = []
    for i in range(1, len(arr) - 1):
        if arr[i] < thr:
            continue
        if arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
            candidates.append(i)

    # Non-maximum suppression pe varfuri apropiate.
    ordered = sorted(candidates, key=lambda i: float(arr[i]), reverse=True)
    kept = []
    for idx in ordered:
        if all(abs(idx - prev) >= min_distance for prev in kept):
            kept.append(idx)

    kept.sort()
    return kept


def select_regular_peaks(peaks, projection, expected):
    if len(peaks) < expected:
        return []
    if len(peaks) == expected:
        return [float(p) for p in peaks]

    best_score = None
    best = []
    for start in range(0, len(peaks) - expected + 1):
        window = peaks[start : start + expected]
        spacings = np.diff(window).astype(np.float32)
        if np.any(spacings <= 0):
            continue

        spacing_med = float(np.median(spacings))
        spacing_err = float(np.mean(np.abs(spacings - spacing_med)) / (spacing_med + 1e-6))
        strength = float(sum(float(projection[p]) for p in window))
        score = strength - (strength * 0.35 * spacing_err)

        if best_score is None or score > best_score:
            best_score = score
            best = [float(p) for p in window]

    return best


def regularize_centers(centers, expected):
    if not centers:
        return []
    if len(centers) == 1:
        return [float(centers[0])] * expected

    start = float(centers[0])
    end = float(centers[-1])
    if expected == 1:
        return [0.5 * (start + end)]
    return [float(v) for v in np.linspace(start, end, expected)]


def median_spacing(values):
    if len(values) < 2:
        return 0.0
    return float(np.median(np.diff(values)))


def make_grid_box(cols, rows, shape):
    h, w = shape[:2]
    col_spacing = median_spacing(cols)
    row_spacing = median_spacing(rows)

    if col_spacing <= 0:
        col_spacing = max(10.0, w / 7.0)
    if row_spacing <= 0:
        row_spacing = max(10.0, h / 24.0)

    x0 = int(max(0, round(cols[0] - 0.55 * col_spacing)))
    x1 = int(min(w - 1, round(cols[-1] + 0.55 * col_spacing)))
    y0 = int(max(0, round(rows[0] - 0.55 * row_spacing)))
    y1 = int(min(h - 1, round(rows[-1] + 0.55 * row_spacing)))
    return (x0, y0, x1, y1)


def estimate_bubble_radius(cols, rows):
    col_spacing = median_spacing(cols)
    row_spacing = median_spacing(rows)
    refs = [v for v in (col_spacing, row_spacing) if v > 0]
    if not refs:
        return 10.0
    return float(min(refs) * 0.34)


def build_bubble_matrix(cols, rows, radius):
    out = []
    for cy in rows:
        for cx in cols:
            out.append({"cx": float(cx), "cy": float(cy), "r": float(radius)})
    return out


def fallback_uniform_rows(height, expected):
    if expected <= 0:
        return []
    y0 = 0.08 * height
    y1 = 0.92 * height
    return [float(v) for v in np.linspace(y0, y1, expected)]


def draw_grid_overlay(img, cols, rows, grid_box, radius=10):
    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    if grid_box is not None:
        x0, y0, x1, y1 = grid_box
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 2)

    for x in cols:
        cv2.line(
            vis,
            (int(round(x)), 0),
            (int(round(x)), vis.shape[0] - 1),
            (255, 255, 0),
            1,
        )

    for y in rows:
        cv2.line(
            vis,
            (0, int(round(y))),
            (vis.shape[1] - 1, int(round(y))),
            (255, 255, 0),
            1,
        )

    for y in rows:
        for x in cols:
            cx = int(round(x))
            cy = int(round(y))
            cv2.circle(vis, (cx, cy), int(round(radius)), (0, 255, 0), 2)
            cv2.circle(vis, (cx, cy), 2, (0, 0, 255), 2)

    return vis
