import cv2
import numpy as np

from helpers import save  # pyright: ignore


def find_bubbles(img, index, n_cols=5, n_rows=20):
    """
    Detecteaza mai intai chenarul grilei de bule, apoi il imparte uniform in n_cols x n_rows.
    Pastreaza aceeasi interfata si aceleasi nume de output ca bubbles.py.
    """
    if img is None or img.size == 0:
        return []

    gray = to_gray(img)
    mask = build_grid_mask(gray)
    blob_mask = build_grid_blob_mask(mask)
    save(f"debug_gray_{index}.png", blob_mask)

    grid_box = detect_grid_box(blob_mask)
    if grid_box is None:
        save(f"bubbles{index}.png", draw_overlay(img, None, [], [], 10))
        return []

    cols, rows = split_grid_uniform(grid_box, n_cols=n_cols, n_rows=n_rows)
    radius = estimate_radius(cols, rows)
    bubbles = build_bubble_matrix(cols, rows, radius)

    vis = draw_overlay(img, grid_box, cols, rows, radius)
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
    Produce o masca in care inelele devin zone albe compacte.
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

    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_small, iterations=1)
    return mask


def build_grid_blob_mask(mask):
    """
    Construieste un blob mare pentru intreaga grila, nu pentru fiecare elipsa separat.
    """
    h, w = mask.shape
    if h == 0 or w == 0:
        return np.zeros_like(mask)

    density = cv2.GaussianBlur(
        mask.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=max(6.0, w * 0.035),
        sigmaY=max(6.0, h * 0.014),
    )

    maxv = float(density.max())
    if maxv <= 0:
        return np.zeros_like(mask)

    density /= maxv
    blob = (density >= 0.20).astype(np.uint8) * 255

    kx = max(9, (w // 16) | 1)
    ky = max(9, (h // 40) | 1)
    blob = cv2.morphologyEx(
        blob,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3)),
        iterations=1,
    )
    blob = cv2.morphologyEx(
        blob,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, ky)),
        iterations=1,
    )
    blob = cv2.morphologyEx(
        blob,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    return blob


def detect_grid_box(blob_mask):
    h, w = blob_mask.shape
    if h == 0 or w == 0:
        return None

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blob_mask, connectivity=8)
    if n_labels <= 1:
        return None

    img_area = float(h * w)
    best_score = None
    best_box = None

    for label in range(1, n_labels):
        x, y, bw, bh, area = stats[label]
        if bw <= 0 or bh <= 0:
            continue
        if area < img_area * 0.08:
            continue

        aspect = float(bw) / float(bh)
        fill = float(area) / float(bw * bh)
        if aspect < 0.10 or aspect > 0.70:
            continue
        if fill < 0.18:
            continue

        score = float(area) * (0.6 + fill)
        if best_score is None or score > best_score:
            best_score = score
            best_box = (x, y, x + bw - 1, y + bh - 1)

    if best_box is None:
        return None

    x0, y0, x1, y1 = best_box
    pad_x = max(4, int((x1 - x0) * 0.02))
    pad_y = max(2, int((y1 - y0) * 0.01))

    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)
    return (x0, y0, x1, y1)


def detect_grid_box_legacy(mask):
    h, w = mask.shape
    if h == 0 or w == 0:
        return None

    x0, x1 = detect_x_bounds(mask)
    if x0 is None or x1 is None or x1 <= x0:
        return None

    y0, y1 = detect_y_bounds(mask, x0, x1)
    if y0 is None or y1 is None or y1 <= y0:
        return None

    pad_x = max(4, int((x1 - x0) * 0.03))
    pad_y = max(2, int((y1 - y0) * 0.01))

    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)
    return (x0, y0, x1, y1)


def detect_x_bounds(mask):
    h, w = mask.shape
    y0 = int(0.08 * h)
    y1 = int(0.92 * h)
    roi = mask[y0:y1, :]
    if roi.size == 0:
        roi = mask

    proj = roi.sum(axis=0).astype(np.float32)
    proj = smooth_projection(proj, max(9, (w // 18) | 1))
    return dominant_active_span(proj, rel_thr=0.18, min_len=max(20, w // 6))


def detect_y_bounds(mask, x0, x1):
    h, w = mask.shape
    span_w = max(1, x1 - x0 + 1)
    # Pe Y ne uitam doar in zona centrala a grilei ca sa ignoram literele si marginile.
    xi0 = max(0, int(round(x0 + 0.28 * span_w)))
    xi1 = min(w, int(round(x1 - 0.28 * span_w)) + 1)
    if xi1 <= xi0:
        xi0 = max(0, x0)
        xi1 = min(w, x1 + 1)
    y_pad0 = int(0.05 * h)
    y_pad1 = int(0.95 * h)
    roi = mask[y_pad0:y_pad1, xi0:xi1]
    if roi.size == 0:
        return None, None

    proj = roi.sum(axis=1).astype(np.float32)
    proj = smooth_projection(proj, max(9, (h // 35) | 1))
    y0, y1 = dominant_active_span(proj, rel_thr=0.30, min_len=max(40, int(h * 0.32)))
    if y0 is None or y1 is None:
        return None, None
    return y0 + y_pad0, y1 + y_pad0


def smooth_projection(arr, kernel):
    if arr.size == 0:
        return arr

    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(arr.reshape(-1, 1), (1, k), 0).ravel()


def dominant_active_span(proj, rel_thr=0.2, min_len=10):
    if proj.size == 0:
        return None, None

    maxv = float(proj.max())
    if maxv <= 0:
        return None, None

    thr = maxv * rel_thr
    active = proj >= thr
    spans = mask_to_spans(active)
    if not spans:
        return None, None

    best_score = None
    best = None
    for start, end in spans:
        length = end - start + 1
        if length < min_len:
            continue

        strength = float(proj[start : end + 1].sum())
        score = strength + (0.35 * length * maxv)
        if best_score is None or score > best_score:
            best_score = score
            best = (start, end)

    if best is not None:
        return best

    # fallback: daca nu exista span suficient de lung, luam cel mai bun span oricum
    best = max(
        spans,
        key=lambda s: float(proj[s[0] : s[1] + 1].sum()) + (0.2 * (s[1] - s[0] + 1) * maxv),
    )
    return best


def mask_to_spans(mask_1d):
    spans = []
    start = None
    for i, value in enumerate(mask_1d):
        if value and start is None:
            start = i
        elif not value and start is not None:
            spans.append((start, i - 1))
            start = None

    if start is not None:
        spans.append((start, len(mask_1d) - 1))
    return spans


def split_grid_uniform(grid_box, n_cols=5, n_rows=20):
    x0, y0, x1, y1 = grid_box

    cols = centers_from_box(x0, x1, n_cols)
    rows = centers_from_box(y0, y1, n_rows)
    return cols, rows


def centers_from_box(start, end, count):
    if count <= 0:
        return []

    size = max(1.0, float(end - start + 1))
    step = size / float(count)
    centers = []
    for i in range(count):
        centers.append(float(start + (i + 0.5) * step))
    return centers


def estimate_radius(cols, rows):
    if len(cols) >= 2:
        col_step = float(np.median(np.diff(cols)))
    else:
        col_step = 20.0

    if len(rows) >= 2:
        row_step = float(np.median(np.diff(rows)))
    else:
        row_step = 20.0

    return float(min(col_step, row_step) * 0.32)


def build_bubble_matrix(cols, rows, radius):
    out = []
    for cy in rows:
        for cx in cols:
            out.append({"cx": float(cx), "cy": float(cy), "r": float(radius)})
    return out


def draw_overlay(img, grid_box, cols, rows, radius):
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
