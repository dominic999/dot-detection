import math

import cv2
import numpy as np

from helpers import save  # pyright: ignore


def find_bubbles(img, index):
    """
    Detecteaza bulinele dintr-un box folosind:
    1. o masca mai permisiva pentru contururile slabe
    2. deduplicare spatiala
    3. selectie pe o grila regulata de 5 coloane

    Pastreaza aceeasi interfata si aceleasi nume de output ca bubbles.py.
    """
    if img is None or img.size == 0:
        return []

    gray = to_gray(img)
    mask = build_bubble_mask(gray)
    save(f"debug_gray_{index}.png", mask)

    candidates = extract_circle_candidates(mask)
    candidates = suppress_duplicate_candidates(candidates)

    columns = select_grid_columns(candidates, expected_cols=5)
    rows = select_grid_rows(candidates, columns)
    accepted = assign_candidates_to_grid(candidates, columns, rows)

    vis = draw_detected_bubbles(img, accepted)
    save(f"bubbles{index}.png", vis)

    out = []
    for bubble in accepted:
        out.append(
            {
                "cx": int(round(bubble["cx"])),
                "cy": int(round(bubble["cy"])),
                "r": int(round(bubble["r"])),
            }
        )

    out.sort(key=lambda b: (b["cy"], b["cx"]))
    return out


def to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def build_bubble_mask(gray):
    """
    Combinatie intre contrast local + adaptive threshold.
    E mai buna pentru inele slabe decat un prag fix simplu.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(blur)

    adaptive = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        7,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def extract_circle_candidates(
    mask,
    min_area=500,
    max_area=5000,
    min_radius=5,
    max_radius=40,
    min_circularity=0.35,
    max_aspect_ratio_diff=0.40,
):
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []

    for cnt in cnts:
        if len(cnt) < 12:
            continue

        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if r < min_radius or r > max_radius:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 0 or w <= 0:
            continue

        aspect_diff = abs(1.0 - (w / h))
        if aspect_diff > max_aspect_ratio_diff:
            continue

        circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        score = circularity - (0.35 * aspect_diff)

        out.append(
            {
                "cx": float(cx),
                "cy": float(cy),
                "r": float(r),
                "area": float(area),
                "bbox": (x, y, w, h),
                "circularity": float(circularity),
                "score": float(score),
            }
        )

    return out


def suppress_duplicate_candidates(candidates, dist_factor=0.65, radius_factor=0.45):
    """
    Cand aceeasi bulina produce mai multe contururi apropiate, pastram unul.
    """
    kept = []
    ordered = sorted(
        candidates,
        key=lambda c: (c["score"], c["circularity"], c["area"]),
        reverse=True,
    )

    for cand in ordered:
        duplicate = False
        for prev in kept:
            dist = math.hypot(cand["cx"] - prev["cx"], cand["cy"] - prev["cy"])
            min_r = max(1.0, min(cand["r"], prev["r"]))
            max_r = max(cand["r"], prev["r"])

            if dist < (min_r * dist_factor) and abs(cand["r"] - prev["r"]) < (
                max_r * radius_factor
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(cand)

    return kept


def cluster_axis(values, tol):
    if not values:
        return []

    ordered = sorted(float(v) for v in values)
    clusters = [[ordered[0]]]

    for value in ordered[1:]:
        if abs(value - np.mean(clusters[-1])) <= tol:
            clusters[-1].append(value)
        else:
            clusters.append([value])

    out = []
    for cluster in clusters:
        center = float(np.mean(cluster))
        out.append({"center": center, "count": len(cluster), "values": cluster})
    return out


def select_grid_columns(candidates, expected_cols=5):
    if len(candidates) < expected_cols:
        return []

    radii = [c["r"] for c in candidates]
    median_r = float(np.median(radii)) if radii else 10.0
    x_tol = max(6.0, median_r * 1.1)

    clusters = cluster_axis([c["cx"] for c in candidates], x_tol)
    if len(clusters) < expected_cols:
        return []

    best_score = None
    best_centers = []

    clusters = sorted(clusters, key=lambda c: c["center"])
    for start in range(0, len(clusters) - expected_cols + 1):
        window = clusters[start : start + expected_cols]
        centers = [c["center"] for c in window]
        counts = [c["count"] for c in window]
        spacings = np.diff(centers)
        if len(spacings) == 0:
            continue

        spacing_med = float(np.median(spacings))
        if spacing_med <= 0:
            continue

        spacing_error = float(np.mean(np.abs(spacings - spacing_med)) / spacing_med)
        count_score = float(sum(counts))
        score = count_score - (6.0 * spacing_error)

        if best_score is None or score > best_score:
            best_score = score
            best_centers = centers

    return best_centers


def select_grid_rows(candidates, columns):
    if not candidates or not columns:
        return []

    radii = [c["r"] for c in candidates]
    median_r = float(np.median(radii)) if radii else 10.0
    x_tol = max(8.0, median_r * 1.2)
    y_tol = max(8.0, median_r * 1.1)

    aligned = []
    for cand in candidates:
        best_dx = min(abs(cand["cx"] - col_x) for col_x in columns)
        if best_dx <= x_tol:
            aligned.append(cand)

    if not aligned:
        return []

    row_clusters = cluster_axis([c["cy"] for c in aligned], y_tol)
    rows = []
    for row in row_clusters:
        row_y = row["center"]
        support = 0
        for col_x in columns:
            has_neighbor = any(
                abs(c["cx"] - col_x) <= x_tol and abs(c["cy"] - row_y) <= y_tol
                for c in aligned
            )
            if has_neighbor:
                support += 1

        if support >= 2:
            rows.append(row_y)

    return rows


def assign_candidates_to_grid(candidates, columns, rows):
    if not candidates or not columns or not rows:
        return []

    radii = [c["r"] for c in candidates]
    median_r = float(np.median(radii)) if radii else 10.0

    col_spacings = np.diff(columns) if len(columns) >= 2 else np.array([median_r * 2.5])
    row_spacings = np.diff(rows) if len(rows) >= 2 else np.array([median_r * 2.5])

    x_tol = min(max(8.0, median_r * 1.25), float(np.median(col_spacings)) * 0.42)
    y_tol = min(max(8.0, median_r * 1.25), float(np.median(row_spacings)) * 0.45)

    accepted = []
    used_ids = set()

    for row_y in rows:
        row_hits = []
        for col_x in columns:
            best_idx = None
            best_score = None

            for idx, cand in enumerate(candidates):
                if idx in used_ids:
                    continue

                dx = abs(cand["cx"] - col_x)
                dy = abs(cand["cy"] - row_y)
                if dx > x_tol or dy > y_tol:
                    continue

                distance_penalty = (dx / x_tol) + (dy / y_tol)
                score = cand["score"] - (0.35 * distance_penalty)

                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                row_hits.append(best_idx)

        if len(row_hits) >= 2:
            for idx in row_hits:
                if idx not in used_ids:
                    accepted.append(candidates[idx])
                    used_ids.add(idx)

    accepted.sort(key=lambda c: (c["cy"], c["cx"]))
    return accepted


def draw_detected_bubbles(img, bubbles):
    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    for bubble in bubbles:
        x = int(round(bubble["cx"]))
        y = int(round(bubble["cy"]))
        r = int(round(bubble["r"]))
        cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
        cv2.circle(vis, (x, y), 2, (0, 0, 255), 2)

    return vis
