import os
import shutil
import json
import cv2
import numpy as np

from main import detect_bubbles, preprocess_bw_for_bubbles


def find_shear(img_bgr):
    """
    Găsește forfecarea (shear) verticală optimă a coloanelor.
    Maximizăm varianța proiecției pe axa X după binarizare.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    h, w = bw.shape
    best_shear = 0.0
    max_var = -1.0

    for sh in np.arange(-0.30, 0.31, 0.02):
        m = np.array([[1, sh, -sh * h / 2], [0, 1, 0]], dtype=np.float32)
        sheared_bw = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST)
        proj_x = np.sum(sheared_bw, axis=0)
        cur_var = float(np.var(proj_x))
        if cur_var > max_var:
            max_var = cur_var
            best_shear = float(sh)

    return best_shear


def _best_horizontal_bar_in_roi(roi_bw):
    cnts, _ = cv2.findContours(roi_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 180 or w < 16 or h < 6:
            continue
        ar = w / float(max(1, h))
        if ar < 2.2:
            continue
        fill = cv2.contourArea(c) / float(w * h + 1e-6)
        if fill < 0.35:
            continue
        score = (w * h) * min(ar, 10.0) * fill
        if score > best_score:
            best_score = score
            best = (x, y, w, h, ar, fill)
    return best


def _detect_corner_bars_points(img_bgr):
    """
    Detectează cele 4 bare negre de colț (TL, TR, BL, BR).
    Returnează punctele-centru în ordinea TL, TR, BL, BR.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thr = int(np.percentile(gray, 12))
    thr = max(45, min(200, thr))
    _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

    # Două seturi de ROI, pentru robustețe pe imagini mai imperfete.
    roi_sets = [
        {
            "tl": (0, 0, int(0.42 * w), int(0.22 * h)),
            "tr": (int(0.58 * w), 0, w, int(0.22 * h)),
            "bl": (0, int(0.78 * h), int(0.42 * w), h),
            "br": (int(0.58 * w), int(0.78 * h), w, h),
        },
        {
            "tl": (0, 0, int(0.50 * w), int(0.28 * h)),
            "tr": (int(0.50 * w), 0, w, int(0.28 * h)),
            "bl": (0, int(0.72 * h), int(0.50 * w), h),
            "br": (int(0.50 * w), int(0.72 * h), w, h),
        },
    ]

    best_pick = None
    for regions in roi_sets:
        picks = {}
        ok = True
        for name, (x0, y0, x1, y1) in regions.items():
            roi = bw[y0:y1, x0:x1]
            b = _best_horizontal_bar_in_roi(roi)
            if b is None:
                ok = False
                break
            x, y, ww, hh, ar, fill = b
            picks[name] = {
                "bbox": (x0 + x, y0 + y, ww, hh),
                "center": (x0 + x + ww / 2.0, y0 + y + hh / 2.0),
                "ar": float(ar),
                "fill": float(fill),
            }
        if not ok:
            continue

        tl = np.array(picks["tl"]["center"], dtype=np.float32)
        tr = np.array(picks["tr"]["center"], dtype=np.float32)
        bl = np.array(picks["bl"]["center"], dtype=np.float32)
        br = np.array(picks["br"]["center"], dtype=np.float32)

        w_top = float(np.linalg.norm(tr - tl))
        w_bot = float(np.linalg.norm(br - bl))
        h_left = float(np.linalg.norm(bl - tl))
        h_right = float(np.linalg.norm(br - tr))
        poly = np.array([tl, tr, br, bl], dtype=np.float32)
        area = abs(float(cv2.contourArea(poly.reshape(-1, 1, 2))))

        if w_top < 0.42 * w or w_bot < 0.42 * w:
            continue
        if h_left < 0.45 * h or h_right < 0.45 * h:
            continue
        if area < 0.22 * (w * h):
            continue

        geom_score = area + 0.5 * (w_top + w_bot) + 0.5 * (h_left + h_right)
        if best_pick is None or geom_score > best_pick["score"]:
            best_pick = {
                "score": geom_score,
                "points": np.array([tl, tr, bl, br], dtype=np.float32),
                "bars": picks,
                "threshold": thr,
            }

    if best_pick is None:
        return None, {"ok": False, "threshold": thr}
    return best_pick["points"], {
        "ok": True,
        "threshold": int(best_pick["threshold"]),
        "bars": best_pick["bars"],
    }


def normalize_by_corner_markers(img_bgr):
    """
    Perspective-normalize folosind centrele celor 4 bare negre din colțuri.
    """
    h, w = img_bgr.shape[:2]
    src, dbg = _detect_corner_bars_points(img_bgr)
    if src is None:
        return img_bgr, False, dbg

    tl, tr, bl, br = src
    left_x = int(round((tl[0] + bl[0]) / 2.0))
    right_x = int(round((tr[0] + br[0]) / 2.0))
    top_y = int(round((tl[1] + tr[1]) / 2.0))
    bot_y = int(round((bl[1] + br[1]) / 2.0))

    left_x = max(0, min(w - 2, left_x))
    right_x = max(left_x + 2, min(w - 1, right_x))
    top_y = max(0, min(h - 2, top_y))
    bot_y = max(top_y + 2, min(h - 1, bot_y))

    dst = np.array(
        [
            [left_x, top_y],
            [right_x, top_y],
            [left_x, bot_y],
            [right_x, bot_y],
        ],
        dtype=np.float32,
    )

    m = cv2.getPerspectiveTransform(src, dst)
    norm = cv2.warpPerspective(
        img_bgr, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    dbg["dst"] = dst.tolist()
    return norm, True, dbg


def draw_corner_markers_debug(img_bgr, marker_dbg):
    dbg_img = img_bgr.copy()
    if not marker_dbg or not marker_dbg.get("ok"):
        cv2.putText(
            dbg_img,
            "Corner markers: NOT FOUND",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        return dbg_img

    colors = {
        "tl": (255, 0, 255),
        "tr": (0, 255, 255),
        "bl": (255, 255, 0),
        "br": (0, 255, 0),
    }
    for name in ["tl", "tr", "bl", "br"]:
        if name not in marker_dbg.get("bars", {}):
            continue
        b = marker_dbg["bars"][name]
        x, y, w, h = b["bbox"]
        cx, cy = b["center"]
        color = colors[name]
        cv2.rectangle(dbg_img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        cv2.circle(dbg_img, (int(cx), int(cy)), 5, color, -1)
        cv2.putText(
            dbg_img,
            name.upper(),
            (int(x), max(18, int(y) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    tlb = marker_dbg["bars"]["tl"]["bbox"]
    trb = marker_dbg["bars"]["tr"]["bbox"]
    blb = marker_dbg["bars"]["bl"]["bbox"]
    brb = marker_dbg["bars"]["br"]["bbox"]
    quad = np.array(
        [
            [tlb[0], tlb[1]],
            [trb[0] + trb[2], trb[1]],
            [brb[0] + brb[2], brb[1] + brb[3]],
            [blb[0], blb[1] + blb[3]],
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))
    cv2.polylines(dbg_img, [quad], True, (0, 0, 255), 2)
    cv2.putText(
        dbg_img,
        f"thr={marker_dbg.get('threshold', '-')}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    return dbg_img


def _smooth_1d(arr, k):
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(arr.astype(np.float32).reshape(-1, 1), (k, 1), 0).ravel()


def _build_layout_binary(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bw_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    bw_adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10
    )
    bw = cv2.bitwise_or(bw_otsu, bw_adapt)
    bw = cv2.medianBlur(bw, 3)
    return bw


def _estimate_content_y_bounds(bw):
    h = bw.shape[0]

    proj_y = bw.sum(axis=1).astype(np.float32)
    proj_y = _smooth_1d(proj_y, max(7, h // 35))
    norm = proj_y / (proj_y.max() + 1e-6)
    idx = np.where(norm > 0.05)[0]

    if len(idx) == 0:
        return 0, h

    y_top = max(0, int(idx[0]) - 6)
    y_bot = min(h, int(idx[-1]) + 6)
    if y_bot - y_top < int(0.45 * h):
        return 0, h
    return y_top, y_bot


def _find_band_split_projection(bw, y_top, y_bot):
    """Split Y robust bazat pe minimul de proiecție în zona centrală."""
    h = bw.shape[0]

    proj_y = bw.sum(axis=1).astype(np.float32)
    kbig = min(101, max(9, h // 5))
    if kbig % 2 == 0:
        kbig += 1
    proj_smooth = cv2.GaussianBlur(proj_y.reshape(-1, 1), (kbig, 1), 0).ravel()
    proj_norm = proj_smooth / (proj_smooth.max() + 1e-6)

    window = max(15, h // 100)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    proj_roll = np.convolve(proj_norm, kernel, mode="same")

    search_a = int(h * 0.38)
    search_b = int(h * 0.62)
    if search_b <= search_a + 4:
        search_a = int(h * 0.30)
        search_b = int(h * 0.70)
    local = proj_roll[search_a:search_b]
    mid = search_a + int(np.argmin(local))

    min_band_h = int(0.22 * max(1, y_bot - y_top))
    if (mid - y_top) < min_band_h or (y_bot - mid) < min_band_h:
        mid = (y_top + y_bot) // 2

    return int(y_top), int(mid), int(y_bot)


def _find_x_projection_local_minima(bw, y_top, y_bot):
    """Separatoare X prin minim local în jurul pozițiilor 20/40/60/80%."""
    h, w = bw.shape
    y_top = max(0, min(h - 1, int(y_top)))
    y_bot = max(y_top + 1, min(h, int(y_bot)))

    roi = bw[y_top:y_bot, :]
    proj_x = roi.sum(axis=0).astype(np.float32)
    ksize = max(3, w // 40)
    if ksize % 2 == 0:
        ksize += 1
    proj_x = cv2.GaussianBlur(proj_x.reshape(1, -1), (1, ksize), 0).ravel()
    borders = [0]
    win = max(30, w // 10)
    for i in range(1, 5):
        target = int(round(i * w / 5.0))
        a = max(1, target - win)
        b = min(w - 2, target + win)
        sep = a + int(np.argmin(proj_x[a:b + 1]))
        borders.append(int(sep))
    borders.append(w)
    return borders


def _find_x_projection_gaps(bw, y_top, y_bot):
    """Fallback: separatoare pe baza celor mai mari gap-uri globale."""
    h, w = bw.shape
    y_top = max(0, min(h - 1, int(y_top)))
    y_bot = max(y_top + 1, min(h, int(y_bot)))
    roi = bw[y_top:y_bot, :]
    proj_x = roi.sum(axis=0).astype(np.float32)
    ksize = max(3, w // 50)
    if ksize % 2 == 0:
        ksize += 1
    proj_x = cv2.GaussianBlur(proj_x.reshape(1, -1), (1, ksize), 0).ravel()
    proj_x = proj_x / (proj_x.max() + 1e-6)
    is_content = proj_x > 0.10
    transitions = []
    for i in range(1, len(is_content)):
        if is_content[i - 1] and not is_content[i]:
            transitions.append(("end", i))
        elif not is_content[i - 1] and is_content[i]:
            transitions.append(("start", i))

    gaps = []
    for i in range(len(transitions) - 1):
        if transitions[i][0] == "end" and transitions[i + 1][0] == "start":
            g0 = transitions[i][1]
            g1 = transitions[i + 1][1]
            gw = g1 - g0
            gc = (g0 + g1) // 2
            gaps.append((gw, gc))

    if len(gaps) >= 4:
        gaps.sort(key=lambda x: x[0], reverse=True)
        separators = sorted([g[1] for g in gaps[:4]])
    else:
        separators = [w * (i + 1) // 5 for i in range(4)]

    return [0] + [int(s) for s in separators] + [w]


def _filter_bubbles_for_layout(bubbles):
    if not bubbles:
        return []
    diam = np.array([b["ellipse"][1][0] for b in bubbles], dtype=np.float32)
    med = float(np.median(diam))
    lo, hi = med * 0.72, med * 1.30
    out = [b for b in bubbles if lo <= b["ellipse"][1][0] <= hi]
    return out


def _kmeans_sorted_centers(vals, k, attempts=8):
    if vals is None or len(vals) < max(30, k * 5):
        return None
    arr = np.array(vals, dtype=np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 120, 0.2)
    _, _, centers = cv2.kmeans(arr, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    return np.sort(centers.flatten())


def _find_x_kmeans(points_xy, w):
    if points_xy is None or len(points_xy) < 250:
        return None
    c = _kmeans_sorted_centers(points_xy[:, 0], 25, attempts=10)
    if c is None:
        return None

    separators = [
        int((c[4] + c[5]) / 2.0),
        int((c[9] + c[10]) / 2.0),
        int((c[14] + c[15]) / 2.0),
        int((c[19] + c[20]) / 2.0),
    ]
    borders = [0] + separators + [w]
    return borders


def _find_y_kmeans(points_xy, h):
    if points_xy is None or len(points_xy) < 300:
        return None
    c = _kmeans_sorted_centers(points_xy[:, 1], 40, attempts=10)
    if c is None:
        return None
    mid = int((c[19] + c[20]) / 2.0)
    top_gap = float(np.median(np.diff(c[:20]))) if len(c) >= 20 else 16.0
    bot_gap = float(np.median(np.diff(c[20:]))) if len(c) >= 40 else 16.0
    y_top = max(0, int(c[0] - 0.9 * top_gap))
    y_bot = min(h, int(c[-1] + 0.9 * bot_gap))
    return y_top, mid, y_bot


def _x_borders_valid(borders, w):
    if len(borders) != 6:
        return False
    if any(borders[i] >= borders[i + 1] for i in range(5)):
        return False
    widths = [borders[i + 1] - borders[i] for i in range(5)]
    min_w = max(45, int(0.09 * w))
    max_w = int(0.36 * w)
    return all(min_w <= ww <= max_w for ww in widths)


def _regularize_borders(borders, w, blend=0.15):
    if len(borders) != 6:
        return borders

    out = [0]
    for i in range(1, 5):
        target = int(round(i * w / 5.0))
        raw = int(borders[i])
        reg = int(round((1.0 - blend) * raw + blend * target))
        out.append(reg)
    out.append(w)

    min_w = max(50, int(0.09 * w))
    for i in range(1, 5):
        out[i] = max(out[i], out[i - 1] + min_w)
    for i in range(4, 0, -1):
        out[i] = min(out[i], out[i + 1] - min_w)
    return out


def _collect_layout_points(unsheared_bgr, bw, y_top, y_bot):
    bubbles, _ = detect_bubbles(unsheared_bgr)
    bubbles = _filter_bubbles_for_layout(bubbles)
    pts = [(float(b["cx"]), float(b["cy"])) for b in bubbles]

    # Fallback pentru imagini cu contrast slab: conectăm și puncte din componente.
    if len(pts) < 220:
        n_labels, _, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
        for i in range(1, n_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            a = stats[i, cv2.CC_STAT_AREA]
            if a < 8 or a > 900 or w < 3 or h < 3:
                continue
            ar = w / float(h + 1e-6)
            fill = a / float(w * h + 1e-6)
            if ar < 0.4 or ar > 2.6 or fill < 0.25 or fill > 0.98:
                continue
            cx, cy = cents[i]
            if cy < y_top - 20 or cy > y_bot + 20:
                continue
            pts.append((float(cx), float(cy)))

    if not pts:
        return np.zeros((0, 2), dtype=np.float32)

    # Dedupe simplu pe celulă 3x3.
    uniq = {}
    for x, y in pts:
        key = (int(x // 3), int(y // 3))
        if key not in uniq:
            uniq[key] = (x, y)
    out = np.array(list(uniq.values()), dtype=np.float32)
    return out


def _boxes_from_layout(borders, y_top, mid, y_bot, h):
    y_top = max(0, int(y_top))
    y_bot = min(h, int(y_bot))
    mid = max(y_top + 20, min(int(mid), y_bot - 20))
    boxes = []
    for by0, by1 in [(y_top, mid), (mid, y_bot)]:
        for i in range(5):
            boxes.append((int(borders[i]), int(by0), int(borders[i + 1]), int(by1)))
    return boxes


def _evaluate_layout(points_xy, bw, borders, y_top, mid, y_bot):
    h, w = bw.shape
    if not _x_borders_valid(borders, w):
        return -1e9, None
    if y_top < 0 or y_bot > h or y_bot <= y_top + 60:
        return -1e9, None

    boxes = _boxes_from_layout(borders, y_top, mid, y_bot, h)
    if len(boxes) != 10:
        return -1e9, None

    if points_xy is None or len(points_xy) == 0:
        return -1e9, None

    px = points_xy[:, 0]
    py = points_xy[:, 1]
    counts = []
    for x0, by0, x1, by1 in boxes:
        inside = (px >= x0) & (px < x1) & (py >= by0) & (py < by1)
        counts.append(int(np.count_nonzero(inside)))
    counts = np.array(counts, dtype=np.float32)

    total_points = float(len(points_xy))
    total_in = float(counts.sum())
    coverage = total_in / (total_points + 1e-6)
    mean_c = float(np.mean(counts)) + 1e-6
    std_c = float(np.std(counts))
    min_c = float(np.min(counts))

    uniform = 1.0 / (1.0 + std_c / mean_c)
    min_frac = min(1.0, min_c / (0.55 * mean_c + 1e-6))

    top_total = float(np.sum(counts[:5]))
    bot_total = float(np.sum(counts[5:]))
    balance = 1.0 - abs(top_total - bot_total) / (top_total + bot_total + 1e-6)

    proj_x = bw[max(0, y_top):min(h, y_bot), :].sum(axis=0).astype(np.float32)
    proj_x = proj_x / (proj_x.max() + 1e-6)
    sep_vals = []
    for s in borders[1:5]:
        s = max(0, min(w - 1, int(s)))
        sep_vals.append(float(proj_x[s]))
    valley_x = 1.0 - float(np.mean(sep_vals))

    proj_y = bw.sum(axis=1).astype(np.float32)
    proj_y = proj_y / (proj_y.max() + 1e-6)
    mid = max(0, min(h - 1, int(mid)))
    valley_y = 1.0 - float(proj_y[mid])

    widths = np.array([borders[i + 1] - borders[i] for i in range(5)], dtype=np.float32)
    w_uniform = 1.0 / (1.0 + float(np.std(widths)) / (float(np.mean(widths)) + 1e-6))

    score = (
        2.4 * coverage
        + 1.4 * uniform
        + 1.1 * min_frac
        + 0.9 * balance
        + 0.7 * valley_x
        + 0.45 * valley_y
        + 0.4 * w_uniform
    )
    if min_c < 10:
        score -= 0.8

    info = {
        "counts": counts.tolist(),
        "coverage": coverage,
        "uniform": uniform,
        "min_frac": min_frac,
        "balance": balance,
        "valley_x": valley_x,
        "valley_y": valley_y,
    }
    return score, info


def _dedupe_x_candidates(candidates):
    out = []
    seen = set()
    for c in candidates:
        if c is None or len(c) != 6:
            continue
        key = tuple(int(round(v / 8.0) * 8) for v in c)
        if key in seen:
            continue
        seen.add(key)
        out.append([int(v) for v in c])
    return out


def _dedupe_y_candidates(candidates):
    out = []
    seen = set()
    for c in candidates:
        if c is None or len(c) != 3:
            continue
        key = tuple(int(round(v / 8.0) * 8) for v in c)
        if key in seen:
            continue
        seen.add(key)
        out.append(tuple(int(v) for v in c))
    return out


def find_shear_candidates(img_bgr, step=0.02, limit=5):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    h, w = bw.shape

    cand = []
    for sh in np.arange(-0.30, 0.301, step):
        m = np.array([[1, sh, -sh * h / 2], [0, 1, 0]], dtype=np.float32)
        sheared = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST)
        var_x = float(np.var(np.sum(sheared, axis=0)))
        cand.append((var_x, float(sh)))
    cand.sort(reverse=True, key=lambda t: t[0])

    def _add_unique(arr, v, thr=0.018):
        if all(abs(v - x) >= thr for x in arr):
            arr.append(float(v))

    # Candidate "safe" în jurul lui 0 pentru a evita deformațiile excesive.
    core = [0.0, -step, step, -2 * step, 2 * step, -3 * step, 3 * step]
    selected = []
    for sh in core:
        _add_unique(selected, float(np.clip(sh, -0.30, 0.30)))

    # Complementează cu opțiuni data-driven, dar păstrăm diversitate.
    for _, sh in cand:
        if abs(sh) < 1e-9:
            continue
        if any(abs(sh - s) < 0.035 for s in selected):
            continue
        selected.append(float(sh))
        if len(selected) >= max(limit, len(core) + 2):
            break

    # Respectăm limita dar păstrăm mereu 0 și vecinii apropiați.
    out = selected[: max(1, limit)]
    if all(abs(x) > 1e-9 for x in out):
        out = [0.0] + out[:-1]
    return out


def find_layout_10_boxes(unsheared_bgr):
    h, w = unsheared_bgr.shape[:2]
    bw = _build_layout_binary(unsheared_bgr)
    y_top0, y_bot0 = _estimate_content_y_bounds(bw)
    points_xy = _collect_layout_points(unsheared_bgr, bw, y_top0, y_bot0)

    y_proj = _find_band_split_projection(bw, y_top0, y_bot0)
    y_ref_mid = y_proj[1]
    y_km = _find_y_kmeans(points_xy, h)
    y_eq = (y_top0, (y_top0 + y_bot0) // 2, y_bot0)
    y_candidates_raw = [y_proj, y_eq]
    if y_km is not None:
        # Folosim k-means doar pentru split-ul central; marginile sus/jos rămân cele
        # detectate din conținut pentru a evita crop-ul agresiv.
        y_candidates_raw.append((y_top0, y_km[1], y_bot0))
    y_candidates = _dedupe_y_candidates(y_candidates_raw)

    x_proj_local = _find_x_projection_local_minima(bw, y_top0, y_bot0)
    x_proj_gaps = _find_x_projection_gaps(bw, y_top0, y_bot0)
    x_km = _find_x_kmeans(points_xy, w)
    x_eq = [int(round(i * w / 5.0)) for i in range(6)]
    x_candidates = _dedupe_x_candidates([x_proj_local, x_proj_gaps, x_km, x_eq])

    base_span = max(1, y_bot0 - y_top0)
    mid_center = (y_top0 + y_bot0) / 2.0
    best_score = -1e9
    best = None
    for y_top, mid_split, y_bot in y_candidates:
        span = y_bot - y_top
        if span < int(0.82 * base_span):
            continue

        center_drift = abs(mid_split - mid_center) / float(base_span + 1e-6)
        if center_drift > 0.16:
            continue

        for borders in x_candidates:
            borders = _regularize_borders(borders, w, blend=0.15)
            score, info = _evaluate_layout(points_xy, bw, borders, y_top, mid_split, y_bot)
            if score < -1e8:
                continue

            # Penalizare pentru crop excesiv față de zona de conținut detectată.
            drift = (abs(y_top - y_top0) + abs(y_bot - y_bot0)) / float(base_span + 1e-6)
            score -= 1.15 * drift

            # Penalizare puternică dacă split-ul central se depărtează de valea
            # din proiecția Y (de obicei separatorul real între cele 2 benzi).
            mid_drift = abs(mid_split - y_ref_mid) / float(base_span + 1e-6)
            score -= 2.25 * mid_drift
            if mid_drift > 0.20:
                score -= 1.0
            score -= 1.6 * center_drift

            if score > best_score:
                best_score = score
                best = (borders, y_top, mid_split, y_bot, info)

    if best is None:
        borders = [int(round(i * w / 5.0)) for i in range(6)]
        y_top, y_bot = 0, h
        mid_split = h // 2
        boxes = _boxes_from_layout(borders, y_top, mid_split, y_bot, h)
        dbg = {
            "y_top": y_top,
            "mid_split": mid_split,
            "y_bot": y_bot,
            "borders": borders,
            "layout_score": -1.0,
        }
        return boxes, dbg

    borders, y_top, mid_split, y_bot, info = best
    boxes = _boxes_from_layout(borders, y_top, mid_split, y_bot, h)
    dbg = {
        "y_top": int(y_top),
        "mid_split": int(mid_split),
        "y_bot": int(y_bot),
        "borders": [int(x) for x in borders],
        "layout_score": float(best_score),
        "coverage": float(info["coverage"]) if info else 0.0,
        "uniform": float(info["uniform"]) if info else 0.0,
    }
    return boxes, dbg


def _filter_bubbles_like_main(roi_bgr):
    """
    Filtrare în aceeași linie cu read_answers_from_bubbles():
    - excludem stânga (zona de numerotare);
    - păstrăm diametre apropiate de mediană.
    """
    raw_bubbles, _ = detect_bubbles(roi_bgr)
    if not raw_bubbles:
        return []

    _, w = roi_bgr.shape[:2]
    crop_x = int(w * 0.15)
    candidates = [b for b in raw_bubbles if b["cx"] >= crop_x]
    if not candidates:
        return []

    diam = np.array([b["ellipse"][1][0] for b in candidates], dtype=np.float32)
    med = float(np.median(diam))
    lo, hi = med * 0.75, med * 1.25
    filtered = [b for b in candidates if lo <= b["ellipse"][1][0] <= hi]
    return filtered


def _contrast_metrics(gray):
    g = gray.astype(np.float32)
    p05 = float(np.percentile(g, 5))
    p10 = float(np.percentile(g, 10))
    p50 = float(np.percentile(g, 50))
    p90 = float(np.percentile(g, 90))
    p95 = float(np.percentile(g, 95))
    spread = float(p90 - p10)
    std = float(np.std(g))
    return {
        "p05": p05,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "spread": spread,
        "std": std,
    }


def _should_use_clahe(metrics):
    spread = float(metrics.get("spread", 999.0))
    std = float(metrics.get("std", 999.0))
    return bool(spread < 82.0 or std < 34.0)


def _bubble_candidate_score(edge_ref, bubble, source):
    (cx0, cy0), (a0, b0), ang = bubble["ellipse"]
    ma = float(max(a0, b0))
    mi = float(min(a0, b0))
    if ma <= 0.0 or mi <= 0.0:
        return None

    x_ref, y_ref, refine_gain = _refine_center_on_edges(
        edge_ref,
        cx0,
        cy0,
        ma,
        mi,
        max_dx=2,
        max_dy=2,
        dist_penalty=0.35,
    )
    edge0 = _ring_edge_score(edge_ref, int(round(float(cx0))), int(round(float(cy0))), ma, mi) / 255.0
    edge1 = _ring_edge_score(edge_ref, int(round(float(x_ref))), int(round(float(y_ref))), ma, mi) / 255.0
    edge_score = max(float(edge0), float(edge1))
    score = 1.40 * edge_score + 0.055 * float(refine_gain)
    if source == "raw":
        score += 0.035

    cand = dict(bubble)
    cand["cx"] = int(round(float(x_ref)))
    cand["cy"] = int(round(float(y_ref)))
    cand["ellipse"] = ((float(x_ref), float(y_ref)), (ma, mi), float(ang))
    cand["candidate_source"] = str(source)
    cand["candidate_score"] = float(score)
    cand["candidate_edge_score"] = float(edge_score)
    cand["candidate_refine_gain"] = float(refine_gain)
    return cand


def _dedupe_candidates_by_score(candidates, max_dx=6, max_dy=6):
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda b: float(b.get("candidate_score", 0.0)), reverse=True)
    kept = []
    for b in ordered:
        repl = -1
        for i, d in enumerate(kept):
            if abs(int(b["cx"]) - int(d["cx"])) <= max_dx and abs(int(b["cy"]) - int(d["cy"])) <= max_dy:
                repl = i
                break
        if repl < 0:
            kept.append(b)
            continue
        if float(b.get("candidate_score", -1e9)) > float(kept[repl].get("candidate_score", -1e9)):
            kept[repl] = b

    kept.sort(key=lambda b: (int(b["cy"]), int(b["cx"])))
    return kept


def _detect_bubbles_robust_union(roi_bgr, return_meta=False):
    """
    Detector robust:
      - view raw;
      - view CLAHE doar când contrastul local este slab;
      - union + dedup pe candidați, păstrând scorul local mai bun.
    """
    h, w = roi_bgr.shape[:2]
    gray_raw = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray_raw, (3, 3), 0)
    edge_ref = cv2.Canny(gray_blur, 40, 120)
    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag, 0.0, 255.0).astype(np.uint8)
    edge_ref = cv2.max(edge_ref, mag)

    contrast = _contrast_metrics(gray_raw)
    use_clahe = _should_use_clahe(contrast)

    raw_bubbles, _ = detect_bubbles(roi_bgr)
    clahe_bubbles = []
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray_raw)
        edge_clahe = cv2.Canny(cv2.GaussianBlur(gray_clahe, (3, 3), 0), 36, 110)
        gx_c = cv2.Sobel(gray_clahe, cv2.CV_32F, 1, 0, ksize=3)
        gy_c = cv2.Sobel(gray_clahe, cv2.CV_32F, 0, 1, ksize=3)
        mag_c = cv2.magnitude(gx_c, gy_c)
        mag_c = np.clip(mag_c, 0.0, 255.0).astype(np.uint8)
        edge_clahe = cv2.max(edge_clahe, mag_c)
        edge_ref = cv2.bitwise_or(edge_ref, edge_clahe)
        clahe_bgr = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)
        clahe_bubbles, _ = detect_bubbles(clahe_bgr)

    # Excludem zona de numerotare din stânga (major source de false positives).
    crop_x = int(w * 0.14)
    raw_bubbles = [b for b in raw_bubbles if int(b["cx"]) >= crop_x]
    clahe_bubbles = [b for b in clahe_bubbles if int(b["cx"]) >= crop_x]

    candidates = []
    for b in raw_bubbles:
        cand = _bubble_candidate_score(edge_ref, b, source="raw")
        if cand is not None:
            candidates.append(cand)
    for b in clahe_bubbles:
        cand = _bubble_candidate_score(edge_ref, b, source="clahe")
        if cand is not None:
            candidates.append(cand)

    merged = _dedupe_candidates_by_score(candidates, max_dx=6, max_dy=6)
    if not merged:
        meta = {
            "used_clahe": bool(use_clahe),
            "contrast": contrast,
            "raw_candidates": int(len(raw_bubbles)),
            "clahe_candidates": int(len(clahe_bubbles)),
            "merged_candidates": 0,
        }
        if return_meta:
            return [], meta
        return []

    diam = np.array([float(max(b["ellipse"][1][0], b["ellipse"][1][1])) for b in merged], dtype=np.float32)
    med = float(np.median(diam))
    lo, hi = med * 0.62, med * 1.55
    merged = [b for b in merged if lo <= float(max(b["ellipse"][1][0], b["ellipse"][1][1])) <= hi]

    meta = {
        "used_clahe": bool(use_clahe),
        "contrast": contrast,
        "raw_candidates": int(len(raw_bubbles)),
        "clahe_candidates": int(len(clahe_bubbles)),
        "merged_candidates": int(len(merged)),
    }
    if return_meta:
        return merged, meta
    return merged


def _estimate_axis_centers(vals, k, q_low, q_high):
    vals = np.array(vals, dtype=np.float32)
    if vals.size == 0:
        return None

    v0 = float(np.percentile(vals, q_low))
    v1 = float(np.percentile(vals, q_high))
    lin = np.linspace(v0, v1, k)

    if vals.size >= max(20, k * 6):
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 120, 0.2)
        arr = vals.reshape(-1, 1)
        _, _, centers = cv2.kmeans(arr, k, None, crit, 8, cv2.KMEANS_PP_CENTERS)
        km = np.sort(centers.flatten())
        # K-means este mai robust pe distribuții asimetrice (ex: cifrele întrebărilor
        # detectate accidental într-o parte a ROI-ului).
        return 0.80 * km + 0.20 * lin
    return lin


def _dedupe_sorted_x(xs, min_gap=7.0):
    xs = [float(x) for x in sorted(xs)]
    if not xs:
        return []

    out = [xs[0]]
    for x in xs[1:]:
        if abs(x - out[-1]) <= min_gap:
            out[-1] = 0.5 * (out[-1] + x)
        else:
            out.append(x)
    return out


def _estimate_col_centers_from_rows(bubbles, row_centers, n_choices=5):
    """
    Estimează coloanele A-E din ferestre pe rând:
    pentru fiecare rând alegem subsecvența de 5 X cu spacing cel mai uniform.
    Astfel ignorăm robust coloanele false (din stânga/dreapta).
    """
    if not bubbles or row_centers is None:
        return None

    row_centers = np.array(row_centers, dtype=np.float32)
    if row_centers.size < 3:
        return None

    row_step = float(np.median(np.diff(row_centers))) if row_centers.size > 1 else 20.0
    gate_y = max(6.0, 0.55 * row_step)
    row_windows = []

    for y in row_centers:
        xs = [float(b["cx"]) for b in bubbles if abs(float(b["cy"]) - float(y)) <= gate_y]
        if len(xs) < n_choices:
            continue

        xs = _dedupe_sorted_x(xs, min_gap=7.0)
        if len(xs) < n_choices:
            continue

        best_win = None
        best_score = 1e9
        for i in range(len(xs) - n_choices + 1):
            win = np.array(xs[i : i + n_choices], dtype=np.float32)
            gaps = np.diff(win)
            if gaps.size != n_choices - 1 or np.any(gaps < 6.0):
                continue

            mean_gap = float(np.mean(gaps))
            std_gap = float(np.std(gaps))
            score = std_gap / (mean_gap + 1e-6)
            if score < best_score:
                best_score = score
                best_win = win

        if best_win is not None:
            row_windows.append(best_win)

    min_rows = max(6, int(0.35 * row_centers.size))
    if len(row_windows) < min_rows:
        return None

    col = np.median(np.array(row_windows, dtype=np.float32), axis=0)
    col = np.sort(col).astype(np.float32)
    for i in range(1, col.size):
        if col[i] - col[i - 1] < 8.0:
            col[i] = col[i - 1] + 8.0
    return col


def _ellipse_offsets(ax, ay, n=56):
    if ax < 2 or ay < 2:
        return np.zeros((0, 2), dtype=np.int32)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    ox = np.round(ax * np.cos(theta)).astype(np.int32)
    oy = np.round(ay * np.sin(theta)).astype(np.int32)
    pts = np.stack([ox, oy], axis=1)
    pts = np.unique(pts, axis=0)
    return pts.astype(np.int32)


def _refine_center_on_edges(
    edge,
    x,
    y,
    ma,
    mi,
    max_dx=4,
    max_dy=4,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    dist_penalty=0.0,
):
    """
    Micro-ajustare a centrului pe baza energiei de muchii pe două inele eliptice.
    """
    h, w = edge.shape[:2]
    x0 = int(round(float(x)))
    y0 = int(round(float(y)))

    ax = max(3, int(round(float(ma) / 2.0)))
    ay = max(3, int(round(float(mi) / 2.0)))
    off_outer = _ellipse_offsets(ax, ay, n=64)
    off_inner = _ellipse_offsets(max(2, int(round(0.72 * ax))), max(2, int(round(0.72 * ay))), n=64)
    if off_outer.size == 0 or off_inner.size == 0:
        return float(x0), float(y0), 0.0

    def score_at(cx, cy):
        if cx - ax - 1 < 0 or cx + ax + 1 >= w or cy - ay - 1 < 0 or cy + ay + 1 >= h:
            return -1e9
        if x_min is not None and cx < int(x_min):
            return -1e9
        if x_max is not None and cx > int(x_max):
            return -1e9
        if y_min is not None and cy < int(y_min):
            return -1e9
        if y_max is not None and cy > int(y_max):
            return -1e9
        p1 = edge[cy + off_outer[:, 1], cx + off_outer[:, 0]].astype(np.float32)
        p2 = edge[cy + off_inner[:, 1], cx + off_inner[:, 0]].astype(np.float32)
        s = float(np.mean(p1) + 0.65 * np.mean(p2))
        if dist_penalty > 1e-6:
            ndx = abs(cx - x0) / float(max(1, max_dx))
            ndy = abs(cy - y0) / float(max(1, max_dy))
            s -= float(dist_penalty) * (ndx + ndy)
        return s

    base = score_at(x0, y0)
    best_s = base
    best_x, best_y = x0, y0
    for dy in range(-max_dy, max_dy + 1):
        cy = y0 + dy
        for dx in range(-max_dx, max_dx + 1):
            cx = x0 + dx
            s = score_at(cx, cy)
            if s > best_s:
                best_s = s
                best_x, best_y = cx, cy

    # Aplicăm ajustarea doar dacă aduce un câștig real al răspunsului de contur.
    gain = float(best_s - base)
    if gain >= 1.5:
        return float(best_x), float(best_y), gain
    return float(x0), float(y0), 0.0


def _apply_affine_point(m_2x3, x, y):
    xx = float(m_2x3[0, 0] * x + m_2x3[0, 1] * y + m_2x3[0, 2])
    yy = float(m_2x3[1, 0] * x + m_2x3[1, 1] * y + m_2x3[1, 2])
    return xx, yy


def _fit_cell_affine(seed_x, seed_y, local_map, n_rows, n_choices, col_step, row_step):
    src = []
    dst = []
    for r in range(n_rows):
        for c in range(n_choices):
            lf = local_map.get((r, c))
            if lf is None:
                continue
            lx, ly, _, _, lscore = lf
            if lscore < 0.12:
                continue
            sx = float(seed_x[r, c])
            sy = float(seed_y[r, c])
            if abs(float(lx) - sx) > 1.10 * col_step:
                continue
            if abs(float(ly) - sy) > 1.00 * row_step:
                continue
            src.append([sx, sy])
            dst.append([float(lx), float(ly)])

    if len(src) < 8:
        return None, 0

    src = np.array(src, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32)
    thr = max(2.5, 0.22 * min(col_step, row_step))
    m, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=thr,
        maxIters=4000,
        confidence=0.995,
        refineIters=25,
    )
    if m is None:
        return None, 0
    nin = int(np.count_nonzero(inliers)) if inliers is not None else 0
    if nin < 6:
        return None, nin
    return m.astype(np.float32), nin


def _fit_bilinear_grid(matches_rcxy, n_rows, n_choices, inlier_thr):
    """
    Potrivește:
      x = a0 + a1*u + a2*v + a3*u*v
      y = b0 + b1*u + b2*v + b3*u*v
    unde u=c/(n_choices-1), v=r/(n_rows-1)
    """
    if len(matches_rcxy) < 12:
        return None

    rows = np.array([m[0] for m in matches_rcxy], dtype=np.float32)
    cols = np.array([m[1] for m in matches_rcxy], dtype=np.float32)
    x_obs = np.array([m[2] for m in matches_rcxy], dtype=np.float32)
    y_obs = np.array([m[3] for m in matches_rcxy], dtype=np.float32)

    den_u = float(max(1, n_choices - 1))
    den_v = float(max(1, n_rows - 1))
    u = cols / den_u
    v = rows / den_v
    A = np.stack([np.ones_like(u), u, v, u * v], axis=1).astype(np.float32)

    def _solve(a_mat, xv, yv):
        cx, *_ = np.linalg.lstsq(a_mat, xv, rcond=None)
        cy, *_ = np.linalg.lstsq(a_mat, yv, rcond=None)
        return cx.astype(np.float32), cy.astype(np.float32)

    cx, cy = _solve(A, x_obs, y_obs)
    x_pred = A @ cx
    y_pred = A @ cy
    res = np.sqrt((x_pred - x_obs) ** 2 + (y_pred - y_obs) ** 2)
    keep = res <= float(inlier_thr)

    if int(np.count_nonzero(keep)) >= 12 and int(np.count_nonzero(keep)) < len(matches_rcxy):
        A2 = A[keep]
        x2 = x_obs[keep]
        y2 = y_obs[keep]
        cx, cy = _solve(A2, x2, y2)

    return cx, cy


def _eval_bilinear(coeff, row, col, n_rows, n_choices):
    u = float(col) / float(max(1, n_choices - 1))
    v = float(row) / float(max(1, n_rows - 1))
    return float(coeff[0] + coeff[1] * u + coeff[2] * v + coeff[3] * u * v)


def _blend_centers_with_anchors(base_centers, anchor_centers, min_points, blend, min_gap):
    base = np.array(base_centers, dtype=np.float32)
    anc = np.array(anchor_centers, dtype=np.float32)
    valid = np.isfinite(anc)
    if int(np.count_nonzero(valid)) < int(min_points):
        return base

    idx = np.where(valid)[0].astype(np.float32)
    vals = anc[valid].astype(np.float32)
    interp = np.interp(np.arange(base.size, dtype=np.float32), idx, vals).astype(np.float32)
    out = (1.0 - float(blend)) * base + float(blend) * interp

    step = float(np.median(np.diff(base))) if base.size > 1 else 10.0
    lim = 0.70 * step
    out = np.clip(out, base - lim, base + lim)

    mg = float(max(2.0, min_gap))
    for i in range(1, out.size):
        if out[i] < out[i - 1] + mg:
            out[i] = out[i - 1] + mg
    return out.astype(np.float32)


def _axis_from_observations(obs_vals, fallback_vals, min_gap, min_valid):
    """
    Reface un ax (rânduri sau coloane) din observații parțiale robuste.
    Dacă avem puține observații, întoarce fallback.
    """
    out = np.array(fallback_vals, dtype=np.float32)
    obs = np.array(obs_vals, dtype=np.float32)
    valid = np.isfinite(obs)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < int(min_valid):
        return out, n_valid

    idx = np.where(valid)[0].astype(np.float32)
    vals = obs[valid].astype(np.float32)
    interp = np.interp(np.arange(out.size, dtype=np.float32), idx, vals).astype(np.float32)

    if n_valid >= 2:
        A = np.stack([idx, np.ones_like(idx)], axis=1).astype(np.float32)
        coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
        lin = (coef[0] * np.arange(out.size, dtype=np.float32) + coef[1]).astype(np.float32)
    else:
        lin = interp.copy()

    conf = float(np.clip((n_valid - min_valid) / max(1, out.size - min_valid), 0.0, 1.0))
    blend_obs = (0.72 * interp + 0.28 * lin).astype(np.float32)
    alpha = float(0.58 + 0.26 * conf)
    out = ((1.0 - alpha) * out + alpha * blend_obs).astype(np.float32)

    step = float(np.median(np.diff(np.array(fallback_vals, dtype=np.float32)))) if out.size > 1 else 10.0
    lim = 0.95 * step
    out = np.clip(out, np.array(fallback_vals, dtype=np.float32) - lim, np.array(fallback_vals, dtype=np.float32) + lim)

    mg = float(max(2.0, min_gap))
    for i in range(1, out.size):
        if out[i] < out[i - 1] + mg:
            out[i] = out[i - 1] + mg
    return out.astype(np.float32), n_valid


def _assign_bubbles_to_cells(bubbles, row_centers, col_centers, gate_x, gate_y, n_rows, n_choices):
    cells = {}
    for b in bubbles:
        r = int(np.argmin(np.abs(row_centers - float(b["cy"]))))
        c = int(np.argmin(np.abs(col_centers - float(b["cx"]))))
        dx = abs(float(b["cx"]) - float(col_centers[c]))
        dy = abs(float(b["cy"]) - float(row_centers[r]))
        if dx <= gate_x * 1.10 and dy <= gate_y * 1.20:
            cells.setdefault((r, c), []).append(b)

    chosen_cell = {}
    matches = []
    reliable = {}
    for r in range(n_rows):
        for c in range(n_choices):
            cand = cells.get((r, c), [])
            if not cand:
                continue
            x0 = float(col_centers[c])
            y0 = float(row_centers[r])
            best = None
            best_d = 1e9
            for b in cand:
                d = abs(float(b["cx"]) - x0) + abs(float(b["cy"]) - y0)
                if d < best_d:
                    best_d = d
                    best = b
            if best is None:
                continue

            chosen_cell[(r, c)] = best
            dx0 = abs(float(best["cx"]) - x0)
            dy0 = abs(float(best["cy"]) - y0)
            if dx0 <= gate_x * 0.65 and dy0 <= gate_y * 0.65:
                matches.append((r, c, float(best["cx"]), float(best["cy"])))
            if dx0 <= gate_x * 0.50 and dy0 <= gate_y * 0.50:
                reliable[(r, c)] = True

    return chosen_cell, matches, reliable


def _centers_to_bounds(centers, lo, hi):
    c = np.array(centers, dtype=np.float32)
    n = c.size
    b = np.zeros((n + 1,), dtype=np.float32)
    if n == 1:
        b[0] = float(lo)
        b[1] = float(hi)
        return b

    mids = 0.5 * (c[:-1] + c[1:])
    b[1:-1] = mids
    b[0] = c[0] - 0.5 * (c[1] - c[0])
    b[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    b[0] = max(float(lo), b[0])
    b[-1] = min(float(hi), b[-1])
    for i in range(1, b.size):
        if b[i] <= b[i - 1] + 1.0:
            b[i] = b[i - 1] + 1.0
    return b


def _best_cell_candidate(candidates, x_ref, y_ref, ma_ref, mi_ref, col_step, row_step, ma_std, mi_std):
    if not candidates:
        return None, 1e9

    sx = max(4.0, 0.45 * col_step)
    sy = max(4.0, 0.45 * row_step)
    sm = max(2.0, 0.35 * ma_std)
    sn = max(2.0, 0.35 * mi_std)

    best = None
    best_s = 1e9
    for b in candidates:
        (_, _), (ma, mi), _ = b["ellipse"]
        s = (
            abs(float(b["cx"]) - float(x_ref)) / sx
            + abs(float(b["cy"]) - float(y_ref)) / sy
            + 0.35 * abs(float(ma) - float(ma_ref)) / sm
            + 0.35 * abs(float(mi) - float(mi_ref)) / sn
        )
        if s < best_s:
            best_s = s
            best = b
    return best, float(best_s)


def _ring_edge_score(edge, cx, cy, ma, mi):
    h, w = edge.shape[:2]
    ax = max(3, int(round(float(ma) / 2.0)))
    ay = max(3, int(round(float(mi) / 2.0)))
    if ax < 2 or ay < 2:
        return 0.0
    if cx - ax - 1 < 0 or cx + ax + 1 >= w or cy - ay - 1 < 0 or cy + ay + 1 >= h:
        return 0.0

    out = _ellipse_offsets(ax, ay, n=64)
    inn = _ellipse_offsets(max(2, int(round(0.72 * ax))), max(2, int(round(0.72 * ay))), n=64)
    if out.size == 0 or inn.size == 0:
        return 0.0

    p1 = edge[cy + out[:, 1], cx + out[:, 0]].astype(np.float32)
    p2 = edge[cy + inn[:, 1], cx + inn[:, 0]].astype(np.float32)
    return float(np.mean(p1) + 0.65 * np.mean(p2))


def _ellipse_fill_ratio_raw(bw_raw, cx, cy, ma, mi, angle_deg=0.0, shrink=0.88):
    h, w = bw_raw.shape[:2]
    cxi = int(round(float(cx)))
    cyi = int(round(float(cy)))
    ax = max(1, int(round(float(ma) * float(shrink) / 2.0)))
    ay = max(1, int(round(float(mi) * float(shrink) / 2.0)))
    if cxi < 0 or cyi < 0 or cxi >= w or cyi >= h:
        return 0.0

    mask = np.zeros_like(bw_raw, dtype=np.uint8)
    cv2.ellipse(mask, (cxi, cyi), (ax, ay), float(angle_deg), 0, 360, 255, -1)
    inside = bw_raw[mask == 255]
    if inside.size == 0:
        return 0.0
    return float(np.count_nonzero(inside)) / float(inside.size)


def _cell_confidence(edge_score, local_score, refine_gain, shift_from_grid, col_step, row_step):
    edge_term = float(np.clip(float(edge_score) / 0.26, 0.0, 1.0))
    local_term = float(np.clip((float(local_score) + 0.25) / 0.55, 0.0, 1.0))
    refine_term = float(np.clip(float(refine_gain) / 5.0, 0.0, 1.0))
    step_ref = max(1.0, 0.85 * max(float(col_step), float(row_step)))
    shift_term = float(np.clip(1.0 - float(shift_from_grid) / step_ref, 0.0, 1.0))
    conf = 0.48 * edge_term + 0.17 * local_term + 0.17 * refine_term + 0.18 * shift_term
    return float(np.clip(conf, 0.0, 1.0))


def _evaluate_cell_qc(edge_score, seed_chosen, has_local_evidence, shift_from_lattice, ma, mi, col_step, row_step):
    qc_ok = True
    qc_reasons = []
    step_ref = max(col_step, row_step)
    if edge_score < 0.09:
        qc_ok = False
        qc_reasons.append("weak_edge")
    if (not seed_chosen) and (not has_local_evidence) and edge_score < 0.12:
        qc_ok = False
        qc_reasons.append("ambiguous_center")
    if shift_from_lattice > 1.10 * step_ref and edge_score < 0.16:
        qc_ok = False
        qc_reasons.append("large_shift")
    if ma <= 0.0 or mi <= 0.0:
        qc_ok = False
        qc_reasons.append("invalid_size")
    if not qc_reasons:
        qc_reasons.append("ok")
    return bool(qc_ok), ",".join(qc_reasons)


def _fit_local_ellipse_in_cell(
    bw,
    edge,
    xl,
    yt,
    xr,
    yb,
    x_seed,
    y_seed,
    ma_ref,
    mi_ref,
    col_step,
    row_step,
    ma_std,
    mi_std,
):
    h, w = bw.shape[:2]
    xpad = max(2, int(round(0.18 * max(8.0, col_step))))
    ypad = max(2, int(round(0.18 * max(8.0, row_step))))
    sx0 = max(0, int(xl) - xpad)
    sy0 = max(0, int(yt) - ypad)
    sx1 = min(w - 1, int(xr) + xpad)
    sy1 = min(h - 1, int(yb) + ypad)
    if sx1 <= sx0 + 4 or sy1 <= sy0 + 4:
        return None

    patch = bw[sy0 : sy1 + 1, sx0 : sx1 + 1]
    cnts, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None

    min_ma = max(6.0, 0.58 * ma_std)
    max_ma = min(1.55 * ma_std, 0.92 * max(8.0, col_step))
    min_mi = max(6.0, 0.58 * mi_std)
    max_mi = min(1.55 * mi_std, 0.95 * max(8.0, row_step))

    best = None
    best_score = -1e9
    for cnt in cnts:
        if len(cnt) < 18:
            continue
        cnt2 = cnt.reshape(-1, 2).astype(np.float32)
        cnt2[:, 0] += float(sx0)
        cnt2[:, 1] += float(sy0)

        if cnt2.shape[0] < 5:
            continue
        ellipse = cv2.fitEllipse(cnt2.reshape(-1, 1, 2))
        (cx, cy), (a0, b0), _ = ellipse
        ma = float(max(a0, b0))
        mi = float(min(a0, b0))
        if ma <= 0 or mi <= 0:
            continue
        if ma < min_ma or ma > max_ma or mi < min_mi or mi > max_mi:
            continue
        ar = ma / max(1e-6, mi)
        if ar > 1.9:
            continue

        # Centru trebuie să rămână în celulă (cu marjă mică).
        if not (int(xl) - 1 <= cx <= int(xr) + 1 and int(yt) - 1 <= cy <= int(yb) + 1):
            continue

        edge_s = _ring_edge_score(edge, int(round(cx)), int(round(cy)), ma, mi) / 255.0
        dist_pen = abs(float(cx) - float(x_seed)) / max(4.0, 0.55 * col_step) + abs(float(cy) - float(y_seed)) / max(4.0, 0.55 * row_step)
        size_pen = abs(ma - float(ma_ref)) / max(2.0, 0.45 * ma_std) + abs(mi - float(mi_ref)) / max(2.0, 0.45 * mi_std)
        score = 1.35 * edge_s - 0.55 * dist_pen - 0.25 * size_pen
        if score > best_score:
            best_score = score
            best = (float(cx), float(cy), float(ma), float(mi), float(score))

    if best is None:
        return None
    if best[4] < -0.12:
        return None
    return best


def _build_complete_grid_ellipses(
    roi_bgr,
    n_rows=20,
    n_choices=5,
    pre_bubbles=None,
    row_centers_override=None,
    col_centers_override=None,
    center_mode="auto",
    fill_threshold=0.50,
):
    """
    Returnează ~n_rows*n_choices elipse, completând lipsurile prin grilă estimată.
    """
    bubbles = pre_bubbles if pre_bubbles is not None else _detect_bubbles_robust_union(roi_bgr)

    use_seed_fallback = False
    if len(bubbles) < 20:
        has_row_override = row_centers_override is not None and len(row_centers_override) == n_rows
        has_col_override = col_centers_override is not None and len(col_centers_override) == n_choices
        if has_row_override and has_col_override:
            roi_h, roi_w = roi_bgr.shape[:2]
            if bubbles:
                ma_vals = np.array([float(max(b["ellipse"][1][0], b["ellipse"][1][1])) for b in bubbles], dtype=np.float32)
                mi_vals = np.array([float(min(b["ellipse"][1][0], b["ellipse"][1][1])) for b in bubbles], dtype=np.float32)
                ma_guess = float(np.median(ma_vals))
                mi_guess = float(np.median(mi_vals))
            else:
                ma_guess = float(np.clip(0.68 * (roi_w / max(1.0, float(n_choices))), 8.0, 34.0))
                mi_guess = float(np.clip(0.68 * (roi_h / max(1.0, float(n_rows))), 8.0, 30.0))

            bubbles = []
            row_ref = np.array(row_centers_override, dtype=np.float32)
            col_ref = np.array(col_centers_override, dtype=np.float32)
            for r in range(n_rows):
                for c in range(n_choices):
                    x = float(col_ref[c])
                    y = float(row_ref[r])
                    bubbles.append(
                        {
                            "cx": int(round(x)),
                            "cy": int(round(y)),
                            "ellipse": ((x, y), (float(ma_guess), float(mi_guess)), 0.0),
                            "synthetic": True,
                            "row": int(r),
                            "col": int(c),
                        }
                    )
            use_seed_fallback = True
        elif len(bubbles) < 8:
            return bubbles

    xs = np.array([b["cx"] for b in bubbles], dtype=np.float32)
    ys = np.array([b["cy"] for b in bubbles], dtype=np.float32)
    row_centers = None
    if row_centers_override is not None and len(row_centers_override) == n_rows:
        row_centers = np.array(row_centers_override, dtype=np.float32)
    else:
        row_centers = _estimate_axis_centers(ys, n_rows, q_low=3, q_high=97)
    col_centers = None
    if col_centers_override is not None and len(col_centers_override) == n_choices:
        col_centers = np.array(col_centers_override, dtype=np.float32)
    else:
        col_centers = _estimate_col_centers_from_rows(bubbles, row_centers, n_choices=n_choices)
        if col_centers is None:
            col_centers = _estimate_axis_centers(xs, n_choices, q_low=5, q_high=95)
    if row_centers is None or col_centers is None:
        return bubbles

    row_centers = np.array(row_centers, dtype=np.float32)
    col_centers = np.array(col_centers, dtype=np.float32)
    if center_mode == "grid_only":
        row_centers = np.linspace(float(row_centers[0]), float(row_centers[-1]), n_rows, dtype=np.float32)
        col_centers = np.linspace(float(col_centers[0]), float(col_centers[-1]), n_choices, dtype=np.float32)
    row_step = float(np.median(np.diff(row_centers))) if n_rows > 1 else 20.0
    col_step = float(np.median(np.diff(col_centers))) if n_choices > 1 else 20.0
    gate_y = max(6.0, 0.60 * row_step)
    gate_x = max(6.0, 0.60 * col_step)

    # Pas 1: ancore brute pentru recalibrarea centrelor pe celule.
    chosen_cell, matches, reliable = _assign_bubbles_to_cells(
        bubbles,
        row_centers,
        col_centers,
        gate_x,
        gate_y,
        n_rows=n_rows,
        n_choices=n_choices,
    )

    row_anchors = np.full((n_rows,), np.nan, dtype=np.float32)
    for r in range(n_rows):
        vals = []
        for c in range(n_choices):
            if not reliable.get((r, c), False):
                continue
            vals.append(float(chosen_cell[(r, c)]["cy"]))
        if len(vals) >= 2:
            row_anchors[r] = float(np.median(np.array(vals, dtype=np.float32)))

    col_anchors = np.full((n_choices,), np.nan, dtype=np.float32)
    for c in range(n_choices):
        vals = []
        for r in range(n_rows):
            if not reliable.get((r, c), False):
                continue
            vals.append(float(chosen_cell[(r, c)]["cx"]))
        if len(vals) >= 6:
            col_anchors[c] = float(np.median(np.array(vals, dtype=np.float32)))

    row_centers = _blend_centers_with_anchors(
        row_centers,
        row_anchors,
        min_points=max(7, int(0.35 * n_rows)),
        blend=0.55,
        min_gap=0.55 * row_step,
    )
    col_centers = _blend_centers_with_anchors(
        col_centers,
        col_anchors,
        min_points=max(3, int(0.6 * n_choices)),
        blend=0.60,
        min_gap=0.55 * col_step,
    )

    if center_mode == "grid_only":
        row_centers = np.linspace(float(row_centers[0]), float(row_centers[-1]), n_rows, dtype=np.float32)
        col_centers = np.linspace(float(col_centers[0]), float(col_centers[-1]), n_choices, dtype=np.float32)

    row_step = float(np.median(np.diff(row_centers))) if n_rows > 1 else row_step
    col_step = float(np.median(np.diff(col_centers))) if n_choices > 1 else col_step
    gate_y = max(6.0, 0.60 * row_step)
    gate_x = max(6.0, 0.60 * col_step)

    # Pas 2: reasignare după recalibrare.
    chosen_cell, matches, reliable = _assign_bubbles_to_cells(
        bubbles,
        row_centers,
        col_centers,
        gate_x,
        gate_y,
        n_rows=n_rows,
        n_choices=n_choices,
    )

    ma_all = np.array([b["ellipse"][1][0] for b in bubbles], dtype=np.float32)
    mi_all = np.array([b["ellipse"][1][1] for b in bubbles], dtype=np.float32)
    ma_med = float(np.median(ma_all))
    mi_med = float(np.median(mi_all))

    # Limităm diametrele ca să evităm intersecțiile între coloane/rânduri.
    ma_max = max(8.0, 0.70 * col_step)
    mi_max = max(8.0, 0.74 * row_step)
    ma_std = min(ma_med, ma_max)
    mi_std = min(mi_med, mi_max)

    # Calibrare dimensiune: model liniar robust pe Y, bazat pe ancore de încredere.
    y_ref = float(np.median(row_centers))
    ma_base = ma_std
    mi_base = mi_std
    ma_slope = 0.0
    mi_slope = 0.0
    yv, mav, miv = [], [], []
    for r in range(n_rows):
        for c in range(n_choices):
            if not reliable.get((r, c), False):
                continue
            ch = chosen_cell.get((r, c))
            if ch is None:
                continue
            ma_r = float(ch["ellipse"][1][0])
            mi_r = float(ch["ellipse"][1][1])
            if ma_r <= 0 or mi_r <= 0:
                continue
            yv.append(float(row_centers[r]))
            mav.append(ma_r)
            miv.append(mi_r)

    if len(yv) >= 10:
        yv = np.array(yv, dtype=np.float32)
        mav = np.array(mav, dtype=np.float32)
        miv = np.array(miv, dtype=np.float32)

        def _mad_keep(v, k=3.0):
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med))) + 1e-6
            keep = np.abs(v - med) <= (k * mad + 0.8)
            return keep

        keep = _mad_keep(mav, 3.2) & _mad_keep(miv, 3.2)
        if int(np.count_nonzero(keep)) >= 8:
            yk = yv[keep]
            mak = mav[keep]
            mik = miv[keep]
            y0 = float(np.median(yk))
            A = np.stack([yk - y0, np.ones_like(yk)], axis=1).astype(np.float32)
            cm, *_ = np.linalg.lstsq(A, mak, rcond=None)
            ci, *_ = np.linalg.lstsq(A, mik, rcond=None)
            ma_slope = float(cm[0])
            mi_slope = float(ci[0])
            ma_base = float(cm[1])
            mi_base = float(ci[1])
            y_ref = y0

            row_span = max(1.0, float(row_centers[-1] - row_centers[0])) if n_rows > 1 else 1.0
            ma_slope_lim = 0.20 * ma_std / row_span
            mi_slope_lim = 0.20 * mi_std / row_span
            ma_slope = float(np.clip(ma_slope, -ma_slope_lim, ma_slope_lim))
            mi_slope = float(np.clip(mi_slope, -mi_slope_lim, mi_slope_lim))
            ma_base = float(np.clip(ma_base, 0.82 * ma_std, 1.18 * ma_std))
            mi_base = float(np.clip(mi_base, 0.82 * mi_std, 1.18 * mi_std))

    fit_thr = max(4.0, 0.28 * min(row_step, col_step))
    bilinear = None
    if center_mode != "no_bilinear" and center_mode != "grid_only" and len(matches) >= 55:
        uniq_rows = len({m[0] for m in matches})
        uniq_cols = len({m[1] for m in matches})
        if uniq_rows >= 14 and uniq_cols >= n_choices:
            bilinear = _fit_bilinear_grid(matches, n_rows=n_rows, n_choices=n_choices, inlier_thr=fit_thr)

    # Validare robustă a fit-ului: dacă deformează prea mult, revenim la grila stabilă.
    if bilinear is not None:
        cx_fit, cy_fit = bilinear
        px = np.zeros((n_rows, n_choices), dtype=np.float32)
        py = np.zeros((n_rows, n_choices), dtype=np.float32)
        for rr in range(n_rows):
            for cc in range(n_choices):
                px[rr, cc] = _eval_bilinear(cx_fit, rr, cc, n_rows, n_choices)
                py[rr, cc] = _eval_bilinear(cy_fit, rr, cc, n_rows, n_choices)

        dx_row = np.diff(px, axis=1)
        dy_col = np.diff(py, axis=0)
        valid = True
        if dx_row.size > 0 and float(np.min(dx_row)) < 4.0:
            valid = False
        if dy_col.size > 0 and float(np.min(dy_col)) < 4.0:
            valid = False

        gap_x = float(np.median(dx_row)) if dx_row.size > 0 else col_step
        gap_y = float(np.median(dy_col)) if dy_col.size > 0 else row_step
        if not (0.50 * col_step <= gap_x <= 1.60 * col_step):
            valid = False
        if not (0.50 * row_step <= gap_y <= 1.60 * row_step):
            valid = False

        seed_x = np.tile(col_centers.reshape(1, -1), (n_rows, 1))
        seed_y = np.tile(row_centers.reshape(-1, 1), (1, n_choices))
        disp = np.sqrt((px - seed_x) ** 2 + (py - seed_y) ** 2)
        med_disp = float(np.median(disp))
        if med_disp > 0.85 * max(col_step, row_step):
            valid = False

        if not valid:
            bilinear = None

    roi_h, roi_w = roi_bgr.shape[:2]
    row_bounds = _centers_to_bounds(row_centers, 0.0, float(roi_h - 1))
    col_bounds = _centers_to_bounds(col_centers, 0.0, float(roi_w - 1))
    gray_raw = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    contrast_local = _contrast_metrics(gray_raw)
    use_clahe_local = _should_use_clahe(contrast_local)

    bw_raw = preprocess_bw_for_bubbles(roi_bgr)
    bw = bw_raw.copy()
    bw_fill_raw = bw_raw.copy()

    gray_blur = cv2.GaussianBlur(gray_raw, (3, 3), 0)
    edge_raw = cv2.Canny(gray_blur, 40, 120)
    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag_raw = cv2.magnitude(gx, gy)
    mag_raw = np.clip(mag_raw, 0.0, 255.0).astype(np.uint8)
    edge = cv2.max(edge_raw, mag_raw)

    if use_clahe_local:
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray_raw)
        roi_clahe_bgr = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)
        bw_clahe = preprocess_bw_for_bubbles(roi_clahe_bgr)
        bw = cv2.bitwise_or(bw, bw_clahe)
        edge_clahe = cv2.Canny(cv2.GaussianBlur(gray_clahe, (3, 3), 0), 34, 108)
        gx_c = cv2.Sobel(gray_clahe, cv2.CV_32F, 1, 0, ksize=3)
        gy_c = cv2.Sobel(gray_clahe, cv2.CV_32F, 0, 1, ksize=3)
        mag_c = cv2.magnitude(gx_c, gy_c)
        mag_c = np.clip(mag_c, 0.0, 255.0).astype(np.uint8)
        edge_clahe = cv2.max(edge_clahe, mag_c)
        edge = cv2.max(edge, edge_clahe)

    seed_x = np.zeros((n_rows, n_choices), dtype=np.float32)
    seed_y = np.zeros((n_rows, n_choices), dtype=np.float32)
    seed_ma = np.zeros((n_rows, n_choices), dtype=np.float32)
    seed_mi = np.zeros((n_rows, n_choices), dtype=np.float32)
    seed_chosen = np.zeros((n_rows, n_choices), dtype=np.uint8)
    local_map = {}

    for r in range(n_rows):
        for c in range(n_choices):
            if bilinear is not None:
                cx_fit, cy_fit = bilinear
                x = _eval_bilinear(cx_fit, r, c, n_rows, n_choices)
                y = _eval_bilinear(cy_fit, r, c, n_rows, n_choices)
            else:
                x = float(col_centers[c])
                y = float(row_centers[r])

            ma = float(np.clip(ma_base + ma_slope * (y - y_ref), 0.78 * ma_std, min(ma_max, 1.22 * ma_std)))
            mi = float(np.clip(mi_base + mi_slope * (y - y_ref), 0.78 * mi_std, min(mi_max, 1.22 * mi_std)))

            chosen = chosen_cell.get((r, c))
            if chosen is not None:
                (_, _), (ma_obs, mi_obs), _ = chosen["ellipse"]
                ma_obs = float(np.clip(ma_obs, 0.82 * ma_std, 1.18 * ma_std))
                mi_obs = float(np.clip(mi_obs, 0.82 * mi_std, 1.18 * mi_std))
                if reliable.get((r, c), False) and not use_seed_fallback:
                    ma = 0.75 * ma + 0.25 * ma_obs
                    mi = 0.75 * mi + 0.25 * mi_obs
                dx = float(chosen["cx"]) - x
                dy = float(chosen["cy"]) - y
                if abs(dx) <= 0.42 * gate_x and abs(dy) <= 0.50 * gate_y and not use_seed_fallback:
                    x = float(chosen["cx"])
                    y = float(chosen["cy"])
                    seed_chosen[r, c] = 1

            seed_x[r, c] = float(x)
            seed_y[r, c] = float(y)
            seed_ma[r, c] = float(ma)
            seed_mi[r, c] = float(mi)

            xl = int(max(0, np.floor(col_bounds[c])))
            xr = int(min(roi_w - 1, np.ceil(col_bounds[c + 1])))
            yt = int(max(0, np.floor(row_bounds[r])))
            yb = int(min(roi_h - 1, np.ceil(row_bounds[r + 1])))
            if xr <= xl:
                xr = min(roi_w - 1, xl + 1)
            if yb <= yt:
                yb = min(roi_h - 1, yt + 1)

            local_map[(r, c)] = _fit_local_ellipse_in_cell(
                bw,
                edge,
                xl,
                yt,
                xr,
                yb,
                x_seed=seed_x[r, c],
                y_seed=seed_y[r, c],
                ma_ref=seed_ma[r, c],
                mi_ref=seed_mi[r, c],
                col_step=col_step,
                row_step=row_step,
                ma_std=ma_std,
                mi_std=mi_std,
            )

    # Reconstrucție de grilă locală pe chenar din centre detectate cu încredere.
    # Ideea: dacă avem suficiente puncte bune în chenar, presupunem grilă aproape
    # echidistantă și completăm toate cele 100 de centre din estimare.
    row_obs = np.full((n_rows,), np.nan, dtype=np.float32)
    col_obs = np.full((n_choices,), np.nan, dtype=np.float32)
    conf_local_count = 0

    for r in range(n_rows):
        vals_y = []
        for c in range(n_choices):
            lf = local_map.get((r, c))
            if lf is not None and float(lf[4]) >= 0.16:
                vals_y.append(float(lf[1]))
                conf_local_count += 1
            elif seed_chosen[r, c] == 1 and reliable.get((r, c), False):
                ch = chosen_cell.get((r, c))
                if ch is not None:
                    vals_y.append(float(ch["cy"]))
        if len(vals_y) >= 2:
            row_obs[r] = float(np.median(np.array(vals_y, dtype=np.float32)))

    for c in range(n_choices):
        vals_x = []
        for r in range(n_rows):
            lf = local_map.get((r, c))
            if lf is not None and float(lf[4]) >= 0.16:
                vals_x.append(float(lf[0]))
            elif seed_chosen[r, c] == 1 and reliable.get((r, c), False):
                ch = chosen_cell.get((r, c))
                if ch is not None:
                    vals_x.append(float(ch["cx"]))
        if len(vals_x) >= 6:
            col_obs[c] = float(np.median(np.array(vals_x, dtype=np.float32)))

    row_lattice, row_valid_n = _axis_from_observations(
        row_obs,
        row_centers,
        min_gap=0.58 * row_step,
        min_valid=max(7, int(0.35 * n_rows)),
    )
    col_lattice, col_valid_n = _axis_from_observations(
        col_obs,
        col_centers,
        min_gap=0.58 * col_step,
        min_valid=max(3, int(0.60 * n_choices)),
    )

    row_shift = np.zeros((n_rows,), dtype=np.float32)
    row_shift_valid = np.zeros((n_rows,), dtype=np.uint8)
    for r in range(n_rows):
        deltas = []
        for c in range(n_choices):
            lf = local_map.get((r, c))
            if lf is None:
                continue
            if float(lf[4]) < 0.16:
                continue
            deltas.append(float(lf[0]) - float(col_lattice[c]))
        if len(deltas) >= 2:
            row_shift[r] = float(np.median(np.array(deltas, dtype=np.float32)))
            row_shift_valid[r] = 1

    if int(np.count_nonzero(row_shift_valid)) >= 3:
        idx = np.where(row_shift_valid > 0)[0].astype(np.float32)
        vals = row_shift[row_shift_valid > 0].astype(np.float32)
        row_shift = np.interp(np.arange(n_rows, dtype=np.float32), idx, vals).astype(np.float32)
        if n_rows >= 3:
            sm = row_shift.copy()
            for i in range(1, n_rows - 1):
                sm[i] = 0.25 * row_shift[i - 1] + 0.50 * row_shift[i] + 0.25 * row_shift[i + 1]
            row_shift = sm
    else:
        row_shift[:] = 0.0
    row_shift = np.clip(row_shift, -0.25 * col_step, 0.25 * col_step)

    lattice_x = np.tile(col_lattice.reshape(1, -1), (n_rows, 1)).astype(np.float32)
    lattice_x += row_shift.reshape(-1, 1)
    lattice_y = np.tile(row_lattice.reshape(-1, 1), (1, n_choices)).astype(np.float32)
    lattice_ready = bool(
        row_valid_n >= max(7, int(0.35 * n_rows))
        and col_valid_n >= max(3, int(0.60 * n_choices))
        and conf_local_count >= max(20, int(0.20 * n_rows * n_choices))
    )

    # Calibrare locală 2D a centrelor (cell-first): învățăm un warp afin robust
    # din ancorele locale detectate, apoi rafinăm fiecare centru în celula sa.
    cell_aff, cell_aff_inliers = _fit_cell_affine(
        seed_x,
        seed_y,
        local_map,
        n_rows=n_rows,
        n_choices=n_choices,
        col_step=col_step,
        row_step=row_step,
    )

    out = []
    for r in range(n_rows):
        for c in range(n_choices):
            if lattice_ready:
                grid_x = float(lattice_x[r, c])
                grid_y = float(lattice_y[r, c])
            else:
                grid_x = float(seed_x[r, c])
                grid_y = float(seed_y[r, c])
            x = float(grid_x)
            y = float(grid_y)
            ma = float(seed_ma[r, c])
            mi = float(seed_mi[r, c])
            ang = 0.0
            local_score = -1.0
            refine_gain = 0.0

            xl = int(max(0, np.floor(col_bounds[c])))
            xr = int(min(roi_w - 1, np.ceil(col_bounds[c + 1])))
            yt = int(max(0, np.floor(row_bounds[r])))
            yb = int(min(roi_h - 1, np.ceil(row_bounds[r + 1])))
            if xr <= xl:
                xr = min(roi_w - 1, xl + 1)
            if yb <= yt:
                yb = min(roi_h - 1, yt + 1)

            if cell_aff is not None and not lattice_ready:
                xa, ya = _apply_affine_point(cell_aff, grid_x, grid_y)
                if abs(xa - grid_x) <= 1.25 * col_step and abs(ya - grid_y) <= 1.25 * row_step:
                    x, y = float(xa), float(ya)

            lf = local_map.get((r, c))
            if lf is not None:
                lx, ly, lma, lmi, lscore = lf
                local_score = float(lscore)
                lma = float(np.clip(lma, 0.80 * ma_std, 1.20 * ma_std))
                lmi = float(np.clip(lmi, 0.80 * mi_std, 1.20 * mi_std))
                ma = 0.65 * ma + 0.35 * lma
                mi = 0.65 * mi + 0.35 * lmi

                dxl = abs(float(lx) - float(x))
                dyl = abs(float(ly) - float(y))
                if lscore >= 0.20 and dxl <= 0.90 * col_step and dyl <= 0.85 * row_step:
                    x, y = float(lx), float(ly)
                elif dxl <= 0.65 * gate_x and dyl <= 0.70 * gate_y:
                    alpha = 0.80 if seed_chosen[r, c] == 0 else 0.70
                    x = alpha * float(lx) + (1.0 - alpha) * float(x)
                    y = alpha * float(ly) + (1.0 - alpha) * float(y)

            # Optimizare locală 2D în interiorul celulei.
            x_ref, y_ref, refine_gain = _refine_center_on_edges(
                edge,
                x,
                y,
                ma,
                mi,
                max_dx=max(2, int(round(0.26 * col_step))),
                max_dy=max(2, int(round(0.24 * row_step))),
                x_min=xl + 1,
                x_max=xr - 1,
                y_min=yt + 1,
                y_max=yb - 1,
                dist_penalty=1.8,
            )
            x, y = float(x_ref), float(y_ref)

            edge_score = _ring_edge_score(edge, int(round(x)), int(round(y)), ma, mi) / 255.0
            shift_from_grid = float(np.sqrt((x - grid_x) ** 2 + (y - grid_y) ** 2))
            confidence = _cell_confidence(edge_score, local_score, refine_gain, shift_from_grid, col_step, row_step)
            filled_ratio = _ellipse_fill_ratio_raw(
                bw_fill_raw,
                x,
                y,
                ma,
                mi,
                angle_deg=ang,
                shrink=0.88,
            )
            filled = bool(filled_ratio >= float(fill_threshold))
            has_local_evidence = bool((lf is not None and local_score >= 0.10) or (refine_gain >= 1.5))
            seed_flag = bool(seed_chosen[r, c] == 1)
            qc_ok, qc_reason = _evaluate_cell_qc(
                edge_score=edge_score,
                seed_chosen=seed_flag,
                has_local_evidence=has_local_evidence,
                shift_from_lattice=shift_from_grid,
                ma=ma,
                mi=mi,
                col_step=col_step,
                row_step=row_step,
            )
            bad_reason = str(qc_reason) if not qc_ok else "ok"

            out.append(
                {
                    "cx": int(round(x)),
                    "cy": int(round(y)),
                    "ellipse": ((x, y), (ma, mi), ang),
                    "synthetic": seed_chosen[r, c] == 0,
                    "row": int(r),
                    "col": int(c),
                    "center": {"x": int(round(x)), "y": int(round(y))},
                    "axes": {"ma": float(ma), "mi": float(mi)},
                    "edge_score": float(edge_score),
                    "refine_gain": float(refine_gain),
                    "shift_from_lattice": float(shift_from_grid),
                    "confidence": float(confidence),
                    "filled_ratio": float(filled_ratio),
                    "filled": bool(filled),
                    "bad_reason": str(bad_reason),
                    "_seed_chosen": bool(seed_flag),
                    "qc_ok": bool(qc_ok),
                    "qc_reason": str(qc_reason),
                    "qc_edge": float(edge_score),
                    "qc_local": float(local_score),
                    "qc_shift": float(shift_from_grid),
                    "qc_refine_gain": float(refine_gain),
                    "qc_affine_inliers": int(cell_aff_inliers),
                    "qc_lattice_ready": bool(lattice_ready),
                    "qc_conf_local_count": int(conf_local_count),
                    "qc_local_clahe": bool(use_clahe_local),
                }
            )

    # Re-ancorare finală pe un model biliniar al centrelor rezultate.
    # Astfel, `shift_from_lattice` reflectă deviația locală reală și putem
    # corecta celulele cu evidență slabă care au fugit spre artefacte.
    if out:
        fit_post = _fit_bilinear_grid(
            [(int(b["row"]), int(b["col"]), float(b["ellipse"][0][0]), float(b["ellipse"][0][1])) for b in out],
            n_rows=n_rows,
            n_choices=n_choices,
            inlier_thr=max(2.6, 0.18 * min(col_step, row_step)),
        )
        if fit_post is not None:
            cx_fit, cy_fit = fit_post
            for b in out:
                r = int(b["row"])
                c = int(b["col"])
                (x0, y0), (ma, mi), ang = b["ellipse"]
                x0 = float(x0)
                y0 = float(y0)
                ma = float(ma)
                mi = float(mi)
                pred_x = _eval_bilinear(cx_fit, r, c, n_rows, n_choices)
                pred_y = _eval_bilinear(cy_fit, r, c, n_rows, n_choices)

                resid = float(np.sqrt((x0 - pred_x) ** 2 + (y0 - pred_y) ** 2))
                local_score = float(b.get("qc_local", -1.0))
                edge_score = float(b.get("qc_edge", 0.0))
                refine_gain = float(b.get("qc_refine_gain", 0.0))
                seed_flag = bool(b.get("_seed_chosen", False))

                weak_evidence = bool(edge_score < 0.08 and local_score < 0.08 and refine_gain < 1.2)
                if weak_evidence and resid > 0.55 * max(col_step, row_step):
                    xl = int(max(0, np.floor(col_bounds[c])))
                    xr = int(min(roi_w - 1, np.ceil(col_bounds[c + 1])))
                    yt = int(max(0, np.floor(row_bounds[r])))
                    yb = int(min(roi_h - 1, np.ceil(row_bounds[r + 1])))
                    if xr <= xl:
                        xr = min(roi_w - 1, xl + 1)
                    if yb <= yt:
                        yb = min(roi_h - 1, yt + 1)

                    xr0, yr0, rg0 = _refine_center_on_edges(
                        edge,
                        pred_x,
                        pred_y,
                        ma,
                        mi,
                        max_dx=max(2, int(round(0.24 * col_step))),
                        max_dy=max(2, int(round(0.22 * row_step))),
                        x_min=xl + 1,
                        x_max=xr - 1,
                        y_min=yt + 1,
                        y_max=yb - 1,
                        dist_penalty=1.4,
                    )
                    x0 = float(xr0)
                    y0 = float(yr0)
                    b["cx"] = int(round(x0))
                    b["cy"] = int(round(y0))
                    b["ellipse"] = ((x0, y0), (ma, mi), ang)
                    b["qc_refine_gain"] = float(max(refine_gain, float(rg0)))
                    refine_gain = float(b["qc_refine_gain"])

                    edge_score = _ring_edge_score(edge, int(round(x0)), int(round(y0)), ma, mi) / 255.0
                    b["qc_edge"] = float(edge_score)
                    b["edge_score"] = float(edge_score)

                resid = float(np.sqrt((float(b["ellipse"][0][0]) - pred_x) ** 2 + (float(b["ellipse"][0][1]) - pred_y) ** 2))
                b["shift_from_lattice"] = float(resid)
                b["qc_shift"] = float(resid)

                has_local_evidence = bool((local_score >= 0.10) or (float(b.get("qc_refine_gain", 0.0)) >= 1.5))
                conf = _cell_confidence(
                    float(b.get("qc_edge", edge_score)),
                    local_score,
                    float(b.get("qc_refine_gain", 0.0)),
                    resid,
                    col_step,
                    row_step,
                )
                b["confidence"] = float(conf)
                b["filled_ratio"] = float(
                    _ellipse_fill_ratio_raw(
                        bw_fill_raw,
                        float(b["ellipse"][0][0]),
                        float(b["ellipse"][0][1]),
                        ma,
                        mi,
                        angle_deg=ang,
                        shrink=0.88,
                    )
                )
                b["filled"] = bool(b["filled_ratio"] >= float(fill_threshold))

                qc_ok, qc_reason = _evaluate_cell_qc(
                    edge_score=float(b.get("qc_edge", edge_score)),
                    seed_chosen=seed_flag,
                    has_local_evidence=has_local_evidence,
                    shift_from_lattice=resid,
                    ma=ma,
                    mi=mi,
                    col_step=col_step,
                    row_step=row_step,
                )
                b["qc_ok"] = bool(qc_ok)
                b["qc_reason"] = str(qc_reason)
                b["bad_reason"] = "ok" if qc_ok else str(qc_reason)

    for b in out:
        if "_seed_chosen" in b:
            del b["_seed_chosen"]

    return out


def _ellipse_poly_points(cx, cy, ma, mi, angle_deg, num_pts=40):
    theta = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
    a = ma / 2.0
    b = mi / 2.0
    ang = np.radians(angle_deg)
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)

    pts = np.zeros((num_pts, 2), dtype=np.float32)
    for k in range(num_pts):
        px = a * np.cos(theta[k])
        py = b * np.sin(theta[k])
        pts[k, 0] = cx + px * cos_a - py * sin_a
        pts[k, 1] = cy + px * sin_a + py * cos_a
    return pts


def _map_points_affine(pts_xy, m_2x3):
    pts_h = np.concatenate([pts_xy, np.ones((pts_xy.shape[0], 1), dtype=np.float32)], axis=1)
    mapped = pts_h @ m_2x3.T
    return mapped.astype(np.int32)


def _cell_to_json(cell):
    (cx, cy), (ma, mi), ang = cell["ellipse"]
    return {
        "row": int(cell.get("row", -1)),
        "col": int(cell.get("col", -1)),
        "center": {"x": float(cx), "y": float(cy)},
        "axes": {"ma": float(ma), "mi": float(mi)},
        "angle": float(ang),
        "edge_score": float(cell.get("edge_score", cell.get("qc_edge", 0.0))),
        "refine_gain": float(cell.get("refine_gain", cell.get("qc_refine_gain", 0.0))),
        "shift_from_lattice": float(cell.get("shift_from_lattice", cell.get("qc_shift", 0.0))),
        "confidence": float(cell.get("confidence", 0.0)),
        "filled_ratio": float(cell.get("filled_ratio", 0.0)),
        "filled": bool(cell.get("filled", False)),
        "synthetic": bool(cell.get("synthetic", False)),
        "bad": bool(not cell.get("qc_ok", False)),
        "bad_reason": str(cell.get("bad_reason", cell.get("qc_reason", "ok"))),
        "qc_ok": bool(cell.get("qc_ok", False)),
        "qc_reason": str(cell.get("qc_reason", "ok")),
        "qc_edge": float(cell.get("qc_edge", 0.0)),
        "qc_local": float(cell.get("qc_local", -1.0)),
        "qc_shift": float(cell.get("qc_shift", 0.0)),
        "qc_refine_gain": float(cell.get("qc_refine_gain", 0.0)),
        "qc_affine_inliers": int(cell.get("qc_affine_inliers", 0)),
        "qc_lattice_ready": bool(cell.get("qc_lattice_ready", False)),
        "qc_conf_local_count": int(cell.get("qc_conf_local_count", 0)),
        "qc_local_clahe": bool(cell.get("qc_local_clahe", False)),
    }


def _detect_variant_from_middle_box(roi_bgr, bubbles, n_choices=5):
    """
    Detectează varianta A-E din pătratul negru aflat deasupra coloanelor A-E.
    Returnează dict cu câmpurile:
      - variant: 'A'..'E' sau None
      - col_idx: 0..4 sau -1
      - method: 'contour' / 'projection' / 'none'
      - marker_bbox: (x, y, w, h) în ROI sau None
      - marker_center: (cx, cy) în ROI sau None
      - strip_bottom: limita inferioară a benzii analizate
      - col_centers: listă cu centrele coloanelor
    """
    if roi_bgr is None or roi_bgr.size == 0 or not bubbles:
        return {
            "variant": None,
            "col_idx": -1,
            "method": "none",
            "marker_bbox": None,
            "marker_center": None,
            "strip_bottom": 0,
            "col_centers": [],
        }

    h, w = roi_bgr.shape[:2]
    letters = "ABCDE"

    col_centers = np.full((n_choices,), np.nan, dtype=np.float32)
    for c in range(n_choices):
        vals = [float(b["cx"]) for b in bubbles if int(b.get("col", -1)) == c]
        if len(vals) >= 4:
            col_centers[c] = float(np.median(np.array(vals, dtype=np.float32)))

    if np.any(np.isnan(col_centers)):
        xs = np.array([float(b["cx"]) for b in bubbles], dtype=np.float32)
        cc = _estimate_axis_centers(xs, n_choices, q_low=5, q_high=95)
        if cc is not None and len(cc) == n_choices:
            cc = np.array(cc, dtype=np.float32)
            for c in range(n_choices):
                if np.isnan(col_centers[c]):
                    col_centers[c] = cc[c]

    if np.any(np.isnan(col_centers)):
        return {
            "variant": None,
            "col_idx": -1,
            "method": "none",
            "marker_bbox": None,
            "marker_center": None,
            "strip_bottom": 0,
            "col_centers": [],
        }

    row0_vals = [float(b["cy"]) for b in bubbles if int(b.get("row", -1)) == 0]
    if len(row0_vals) >= 3:
        row0_y = float(np.median(np.array(row0_vals, dtype=np.float32)))
    else:
        ys = np.array([float(b["cy"]) for b in bubbles], dtype=np.float32)
        rc = _estimate_axis_centers(ys, 20, q_low=3, q_high=97)
        row0_y = float(rc[0]) if rc is not None and len(rc) >= 1 else float(np.percentile(ys, 8))

    row_all = np.array(sorted({int(b.get("row", -1)) for b in bubbles if int(b.get("row", -1)) >= 0}), dtype=np.int32)
    row_step = max(10.0, h / 22.0)
    if row_all.size >= 2:
        y_by_row = []
        for rr in row_all:
            vv = [float(b["cy"]) for b in bubbles if int(b.get("row", -1)) == int(rr)]
            if vv:
                y_by_row.append(float(np.median(np.array(vv, dtype=np.float32))))
        if len(y_by_row) >= 2:
            row_step = float(np.median(np.diff(np.array(y_by_row, dtype=np.float32))))

    strip_bottom = int(np.clip(row0_y - 0.42 * row_step, 8, h - 2))
    strip = roi_bgr[:strip_bottom, :]
    if strip.size == 0 or strip.shape[0] < 6:
        return {
            "variant": None,
            "col_idx": -1,
            "method": "none",
            "marker_bbox": None,
            "marker_center": None,
            "strip_bottom": strip_bottom,
            "col_centers": [float(x) for x in col_centers],
        }

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

    col_step = float(np.median(np.diff(np.sort(col_centers)))) if n_choices > 1 else max(12.0, w / 8.0)
    expected_side = float(np.clip(0.25 * col_step, 6.0, 26.0))
    max_dx = max(6.0, 0.58 * col_step)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1e9
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < 14.0:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        if ww < 3 or hh < 3:
            continue
        if area > 0.08 * float(strip.shape[0] * strip.shape[1]):
            continue
        ar = ww / float(max(1, hh))
        if ar < 0.55 or ar > 1.85:
            continue
        fill = area / float(ww * hh + 1e-6)
        if fill < 0.42:
            continue

        cx = x + ww / 2.0
        col_idx = int(np.argmin(np.abs(col_centers - float(cx))))
        dx = abs(float(cx) - float(col_centers[col_idx]))
        if dx > max_dx:
            continue

        size_pen = abs(float(ww) - expected_side) + abs(float(hh) - expected_side)
        score = 1.8 * fill + 0.005 * area - 0.040 * dx - 0.030 * size_pen
        if score > best_score:
            best_score = score
            best = {
                "col_idx": col_idx,
                "bbox": (int(x), int(y), int(ww), int(hh)),
                "center": (float(cx), float(y + hh / 2.0)),
            }

    if best is not None:
        col_idx = int(best["col_idx"])
        return {
            "variant": letters[col_idx],
            "col_idx": col_idx,
            "method": "contour",
            "marker_bbox": best["bbox"],
            "marker_center": best["center"],
            "strip_bottom": int(strip_bottom),
            "col_centers": [float(x) for x in col_centers],
        }

    # Fallback: profil pe întunecare în banda de sus.
    inv = (255 - gray).astype(np.float32)
    profile = np.mean(inv, axis=0)
    win = max(3, int(round(0.20 * col_step)))
    col_energy = np.zeros((n_choices,), dtype=np.float32)
    for i in range(n_choices):
        cx = int(round(float(col_centers[i])))
        x0 = max(0, cx - win)
        x1 = min(w - 1, cx + win)
        if x1 <= x0:
            continue
        col_energy[i] = float(np.mean(profile[x0 : x1 + 1]))

    idx = int(np.argmax(col_energy))
    med_e = float(np.median(col_energy))
    peak_e = float(col_energy[idx])
    if peak_e < med_e + 4.0:
        return {
            "variant": None,
            "col_idx": -1,
            "method": "none",
            "marker_bbox": None,
            "marker_center": None,
            "strip_bottom": int(strip_bottom),
            "col_centers": [float(x) for x in col_centers],
        }

    return {
        "variant": letters[idx],
        "col_idx": idx,
        "method": "projection",
        "marker_bbox": None,
        "marker_center": (float(col_centers[idx]), float(strip_bottom * 0.5)),
        "strip_bottom": int(strip_bottom),
        "col_centers": [float(x) for x in col_centers],
    }


def main():
    cv2.setRNGSeed(12345)
    input_dir = "extracted_grids_only/"
    output_dir = os.environ.get("PROCESS_OUTPUT_DIR", "processed_grids_output/")
    center_mode = os.environ.get("ELLIPSE_CENTER_MODE", "grid_only").strip().lower()
    if center_mode not in {"auto", "no_bilinear", "grid_only"}:
        center_mode = "auto"
    try:
        fill_threshold = float(os.environ.get("ELLIPSE_FILL_THRESHOLD", "0.50"))
    except Exception:
        fill_threshold = 0.50
    fill_threshold = float(np.clip(fill_threshold, 0.05, 0.95))

    only_names_env = os.environ.get("PROCESS_ONLY_NAMES", "").strip()
    only_names = None
    if only_names_env:
        only_names = {x.strip() for x in only_names_env.split(",") if x.strip()}

    os.makedirs(output_dir, exist_ok=True)

    # Curățăm output-ul anterior ca fiecare test nou să pornească de la zero.
    for entry in os.listdir(output_dir):
        p = os.path.join(output_dir, entry)
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)

    photos = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    photos.sort()
    if only_names:
        photos = [f for f in photos if os.path.splitext(f)[0] in only_names]

    if not photos:
        print(f"Nu s-au gasit imagini in {input_dir}")
        return

    print("==================================================")
    print("Detectare chenare + elipse pe grile extrase")
    print("==================================================")
    print(f"Mod centrare elipse: {center_mode}")
    print(f"Prag fill ratio: {fill_threshold:.2f}")

    colors = [
        (255, 0, 255), (0, 255, 255), (255, 128, 0),
        (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (128, 0, 255), (0, 128, 255), (255, 0, 128), (128, 255, 0),
    ]

    for photo_name in photos:
        name = os.path.splitext(photo_name)[0]
        photo_path = os.path.join(input_dir, photo_name)
        print(f"\n-> Procesare: {name}")

        grid_img = cv2.imread(photo_path)
        if grid_img is None:
            print("  ❌ Eroare la citire.")
            continue

        marker_norm_img, marker_ok, marker_dbg = normalize_by_corner_markers(grid_img)
        base_img = marker_norm_img if marker_ok else grid_img
        if marker_ok:
            print(f"  🎯 Normalizare pe markerii de colț aplicată (thr={marker_dbg.get('threshold')})")
        else:
            print("  ⚠️ Markerii de colț nu au fost detectați robust; continui fără normalizare de colț.")

        h, w = base_img.shape[:2]
        print(f"  Imagine: {w}x{h}")

        shear_cands = find_shear_candidates(base_img, step=0.02, limit=9)
        best_trial = None
        zero_trial = None
        for sh in shear_cands:
            m_try = np.array([[1, sh, -sh * h / 2], [0, 1, 0]], dtype=np.float32)
            u_try = cv2.warpAffine(
                base_img,
                m_try,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            boxes_try, dbg_try = find_layout_10_boxes(u_try)
            score_try = float(dbg_try.get("layout_score", -1e9))
            # Penalizare pe shear mare: în practică foaia deja e aproape rectificată după warp.
            score_adj = score_try - 1.65 * abs(float(sh))
            if abs(float(sh)) > 0.14:
                score_adj -= 3.50 * (abs(float(sh)) - 0.14)

            trial = {
                "score": score_try,
                "score_adj": score_adj,
                "shear": float(sh),
                "img": u_try,
                "boxes": boxes_try,
                "dbg": dbg_try,
            }
            if abs(float(sh)) < 1e-9:
                zero_trial = trial

            if (
                best_trial is None
                or trial["score_adj"] > best_trial["score_adj"] + 1e-6
                or (
                    abs(trial["score_adj"] - best_trial["score_adj"]) <= 1e-6
                    and abs(float(sh)) < abs(best_trial["shear"])
                )
            ):
                best_trial = {
                    "score": trial["score"],
                    "score_adj": trial["score_adj"],
                    "shear": trial["shear"],
                    "img": trial["img"],
                    "boxes": trial["boxes"],
                    "dbg": trial["dbg"],
                }

        # Fallback conservator: dacă shear extrem nu aduce câștig real, preferăm 0.
        if (
            best_trial is not None
            and zero_trial is not None
            and abs(best_trial["shear"]) > 0.14
            and best_trial["score"] < zero_trial["score"] + 0.45
        ):
            best_trial = zero_trial

        if best_trial is None:
            print("  ❌ Nu am putut construi un layout robust.")
            continue

        opt_sh = best_trial["shear"]
        unsheared_img = best_trial["img"]
        boxes = best_trial["boxes"]
        layout_dbg = best_trial["dbg"]

        if abs(opt_sh) > 0.01:
            print(f"  📐 De-forfecare aplicată (shear={opt_sh:.3f})")
        print(
            "  Layout:",
            f"y=[{layout_dbg['y_top']},{layout_dbg['mid_split']},{layout_dbg['y_bot']}],",
            f"x={layout_dbg['borders']},",
            f"score={layout_dbg.get('layout_score', 0.0):.3f}",
        )

        dbg_orig_boxes = base_img.copy()
        dbg_orig_ellipses = base_img.copy()
        dbg_orig_centers = base_img.copy()
        dbg_orig_variant = base_img.copy()
        dbg_unsheared = unsheared_img.copy()
        dbg_unsheared_centers = unsheared_img.copy()

        m_inv = np.array([[1, -opt_sh, opt_sh * h / 2], [0, 1, 0]], dtype=np.float32)

        thick = max(2, int(w / 400))
        font_scale = max(0.5, w / 2000)

        total_marked = 0
        total_found = 0
        qc_bad_boxes = 0
        qc_bad_cells = 0
        sheet_variant = None
        sheet_variant_method = "none"
        variant_dbg = {
            "variant": None,
            "col_idx": -1,
            "method": "none",
            "marker_bbox": None,
            "marker_center": None,
            "strip_bottom": 0,
            "col_centers": [],
        }
        box_json_payload = []

        box_infos = []
        for idx, (x0, y0, x1, y1) in enumerate(boxes):
            band = idx // 5
            col = idx % 5
            q_start = (1 + col * 20) if band == 0 else (101 + col * 20)
            label = f"Q{q_start}-{q_start + 19}"
            color = colors[idx]

            roi_bgr = unsheared_img[y0:y1, x0:x1]
            is_valid = not (
                roi_bgr.size == 0 or roi_bgr.shape[0] < 10 or roi_bgr.shape[1] < 10
            )
            if is_valid:
                raw_bubbles, detect_meta = _detect_bubbles_robust_union(roi_bgr, return_meta=True)
            else:
                raw_bubbles, detect_meta = [], {
                    "used_clahe": False,
                    "contrast": {},
                    "raw_candidates": 0,
                    "clahe_candidates": 0,
                    "merged_candidates": 0,
                }
            box_infos.append(
                {
                    "idx": idx,
                    "band": band,
                    "col": col,
                    "q_start": q_start,
                    "label": label,
                    "color": color,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "roi_bgr": roi_bgr,
                    "valid": is_valid,
                    "raw_bubbles": raw_bubbles,
                    "detect_meta": detect_meta,
                    "bubbles": [],
                }
            )

        # Șablon global de rânduri (normalizat pe înălțimea ROI) învățat din chenarele
        # cu detecție bună. Va fi folosit ca fallback pentru chenare dificile.
        row_template_norm = None
        row_templates = []
        for info in box_infos:
            if not info["valid"] or len(info["raw_bubbles"]) < 90:
                continue
            ys = np.array([float(b["cy"]) for b in info["raw_bubbles"]], dtype=np.float32)
            rc = _estimate_axis_centers(ys, 20, q_low=3, q_high=97)
            if rc is None:
                continue
            roi_h = max(1.0, float(info["roi_bgr"].shape[0] - 1))
            rn = np.clip(np.array(rc, dtype=np.float32) / roi_h, 0.0, 1.0)
            row_templates.append(rn)
        if row_templates:
            row_template_norm = np.median(np.vstack(row_templates), axis=0).astype(np.float32)
            lin = np.linspace(
                float(row_template_norm[0]),
                float(row_template_norm[-1]),
                20,
                dtype=np.float32,
            )
            row_template_norm = (0.40 * row_template_norm + 0.60 * lin).astype(np.float32)

        # Șablon global de coloane A-E (normalizat pe lățimea ROI), învățat din chenare bune.
        col_template_norm = None
        col_templates = []
        for info in box_infos:
            if not info["valid"] or len(info["raw_bubbles"]) < 90:
                continue
            ys = np.array([float(b["cy"]) for b in info["raw_bubbles"]], dtype=np.float32)
            xs = np.array([float(b["cx"]) for b in info["raw_bubbles"]], dtype=np.float32)
            rc = _estimate_axis_centers(ys, 20, q_low=3, q_high=97)
            cc = _estimate_col_centers_from_rows(info["raw_bubbles"], rc, n_choices=5)
            if cc is None:
                cc = _estimate_axis_centers(xs, 5, q_low=5, q_high=95) if xs.size >= 15 else None
            if cc is None:
                continue
            roi_w = max(1.0, float(info["roi_bgr"].shape[1] - 1))
            cn = np.clip(np.array(cc, dtype=np.float32) / roi_w, 0.0, 1.0)
            col_templates.append(cn)
        if col_templates:
            col_template_norm = np.median(np.vstack(col_templates), axis=0).astype(np.float32)
            lin = np.linspace(
                float(col_template_norm[0]),
                float(col_template_norm[-1]),
                5,
                dtype=np.float32,
            )
            col_template_norm = (0.45 * col_template_norm + 0.55 * lin).astype(np.float32)

        # Stabilizare pe bandă: aceleași 20 rânduri pentru toate cele 5 chenare
        # din partea de sus/jos.
        band_rows = {}
        for band in [0, 1]:
            ys = []
            band_heights = []
            band_counts = []
            for info in box_infos:
                if info["band"] != band:
                    continue
                band_heights.append(float(info["roi_bgr"].shape[0]))
                band_counts.append(len(info["raw_bubbles"]))
                ys.extend([float(b["cy"]) for b in info["raw_bubbles"]])
            rc_band = None
            if len(ys) >= 80:
                rc = _estimate_axis_centers(np.array(ys, dtype=np.float32), 20, q_low=3, q_high=97)
                if rc is not None:
                    rc = np.array(rc, dtype=np.float32)
                    lin = np.linspace(float(rc[0]), float(rc[-1]), 20, dtype=np.float32)
                    # Blend cu o scară liniară pentru a evita deformări locale pe imagini imperfecte.
                    rc_band = (0.45 * rc + 0.55 * lin).astype(np.float32)

            rc_tmpl = None
            if row_template_norm is not None and band_heights:
                h_ref = max(1.0, float(np.median(np.array(band_heights, dtype=np.float32)) - 1.0))
                rc_tmpl = (row_template_norm * h_ref).astype(np.float32)

            if rc_band is not None and rc_tmpl is not None and band_counts:
                avg_cnt = float(np.mean(np.array(band_counts, dtype=np.float32)))
                # Dacă detecțiile pe bandă sunt slabe, ne bazăm mai mult pe șablon.
                conf = float(np.clip((avg_cnt - 65.0) / 30.0, 0.0, 1.0))
                band_rows[band] = (conf * rc_band + (1.0 - conf) * rc_tmpl).astype(np.float32)
            elif rc_band is not None:
                band_rows[band] = rc_band
            elif rc_tmpl is not None:
                band_rows[band] = rc_tmpl

        # Override de rânduri per chenar: pornim de la stabilizarea pe bandă, apoi
        # ajustăm cu un shift local robust (mai stabil pe imagini deformate).
        for info in box_infos:
            info["row_override"] = None
            if not info["valid"]:
                continue

            base_rows = band_rows.get(info["band"])
            roi_h = max(1.0, float(info["roi_bgr"].shape[0] - 1.0))
            if base_rows is None and row_template_norm is not None:
                base_rows = (row_template_norm * roi_h).astype(np.float32)
            if base_rows is None:
                continue

            base_rows = np.array(base_rows, dtype=np.float32)
            local_rows = None
            if len(info["raw_bubbles"]) >= 30:
                ys_local = np.array([float(b["cy"]) for b in info["raw_bubbles"]], dtype=np.float32)
                rc_local = _estimate_axis_centers(ys_local, 20, q_low=3, q_high=97)
                if rc_local is not None and len(rc_local) == 20:
                    local_rows = np.array(rc_local, dtype=np.float32)

            if local_rows is None:
                info["row_override"] = base_rows
                continue

            row_step = float(np.median(np.diff(base_rows))) if base_rows.size > 1 else max(10.0, roi_h / 22.0)
            deltas = local_rows - base_rows
            d_mid = deltas[3:17] if deltas.size >= 18 else deltas
            delta = float(np.median(d_mid))
            delta = float(np.clip(delta, -0.35 * row_step, 0.35 * row_step))
            info["row_override"] = (base_rows + delta).astype(np.float32)

        # Override de coloane per chenar: blend între estimarea locală și șablonul global.
        for info in box_infos:
            info["col_override"] = None
            if not info["valid"] or col_template_norm is None:
                continue

            roi_w = max(1.0, float(info["roi_bgr"].shape[1] - 1.0))
            cc_tmpl = (col_template_norm * roi_w).astype(np.float32)

            local_cc = None
            if len(info["raw_bubbles"]) >= 25:
                row_seed = band_rows.get(info["band"])
                local_cc = _estimate_col_centers_from_rows(info["raw_bubbles"], row_seed, n_choices=5)
                if local_cc is None:
                    xs = np.array([float(b["cx"]) for b in info["raw_bubbles"]], dtype=np.float32)
                    if xs.size >= 15:
                        local_cc = _estimate_axis_centers(xs, 5, q_low=5, q_high=95)

            if local_cc is None:
                info["col_override"] = cc_tmpl
                continue

            local_cc = np.array(local_cc, dtype=np.float32)
            cnt = float(len(info["raw_bubbles"]))
            conf = float(np.clip((cnt - 70.0) / 35.0, 0.0, 1.0))
            if local_cc.size >= 5:
                gaps = np.diff(local_cc)
                if gaps.size == 4 and float(np.mean(gaps)) > 1e-3:
                    uniform = 1.0 / (1.0 + float(np.std(gaps)) / float(np.mean(gaps)))
                    conf *= float(np.clip((uniform - 0.60) / 0.35, 0.0, 1.0))
            info["col_override"] = (conf * local_cc + (1.0 - conf) * cc_tmpl).astype(np.float32)

        # =========================
        # Faza 1: ellipse_pipeline
        # =========================
        for info in box_infos:
            idx = info["idx"]
            band = info["band"]
            col = info["col"]
            q_start = info["q_start"]
            label = info["label"]
            color = info["color"]
            x0, y0, x1, y1 = info["x0"], info["y0"], info["x1"], info["y1"]
            roi_bgr = info["roi_bgr"]
            detect_meta = info.get("detect_meta", {})

            # Chenare pe imaginea unsheared
            cv2.rectangle(dbg_unsheared, (x0, y0), (x1, y1), color, thick)
            cv2.putText(
                dbg_unsheared,
                label,
                (x0 + 8, y0 + int(28 * font_scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thick,
            )

            # Colțuri box mapate înapoi pe imaginea originală
            box_pts_u = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            box_pts_o = _map_points_affine(box_pts_u, m_inv).reshape((-1, 1, 2))
            cv2.polylines(dbg_orig_boxes, [box_pts_o], True, color, thick)
            cv2.polylines(dbg_orig_ellipses, [box_pts_o], True, color, thick)
            cv2.polylines(dbg_orig_centers, [box_pts_o], True, color, thick)
            cv2.polylines(dbg_orig_variant, [box_pts_o], True, color, thick)
            cv2.rectangle(dbg_unsheared_centers, (x0, y0), (x1, y1), color, thick)

            tl = box_pts_o[0, 0]
            lbl_pt = (int(tl[0]) + 10, int(tl[1]) + int(30 * font_scale))
            if lbl_pt[1] < 20:
                lbl_pt = (lbl_pt[0], 30)
            cv2.putText(dbg_orig_boxes, label, lbl_pt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick)
            cv2.putText(dbg_orig_ellipses, label, lbl_pt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick)
            cv2.putText(dbg_orig_centers, label, lbl_pt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick)
            cv2.putText(dbg_orig_variant, label, lbl_pt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick)
            cv2.putText(
                dbg_unsheared_centers,
                label,
                (x0 + 8, y0 + int(28 * font_scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thick,
            )

            if not info["valid"]:
                qc_bad_boxes += 1
                qc_bad_cells += 100
                box_json_payload.append(
                    {
                        "idx": int(idx),
                        "band": int(band),
                        "col": int(col),
                        "label": label,
                        "q_start": int(q_start),
                        "rect_unsheared": [int(x0), int(y0), int(x1), int(y1)],
                        "valid": False,
                        "clahe_used": bool(detect_meta.get("used_clahe", False)),
                        "clahe_metrics": detect_meta.get("contrast", {}),
                        "raw_candidates": int(detect_meta.get("raw_candidates", 0)),
                        "clahe_candidates": int(detect_meta.get("clahe_candidates", 0)),
                        "merged_candidates": int(detect_meta.get("merged_candidates", 0)),
                        "ellipse_count": 0,
                        "synthetic_count": 0,
                        "bad_cells": 100,
                        "cells": [],
                    }
                )
                continue

            bubbles = _build_complete_grid_ellipses(
                roi_bgr,
                n_rows=20,
                n_choices=5,
                pre_bubbles=info["raw_bubbles"],
                row_centers_override=info.get("row_override"),
                col_centers_override=info.get("col_override"),
                center_mode=center_mode,
                fill_threshold=fill_threshold,
            )
            info["bubbles"] = bubbles
            total_found += len(bubbles)
            total_marked += int(sum(1 for b in bubbles if b.get("filled", False)))

            synth = int(sum(1 for b in bubbles if b.get("synthetic", False)))
            bad_cells_box = int(sum(1 for b in bubbles if not b.get("qc_ok", False)))
            qc_bad_cells += bad_cells_box
            if bad_cells_box > 0:
                qc_bad_boxes += 1

            # Puncte brute detectate (raw) pentru diagnostic.
            for rb in info["raw_bubbles"]:
                rx_u = float(rb["cx"] + x0)
                ry_u = float(rb["cy"] + y0)
                cv2.circle(dbg_unsheared_centers, (int(round(rx_u)), int(round(ry_u))), 2, (90, 90, 90), -1)
                rp_u = np.array([[rx_u, ry_u]], dtype=np.float32)
                rp_o = _map_points_affine(rp_u, m_inv).reshape((-1, 2))
                rxo, ryo = int(rp_o[0, 0]), int(rp_o[0, 1])
                cv2.circle(dbg_orig_centers, (rxo, ryo), 2, (90, 90, 90), -1)

            # Elipse: desen în spațiul unsheared + mapare poligonală pe original
            for b in bubbles:
                (cx, cy), (ma, mi), ang = b["ellipse"]
                bad_cell = not b.get("qc_ok", False)
                draw_color = (0, 0, 255) if bad_cell else color
                cx_u = float(cx + x0)
                cy_u = float(cy + y0)
                cv2.ellipse(
                    dbg_unsheared,
                    (int(cx_u), int(cy_u)),
                    (max(1, int(ma / 2)), max(1, int(mi / 2))),
                    ang,
                    0,
                    360,
                    draw_color,
                    max(1, thick // 2),
                )

                poly_u = _ellipse_poly_points(cx_u, cy_u, ma, mi, ang, num_pts=32)
                poly_o = _map_points_affine(poly_u, m_inv).reshape((-1, 1, 2))
                cv2.polylines(dbg_orig_ellipses, [poly_o], True, draw_color, max(1, thick // 2))

                # Centre finale folosite pentru mască.
                cx_i, cy_i = int(round(cx_u)), int(round(cy_u))
                cv2.drawMarker(
                    dbg_unsheared_centers,
                    (cx_i, cy_i),
                    draw_color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=8,
                    thickness=max(1, thick // 2),
                )
                cp_u = np.array([[cx_u, cy_u]], dtype=np.float32)
                cp_o = _map_points_affine(cp_u, m_inv).reshape((-1, 2))
                cxo, cyo = int(cp_o[0, 0]), int(cp_o[0, 1])
                cv2.drawMarker(
                    dbg_orig_centers,
                    (cxo, cyo),
                    draw_color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=8,
                    thickness=max(1, thick // 2),
                )
                if bad_cell:
                    cv2.circle(dbg_unsheared_centers, (cx_i, cy_i), 6, (0, 0, 255), 1)
                    cv2.circle(dbg_orig_centers, (cxo, cyo), 6, (0, 0, 255), 1)

            box_json_payload.append(
                {
                    "idx": int(idx),
                    "band": int(band),
                    "col": int(col),
                    "label": label,
                    "q_start": int(q_start),
                    "rect_unsheared": [int(x0), int(y0), int(x1), int(y1)],
                    "valid": True,
                    "clahe_used": bool(detect_meta.get("used_clahe", False)),
                    "clahe_metrics": detect_meta.get("contrast", {}),
                    "raw_candidates": int(detect_meta.get("raw_candidates", 0)),
                    "clahe_candidates": int(detect_meta.get("clahe_candidates", 0)),
                    "merged_candidates": int(detect_meta.get("merged_candidates", 0)),
                    "ellipse_count": int(len(bubbles)),
                    "synthetic_count": int(synth),
                    "bad_cells": int(bad_cells_box),
                    "cells": [_cell_to_json(b) for b in bubbles],
                }
            )

        # =========================
        # Faza 2: variant_pipeline
        # =========================
        mid_info = next((bi for bi in box_infos if bi.get("idx") == 2), None)
        if mid_info is not None and mid_info.get("valid", False) and mid_info.get("bubbles"):
            variant_dbg = _detect_variant_from_middle_box(mid_info["roi_bgr"], mid_info["bubbles"], n_choices=5)
            sheet_variant = variant_dbg.get("variant")
            sheet_variant_method = variant_dbg.get("method", "none")
            strip_bottom = int(variant_dbg.get("strip_bottom", 0))
            col_idx_sel = int(variant_dbg.get("col_idx", -1))
            col_centers_v = variant_dbg.get("col_centers", [])
            x0, y0, x1, y1 = mid_info["x0"], mid_info["y0"], mid_info["x1"], mid_info["y1"]

            if strip_bottom > 0:
                cv2.line(dbg_unsheared_centers, (x0, y0 + strip_bottom), (x1, y0 + strip_bottom), (0, 0, 255), 1)
                seg_u = np.array([[x0, y0 + strip_bottom], [x1, y0 + strip_bottom]], dtype=np.float32)
                seg_o = _map_points_affine(seg_u, m_inv).reshape((-1, 1, 2))
                cv2.polylines(dbg_orig_variant, [seg_o], False, (0, 0, 255), 1)

            for ci, cx_rel in enumerate(col_centers_v):
                xu = int(round(x0 + float(cx_rel)))
                col_col = (160, 160, 160)
                if ci == col_idx_sel:
                    col_col = (0, 255, 255)
                if strip_bottom > 0:
                    cv2.line(dbg_unsheared_centers, (xu, y0), (xu, y0 + strip_bottom), col_col, 1)
                pu = np.array([[float(xu), float(y0 + max(5, strip_bottom // 2))]], dtype=np.float32)
                po = _map_points_affine(pu, m_inv).reshape((-1, 2))
                cv2.circle(dbg_orig_variant, (int(po[0, 0]), int(po[0, 1])), 2, col_col, -1)

            mb = variant_dbg.get("marker_bbox")
            if mb is not None:
                mx, my, mw, mh = mb
                cv2.rectangle(
                    dbg_unsheared_centers,
                    (x0 + int(mx), y0 + int(my)),
                    (x0 + int(mx) + int(mw), y0 + int(my) + int(mh)),
                    (0, 0, 255),
                    2,
                )
                boxm_u = np.array(
                    [
                        [x0 + int(mx), y0 + int(my)],
                        [x0 + int(mx) + int(mw), y0 + int(my)],
                        [x0 + int(mx) + int(mw), y0 + int(my) + int(mh)],
                        [x0 + int(mx), y0 + int(my) + int(mh)],
                    ],
                    dtype=np.float32,
                )
                boxm_o = _map_points_affine(boxm_u, m_inv).reshape((-1, 1, 2))
                cv2.polylines(dbg_orig_variant, [boxm_o], True, (0, 0, 255), 2)

            mc = variant_dbg.get("marker_center")
            if mc is not None:
                mcx, mcy = mc
                p_u = np.array([[x0 + float(mcx), y0 + float(mcy)]], dtype=np.float32)
                p_o = _map_points_affine(p_u, m_inv).reshape((-1, 2))
                cv2.drawMarker(
                    dbg_orig_variant,
                    (int(p_o[0, 0]), int(p_o[0, 1])),
                    (0, 0, 255),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=14,
                    thickness=2,
                )

        image_reject = qc_bad_cells >= 1
        if image_reject:
            print(f"  ⚠️ QC STRICT: {qc_bad_cells} elipse neclare în {qc_bad_boxes} boxuri -> REJECT")
            cv2.putText(
                dbg_orig_ellipses,
                "REJECT - CEL PUTIN 1 ELIPSA NECLARA",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                dbg_orig_centers,
                "REJECT - CEL PUTIN 1 ELIPSA NECLARA",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
        else:
            print("  ✅ QC STRICT: OK (0 elipse neclare)")

        if sheet_variant is not None:
            print(f"  🧭 Varianta detectată: {sheet_variant} (method={sheet_variant_method})")
        else:
            print("  ⚠️ Varianta nu a putut fi detectată robust.")

        variant_txt = f"VARIANTA: {sheet_variant if sheet_variant is not None else 'UNKNOWN'}"
        cv2.putText(
            dbg_orig_variant,
            variant_txt,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255) if sheet_variant is None else (0, 255, 255),
            2,
        )

        print(f"  ✅ Boxuri: {len(boxes)} | Elipse detectate: {total_found} | Celule marcate: {total_marked}")

        box_json_payload.sort(key=lambda x: int(x.get("idx", 999)))
        image_payload = {
            "image": name,
            "center_mode": center_mode,
            "fill_threshold": float(fill_threshold),
            "marker_normalized": bool(marker_ok),
            "marker_debug": marker_dbg,
            "layout": {
                "shear": float(opt_sh),
                "y_top": int(layout_dbg.get("y_top", 0)),
                "mid_split": int(layout_dbg.get("mid_split", h // 2)),
                "y_bot": int(layout_dbg.get("y_bot", h)),
                "borders": [int(v) for v in layout_dbg.get("borders", [])],
                "layout_score": float(layout_dbg.get("layout_score", 0.0)),
                "coverage": float(layout_dbg.get("coverage", 0.0)),
                "uniform": float(layout_dbg.get("uniform", 0.0)),
            },
            "variant": {
                "value": sheet_variant if sheet_variant is not None else "UNKNOWN",
                "method": sheet_variant_method,
                "debug": variant_dbg,
            },
            "summary": {
                "total_boxes": int(len(boxes)),
                "expected_cells": int(len(boxes) * 100),
                "detected_cells": int(total_found),
                "filled_cells": int(total_marked),
                "qc_bad_boxes": int(qc_bad_boxes),
                "qc_bad_cells": int(qc_bad_cells),
                "qc_reject": int(image_reject),
            },
            "boxes": box_json_payload,
        }

        qc_box_lines = []
        for box in image_payload["boxes"]:
            ell_n = int(box.get("ellipse_count", 0))
            synth_n = int(box.get("synthetic_count", 0))
            bad_n = int(box.get("bad_cells", 0))
            frac_s = float(synth_n) / float(max(1, ell_n))
            frac_b = float(bad_n) / float(max(1, ell_n))
            qc_box_lines.append(
                f"box={int(box.get('idx', -1)):02d} {box.get('label', '?')} "
                f"clahe={int(bool(box.get('clahe_used', False)))} "
                f"synthetic={synth_n}/{ell_n} ({frac_s:.2f}) "
                f"bad_cells={bad_n}/{ell_n} ({frac_b:.2f})"
            )

        out_boxed = os.path.join(output_dir, f"{name}_boxed.jpg")
        out_ell = os.path.join(output_dir, f"{name}_ellipses.jpg")
        out_json = os.path.join(output_dir, f"{name}_ellipses.json")
        out_cent = os.path.join(output_dir, f"{name}_centers_debug.jpg")
        out_unsh = os.path.join(output_dir, f"{name}_unsheared_debug.jpg")
        out_unsh_cent = os.path.join(output_dir, f"{name}_unsheared_centers_debug.jpg")
        out_var = os.path.join(output_dir, f"{name}_variant_debug.jpg")
        out_corner = os.path.join(output_dir, f"{name}_corner_markers.jpg")
        out_qc = os.path.join(output_dir, f"{name}_qc.txt")
        corner_dbg = draw_corner_markers_debug(grid_img, marker_dbg)
        cv2.imwrite(out_boxed, dbg_orig_boxes)
        cv2.imwrite(out_ell, dbg_orig_ellipses)
        cv2.imwrite(out_cent, dbg_orig_centers)
        cv2.imwrite(out_unsh, dbg_unsheared)
        cv2.imwrite(out_unsh_cent, dbg_unsheared_centers)
        cv2.imwrite(out_var, dbg_orig_variant)
        cv2.imwrite(out_corner, corner_dbg)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(image_payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        with open(out_qc, "w", encoding="utf-8") as f:
            f.write(f"image={image_payload['image']}\n")
            f.write(f"center_mode={image_payload['center_mode']}\n")
            f.write(f"fill_threshold={image_payload['fill_threshold']:.2f}\n")
            f.write("qc_policy=strict_any_bad_ellipse_reject\n")
            f.write(f"qc_bad_boxes={int(image_payload['summary']['qc_bad_boxes'])}\n")
            f.write(f"qc_bad_cells={int(image_payload['summary']['qc_bad_cells'])}\n")
            f.write(f"qc_reject={int(image_payload['summary']['qc_reject'])}\n")
            f.write(f"variant={image_payload['variant']['value']}\n")
            f.write(f"variant_method={image_payload['variant']['method']}\n")
            for ln in qc_box_lines:
                f.write(ln + "\n")
        print(
            "  🖼️ Salvate:",
            f"{os.path.basename(out_boxed)},",
            f"{os.path.basename(out_ell)},",
            f"{os.path.basename(out_json)},",
            f"{os.path.basename(out_cent)},",
            f"{os.path.basename(out_unsh)},",
            f"{os.path.basename(out_unsh_cent)},",
            f"{os.path.basename(out_var)},",
            f"{os.path.basename(out_corner)},",
            f"{os.path.basename(out_qc)}",
        )


if __name__ == "__main__":
    main()
