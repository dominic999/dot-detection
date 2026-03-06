import cv2
import itertools
import numpy as np
from edit import warp_crop_from_quad # pyright: ignore[reportImplicitRelativeImport, reportUnknownVariableType]
from helpers import save  # pyright: ignore[reportImplicitRelativeImport, reportUnknownVariableType]


# =========================
# Preprocess
# =========================

def turnGray(img_bgr):
    # grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    save("imagine_gri.png", gray)

    # blur (edge-preserving)
    gray_f = cv2.bilateralFilter(gray, d=21, sigmaColor=50, sigmaSpace=50)
    save("imagine_blurata.png", gray_f)
    return gray_f


def findShapes(img_gray):
    # simple threshold (INV) -> alb = "negru"
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    save("reversed.png", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return img_gray, contours

def order_quad_by_angle(pts4):
    """
    Ordonează 4 puncte în ordine circulară, apoi rotește astfel încât primul să fie TL-ish.
    Return: (4,2) float32 în ordinea TL,TR,BR,BL (consistent pt warp).
    """
    P = np.asarray(pts4, dtype=np.float32)
    c = P.mean(axis=0)

    ang = np.arctan2(P[:,1] - c[1], P[:,0] - c[0])
    order = np.argsort(ang)
    P = P[order]

    # pornește din "top-left": min y, apoi min x
    start = np.lexsort((P[:,0], P[:,1]))[0]
    P = np.roll(P, -start, axis=0)

    # asigură orientarea TL,TR,BR,BL (nu TL,BL,BR,TR)
    # Dacă punctul 1 e mai la stânga decât punctul 3, e inversat -> swap sens
    if P[1,0] < P[3,0]:
        P = np.array([P[0], P[3], P[2], P[1]], dtype=np.float32)

    return P

def poly_area(pts):
    """
    Shoelace formula. pts: (N,2).
    """
    P = np.asarray(pts, dtype=np.float32)
    x = P[:,0]
    y = P[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def quad_is_valid(quad, min_area=5000.0, min_sep=10.0):
    """
    Verifică dacă patrulaterul e "ok": convex + puncte distincte + aria minimă.
    """
    Q = np.asarray(quad, dtype=np.float32)

    # puncte distincte
    for i in range(4):
        for j in range(i+1, 4):
            if np.hypot(*(Q[i]-Q[j])) < min_sep:
                return False

    area = poly_area(Q)
    if area < min_area:
        return False

    # convexitate
    return cv2.isContourConvex(Q.astype(np.int32))

def best_outer_quad_max_area(kept, min_area=5000.0):
    """
    kept: listă de 4 markeri, fiecare cu rect la index 5.
    Return:
      best_quad (TL,TR,BR,BL) float32
      best_choice: (i0,i1,i2,i3) colțurile alese (0..3) pentru fiecare marker
      best_area: aria patrulaterului
    """
    if len(kept) != 4:
        return None, None, 0.0

    # pentru fiecare marker: 4 colțuri (boxPoints)
    boxes = [cv2.boxPoints(k[5]).astype(np.float32) for k in kept]  # listă de (4,2)

    best_quad = None
    best_choice = None
    best_area = -1.0

    # toate combinațiile: 4^4 = 256
    for choice in itertools.product(range(4), repeat=4):
        pts = np.array([boxes[m][choice[m]] for m in range(4)], dtype=np.float32)

        quad = order_quad_by_angle(pts)

        if not quad_is_valid(quad, min_area=min_area):
            continue

        area = poly_area(quad)
        if area > best_area:
            best_area = area
            best_quad = quad
            best_choice = choice

    return best_quad, best_choice, best_area


# =========================
# Pick bars (markeri)
# =========================

def pick_bars(img_gray, contours):
    """
    img_gray: imagine grayscale
    contours: contururile găsite pe masca binară
    Returnează kept = listă cu 4 markeri (dacă reușește), altfel lista filtrată.
    """
    H, W = img_gray.shape[:2]
    bar_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    min_area = 3000
    kept = []

    aspect_min = 3.0
    rect_score_min = 0.90
    mean_gray_max = 120
    max_h_ratio = 0.20

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), angle = rect
        if rw < 1 or rh < 1:
            continue

        long_side = max(rw, rh)
        short_side = min(rw, rh)
        aspect = long_side / (short_side + 1e-6)

        if aspect < aspect_min:
            continue
        if short_side > max_h_ratio * H:
            continue
        if long_side > 0.3 * W:
            continue

        rect_area = rw * rh
        rect_score = float(area) / float(rect_area + 1e-6)
        if rect_score < rect_score_min:
            continue

        # mean gray inside contour (trebuie să fie "negru")
        mask = np.zeros_like(img_gray, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        vals = img_gray[mask == 255]
        if vals.size == 0:
            continue
        mean_gray = float(np.mean(vals))
        if mean_gray > mean_gray_max:
            continue

        kept.append((contour, area, aspect, rect_score, mean_gray, rect))

        # debug draw
        box = cv2.boxPoints(rect).astype(int)
        cv2.polylines(bar_img, [box], True, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(bar_img, f"A={area:.0f}", (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    save("contours.png", bar_img)

    # trebuie exact 4
    if len(kept) != 4:
        print("POZA NU ESTE BUNA! (nu am 4 markeri)")
        return kept

    # arii similare
    areas = np.array([k[1] for k in kept], dtype=np.float32)
    area_ratio = float(areas.max() / (areas.min() + 1e-6))
    if area_ratio > 1.6:
        cv2.putText(bar_img, f"POZA NU ESTE BUNA (area_ratio={area_ratio:.2f})", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        save("contours.png", bar_img)
        print(f"POZA NU ESTE BUNA! (arii prea diferite, ratio={area_ratio:.2f})")
        return kept

    # outer quad (NEW)
    quad, choice, area = best_outer_quad_max_area(kept, min_area=8000.0)

    if quad is None:
        print("Nu am găsit un patrulater valid (max-area). Poza posibil proastă / markeri greșiți.")
        return kept

    bar_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.polylines(bar_img, [quad.astype(int)], True, (255,0,0), 4)
    save("outer_quad.png", bar_img)
    cropped = warp_crop_from_quad(bar_img, quad)   # sau img_gray
    save("cropped.png", cropped)
    print("best choice corners:", choice, "area:", area)
    return kept
