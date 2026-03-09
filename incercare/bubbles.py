import cv2
import math
import numpy as np
from helpers import save # pyright: ignore


def find_bubbles(img, index):
    """
    Detecteaza simplu buline (cercuri) intr-o imagine.
    Returneaza lista de dict-uri: {"cx": int, "cy": int, "r": int}
    """
    if img is None or img.size == 0:
        return []

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()


    # gray = cv2.GaussianBlur(gray, (3, 3), 2)
    kernel_sharpen = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])

    kernel_detect = np.array([
        [-1,-1,-1],
        [-1,8,-1],
        [-1,-1,-1]
        ])
    # gray = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel_sharpen)
    # gray = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel_detect)
    gray = cv2.filter2D(gray, -1, kernel_sharpen,  anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
    mask = very_black_to_white_else_black(gray)
    save(f"debug_gray_{index}.png", mask)

    # Bulele foarte slab conturate ajung adesea cu arie mai mică după binarizare.
    circles = circles_from_mask(mask, 
                                min_area=400, 
                                max_area=6000)
    circles = suppress_duplicate_circles(circles)

    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    for c in circles:
        x, y = c["center"]
        r = c["r"]
        cv2.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), 2)

    save(f"bubbles{index}.png", vis)

    bubbles = []
    for c in circles:
        x, y = c["center"]
        r = c["r"]
        bubbles.append({"cx": int(x), "cy": int(y), "r": int(r)})

    bubbles.sort(key=lambda b: (b["cy"], b["cx"]))
    return bubbles

def suppress_duplicate_circles(circles, min_center_dist_factor=0.80):
    """
    Pastreaza un singur cerc atunci cand mai multe detectii cad aproape in acelasi loc.
    Alegem cercul cu circularitate/arie mai buna si ignoram restul.
    """
    kept = []
    ordered = sorted(
        circles,
        key=lambda c: (c["circularity"], c["area"]),
        reverse=True,
    )

    for circle in ordered:
        cx, cy = circle["center"]
        r = circle["r"]
        is_duplicate = False

        for kept_circle in kept:
            kx, ky = kept_circle["center"]
            kr = kept_circle["r"]
            dist = math.hypot(cx - kx, cy - ky)
            min_dist = max(r, kr) * min_center_dist_factor
            if dist < min_dist:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(circle)

    kept.sort(key=lambda c: (c["center"][1], c["center"][0]))
    return kept

def circles_from_mask(
    mask,
    min_area=30,
    max_area=3000,
    min_radius=3,
    max_radius=100,
    min_circularity=0.2,
    min_fill_ratio=0.0,
    min_solidity=0,
    max_aspect_ratio_diff=0.3,  # cat de mult poate devia w/h de la 1
):
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        # Cerc minim care acopera contour-ul
        (x, y), r = cv2.minEnclosingCircle(c)
        if r < min_radius or r > max_radius:
            continue

        circle_area = math.pi * r * r
        if circle_area == 0:
            continue

        # 1) Circularity
        circularity = 4 * math.pi * area / (perimeter * perimeter)

        # 2) Cat din cerc e ocupat efectiv de contour
        fill_ratio = area / circle_area

        # 3) Aspect ratio din bounding box
        bx, by, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect_ratio = w / h
        aspect_diff = abs(1.0 - aspect_ratio)

        # 4) Solidity
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area

        if circularity < min_circularity:
            continue
        if fill_ratio < min_fill_ratio:
            continue
        if solidity < min_solidity:
            continue
        if aspect_diff > max_aspect_ratio_diff:
            continue

        out.append({
            "center": (x, y),
            "r": r,
            "area": area,
            "cnt": c,
            "circularity": circularity,
            "fill_ratio": fill_ratio,
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
        })

    return out

def very_black_to_white_else_black(img, thr=160):
    """
    Transformare binară:
      - pixel devine ALB (255) dacă e foarte negru (aproape 0)
      - altfel devine NEGRU (0)

    thr: pragul de "foarte negru" (0..255). Mai mic = mai strict.
    Funcționează pe grayscale sau BGR.
    Returnează o mască grayscale (H,W) cu 0/255.
    """
    if img.ndim == 3:  # BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Netezim puțin înainte de prag ca să nu rupem inelele foarte subțiri.
    gray = cv2.GaussianBlur(gray, (1, 1), 0)

    # 255 unde gray < thr, altfel 0
    mask = (gray < thr).astype(np.uint8) * 255

    # Leagă golurile mici din conturul elipsei fără să umfle agresiv forma.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask
