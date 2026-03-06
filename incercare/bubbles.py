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
    # gray = cv2.filter2D(gray, -1, kernel2,  anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
    gray = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel_sharpen)
    # gray = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel_detect)
    gray = very_black_to_white_else_black(gray)
    save(f"debug_gray_{index}.png", gray)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        # minDist=max(12, gray.shape[0] // 40),
        minDist=20,
        param1=10,
        param2=40,
        minRadius=20,
        maxRadius=60,
    )

    if circles is None:
        # vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        vis = img.copy()
        save(f"bubbles{index}.png", vis)
        return []

    kept = []
    min_area = 1500
    for x, y, r in np.round(circles[0, :]).astype(int):
        area = math.pi * (int(r) ** 2)
        if area < min_area:
            continue
        kept.append((int(x), int(y), int(r)))

    vis = img.copy()
    for x, y, r in kept:
        cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
        cv2.circle(vis, (x, y), 2, (0, 0, 255), 2)

    save(f"bubbles{index}.png", vis)

    bubbles = []
    for x, y, r in kept:
        bubbles.append({"cx": x, "cy": y, "r": r})

    bubbles.sort(key=lambda b: (b["cy"], b["cx"]))
    return bubbles


def circles_from_mask(mask, min_area=30, max_area=3000):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        (x,y), r = cv2.minEnclosingCircle(c)
        out.append({"center": (x,y), "r": r, "area": area, "cnt": c})
    return out

def very_black_to_white_else_black(img, thr=150):
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
        gray = img

    # 255 unde gray < thr, altfel 0
    mask = (gray < thr).astype(np.uint8) * 255
    return mask
