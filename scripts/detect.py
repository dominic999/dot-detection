import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]      # tl
    rect[2] = pts[np.argmax(s)]      # br
    rect[1] = pts[np.argmin(diff)]   # tr
    rect[3] = pts[np.argmax(diff)]   # bl
    return rect

def four_point_warp(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH)), rect

def find_paper_and_warp(img_bgr, debug=False):
    orig = img_bgr.copy()
    h0, w0 = orig.shape[:2]

    # downscale pt viteză și stabilitate
    scale = 1400.0 / max(h0, w0)
    img = cv2.resize(orig, (int(w0*scale), int(h0*scale))) if scale < 1 else orig.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)

    # threshold pe alb (foaia)
    # OTSU inverse: foaia devine 1 (alb în mask) sau invers, depinde de lumină.
    # Facem binarizare + apoi alegem varianta cu "componentă mare".
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # dacă fundalul devine alb și foaia neagră, inversăm
    # (heuristic: foaia ocupă mult, deci vrem mask cu mult alb)
    if np.mean(th) < 127:
        th = cv2.bitwise_not(th)

    # închidem găurile din foaie (text/grile) ca să devină un blob compact
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
    mask = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    # contururi
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("Nu găsesc contururi. Încearcă lumină mai bună / foaia mai contrastată.")

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    paper_cnt = None
    H, W = mask.shape[:2]
    img_area = H * W

    for c in cnts[:10]:
        area = cv2.contourArea(c)
        if area < 0.20 * img_area:   # foaia trebuie să fie mare
            continue

        # încearcă să aproximezi în 4 colțuri
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            paper_cnt = approx.reshape(4,2).astype(np.float32)
            break

    # fallback: dacă nu obții 4 puncte, folosești minAreaRect (mereu dă 4 colțuri)
    if paper_cnt is None:
        c = cnts[0]
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        paper_cnt = box.astype(np.float32)

    warped_small, rect_small = four_point_warp(img, paper_cnt)

    # scale back
    if scale < 1:
        rect_orig = rect_small / scale
        warped, _ = four_point_warp(orig, rect_orig)
    else:
        warped = warped_small
        rect_orig = rect_small

    if debug:
        dbg = orig.copy()
        cv2.polylines(dbg, [rect_orig.astype(int)], True, (0,255,0), 4)
        return warped, dbg, mask

    return warped, None, mask

# ==== utilizare ====
output_dir = "detect-poze-output"
input_dir = "detect-poze-test"
img = cv2.imread(input_dir + "/IMG_0530.jpg")
warped, dbg, mask = find_paper_and_warp(img, debug=True)

cv2.imwrite(output_dir + "/paper_warped_detect.png", warped)
cv2.imwrite(output_dir + "paper_debug_detect.png", dbg)
cv2.imwrite(output_dir + "paper_mask_detect.png", mask)
print("Salvat: paper_warped.png + paper_debug.png")
