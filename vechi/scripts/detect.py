# import cv2
# import numpy as np
#
# def find_bar_candidates(img_bgr, thr=80):
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5,5), 0)
#
#     bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)[1]
#     bw = cv2.morphologyEx(
#         bw, cv2.MORPH_CLOSE,
#         cv2.getStructuringElement(cv2.MORPH_RECT, (11,11)),
#         iterations=2
#     )
#
#     cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     H, W = bw.shape
#     cand = []
#
#     for c in cnts:
#         area = cv2.contourArea(c)
#         if area < (H*W)*0.0003:
#             continue
#
#         x,y,w,h = cv2.boundingRect(c)
#         ar = w / float(h+1e-6)
#
#         # bară: lată și joasă (ajustezi dacă ai bare “pătrate”)
#         if ar > 4.0 and w > 0.03*W and h < 0.12*H:
#             cx = x + w/2
#             cy = y + h/2
#             cand.append({
#                 "cnt": c, "bbox": (x,y,w,h), "area": area, "ar": ar,
#                 "center": (cx, cy)
#             })
#
#     return cand, bw
#
# def pick_4_markers(cand):
#     if len(cand) < 4:
#         raise RuntimeError("Nu am găsit 4 markeri. Ajustează threshold/filtrele.")
#
#     # folosim centrele
#     pts = np.array([c["center"] for c in cand], dtype=np.float32)
#
#     s = pts[:,0] + pts[:,1]
#     d = pts[:,0] - pts[:,1]
#
#     tl = pts[np.argmin(s)]
#     br = pts[np.argmax(s)]
#     tr = pts[np.argmax(d)]
#     bl = pts[np.argmin(d)]
#
#     return np.array([tl, tr, br, bl], dtype=np.float32)
#
# def align_by_4_markers(img_bgr, out_size=(3508, 2480), thr=80, debug=True):
#     cand, bw = find_bar_candidates(img_bgr, thr=thr)
#     src = pick_4_markers(cand)  # tl,tr,br,bl
#
#     Wout, Hout = out_size
#     margin_x = int(0.08 * Wout)
#     margin_y = int(0.10 * Hout)
#
#     dst = np.array([
#         [margin_x, margin_y],
#         [Wout - margin_x, margin_y],
#         [Wout - margin_x, Hout - margin_y],
#         [margin_x, Hout - margin_y],
#     ], dtype=np.float32)
#
#     H = cv2.getPerspectiveTransform(src, dst)
#     aligned = cv2.warpPerspective(img_bgr, H, (Wout, Hout))
#
#     dbg = None
#     if debug:
#         dbg = img_bgr.copy()
#         # desenează candidații
#         for c in cand:
#             x,y,w,h = c["bbox"]
#             cv2.rectangle(dbg, (x,y), (x+w,y+h), (0,255,255), 2)
#         # desenează cei 4 aleși
#         for i, (x,y) in enumerate(src):
#             cv2.circle(dbg, (int(x),int(y)), 10, (0,0,255), -1)
#             cv2.putText(dbg, ["TL","TR","BR","BL"][i], (int(x)+10,int(y)-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
#     return aligned, dbg, bw
#
# input_dir = "/home/domi/soft31/detect-poze-test/"
# output_dir = "/home/domi/soft31/detect-poze-output/"
# img = cv2.imread(input_dir + "IMG_0530.jpg")
# aligned, dbg, bw = align_by_4_markers(img, out_size=(3508,2480), thr=80, debug=True)
#
# cv2.imwrite(output_dir + "markers_bw.png", bw)
# cv2.imwrite(output_dir + "markers_debug.png", dbg)
# cv2.imwrite(output_dir + "aligned.png", aligned)

import cv2
import numpy as np

def highlight_very_black_rectangles(
    img_bgr,
    thr=120,                 # prag inițial pt “întunecat” (ajustezi 90–150)
    min_area_ratio=0.0003,   # ignoră pete mici
    close_kernel=11,
    close_iter=2,
    black_ratio_min=0.95     # >= 95% negre în interiorul bbox
):
    """
    Detectează regiuni foarte negre (>=95% pixeli "întunecați" în bbox) și le colorează.
    Returnează: out, bw_dark, rects
      rect = (x,y,w,h, black_ratio)
    """

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1) mask pt pixeli întunecați
    bw_dark = cv2.threshold(gray_blur, thr, 255, cv2.THRESH_BINARY_INV)[1]

    # 2) close ca să unească dreptunghiurile
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
    bw_dark = cv2.morphologyEx(bw_dark, cv2.MORPH_CLOSE, k, iterations=close_iter)

    cnts, _ = cv2.findContours(bw_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = gray.shape
    min_area = (H * W) * min_area_ratio

    out = img_bgr.copy()
    rects = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 8:
            continue

        # 3) calculează cât de "negru" e bbox-ul, folosind gray (NU bw_dark ca să fie corect)
        patch = gray[y:y+h, x:x+w]
        if patch.size == 0:
            continue

        # “negru” = pixeli sub prag
        black_ratio = float(np.mean(patch < thr))

        if black_ratio < black_ratio_min:
            continue

        rects.append((x, y, w, h, black_ratio))

        # desen: chenar + fill transparent
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 3)
        overlay = out.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), -1)
        out = cv2.addWeighted(overlay, 0.20, out, 0.80, 0)

        cv2.putText(out, f"{black_ratio:.2f}", (x, max(0, y-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    rects.sort(key=lambda r: (r[1], r[0]))
    return out, bw_dark, rects

input_dir = "/home/domi/soft31/detect-poze-test/"
output_dir = "/home/domi/soft31/detect-poze-output/"
# ====== Exemplu de utilizare ======
img = cv2.imread(input_dir + "IMG_0530.jpg")
out, bw, rects = highlight_very_black_rectangles(img, thr=90)
cv2.imwrite(output_dir + "black_rects.png", out)
cv2.imwrite(output_dir + "black_mask.png", bw)
print("Rectangles:", rects)
