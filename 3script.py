# from pdf2image import convert_from_path
# import cv2
# import numpy as np
#
# # =========================
# # Utils
# # =========================
#
# def pil_to_bgr(pil_img):
#     rgb = np.array(pil_img)  # RGB
#     return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#
# def binarize_for_layout(img_bgr):
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     bw = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
#     return bw
#
# def smooth_1d(arr, k=51):
#     k = k if k % 2 == 1 else k + 1
#     return cv2.GaussianBlur(arr.astype(np.float32).reshape(-1, 1), (1, k), 0).ravel()
#
# def find_valleys_between_peaks(proj, n_sections):
#     p = proj.copy().astype(np.float32)
#     p = (p - p.min()) / (p.max() - p.min() + 1e-6)
#
#     mask = p > 0.25
#     idx = np.where(mask)[0]
#     if len(idx) == 0:
#         raise RuntimeError("Nu găsesc conținut în proiecție. Prag prea mare sau binarizare slabă.")
#
#     start = int(idx[0])
#     end = int(idx[-1])
#
#     cuts = [start]
#     for i in range(1, n_sections):
#         target = int(start + (end - start) * i / n_sections)
#
#         win = max(20, (end - start) // (n_sections * 8))
#         a = max(start, target - win)
#         b = min(end, target + win)
#
#         valley = a + int(np.argmin(p[a:b + 1]))
#         cuts.append(valley)
#
#     cuts.append(end)
#     return cuts
#
# def find_best_horizontal_cut(proj_y, H, mid_frac=0.52, band=0.18):
#     p = proj_y.astype(np.float32)
#     p = (p - p.min()) / (p.max() - p.min() + 1e-6)
#
#     mid = int(H * mid_frac)
#     a = max(0, int(mid - H * band))
#     b = min(H - 1, int(mid + H * band))
#
#     thresh = np.percentile(p[a:b + 1], 15)  # 15% cele mai mici
#     low = p[a:b + 1] <= thresh
#
#     best_len = 0
#     best_center = mid
#     i = 0
#     while i < len(low):
#         if not low[i]:
#             i += 1
#             continue
#         j = i
#         while j < len(low) and low[j]:
#             j += 1
#         run_len = j - i
#         if run_len > best_len:
#             best_len = run_len
#             best_center = a + (i + j) // 2
#         i = j
#
#     return best_center
#
# # =========================
# # Box detection + debug info
# # =========================
#
# def get_10_boxes_fixedBands(page2_bgr):
#     """
#     2 benzi pe Y definite de cut_top/cut_mid_bot/cut_mid_top/cut_bot
#     + 5 coloane pe X (xb) => 10 box-uri.
#     """
#     H, W = page2_bgr.shape[:2]
#
#     # 1) crop partea dreaptă (zona grilei)
#     x_off = int(W * 0.50)
#     right = page2_bgr[:, x_off:W-80]
#     bw = binarize_for_layout(right)
#
#     # 2) taie sus/jos (fara bare)
#     cut_top = int(right.shape[0] * 0.08)
#     cut_bot = int(right.shape[0] * 0.90)
#
#     # 3) definește “gap”-ul central (zona pe care o sari)
#     mid = int((cut_top + cut_bot) / 2)
#     cut_mid_bot = mid - 40   # limita de jos a benzii de sus
#     cut_mid_top = mid + 40   # limita de sus a benzii de jos
#
#     # sanity (ca să nu iasă invers)
#     cut_mid_bot = max(cut_top + 10, cut_mid_bot)
#     cut_mid_top = min(cut_bot - 10, cut_mid_top)
#
#     # 4) calculează xb (5 coloane) folosind TOT intervalul [cut_top:cut_bot]
#     bw_mid = bw[cut_top:cut_bot, :]
#     proj_x = smooth_1d(bw_mid.sum(axis=0), 81)
#     xb = find_valleys_between_peaks(proj_x, 5)  # 6 valori
#
#     # 5) construiește 10 box-uri:
#     #    - sus: cut_top .. cut_mid_bot
#     #    - jos: cut_mid_top .. cut_bot
#     boxes = []
#     bands = [
#         ("TOP", cut_top, cut_mid_bot),
#         ("BOT", cut_mid_top, cut_bot),
#     ]
#
#     for band_name, yA, yB in bands:
#         # padding mic pe Y, dar NU peste gap
#         y0 = max(0, yA - 10)
#         y1 = min(right.shape[0] - 1, yB + 10)
#
#         for col in range(5):
#             x0 = max(0, xb[col] - 15)
#             x1 = min(right.shape[1] - 1, xb[col + 1] + 15)
#
#             # convert în coordonate full-image
#             boxes.append((x_off + x0, y0, x_off + x1, y1))
#
#     dbg = {
#         "x_off": x_off,
#         "right_shape": right.shape,
#         "cut_top": cut_top,
#         "cut_mid_bot": cut_mid_bot,
#         "cut_mid_top": cut_mid_top,
#         "cut_bot": cut_bot,
#         "xb": xb,
#     }
#     return boxes, dbg
#
# # =========================
# # Detectare buline
# # =========================
#
# def detect_bubbles(roi_bgr, debug_draw=False):
#     """
#     Returnează o listă de bubbles:
#       bubble = {"cx":..., "cy":..., "w":..., "h":..., "area":..., "bbox":(x,y,w,h)}
#     coordonate RELATIVE la roi.
#     """
#     gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5,5), 0)
#
#     # binarizare (invers) - bulele devin "albe" (255)
#     bw = cv2.adaptiveThreshold(gray, 255,
#                                cv2.ADAPTIVE_THRESH_MEAN_C,
#                                cv2.THRESH_BINARY_INV,
#                                35, 10)
#
#     # mic cleanup: unește contururi sparte
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#     bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
#     bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
#
#     # contururi
#     contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     H, W = bw.shape
#     bubbles = []
#
#     # praguri (le poți ajusta 1-2 minute)
#     # min_area = (H * W) * 0.00015
#     min_area = (H * W) * 0.002
#     max_area = (H * W) * 0.01
#
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area < min_area or area > max_area:
#             continue
#
#         x, y, w, h = cv2.boundingRect(cnt)
#
#         # filtre rapide: dimensiune + aspect
#         if w < 8 or h < 8:
#             continue
#         ar = w / float(h)
#         if ar < 0.6 or ar > 1.6:
#             continue
#
#         # circularitate (aproape de cerc/elipsă)
#         per = cv2.arcLength(cnt, True)
#         if per <= 0:
#             continue
#         circ = 4 * np.pi * area / (per * per)
#         if circ < 0.35:  # elipsele sunt ok ~0.5-0.9; textul e mai jos
#             continue
#
#         # centru din momente
#         M = cv2.moments(cnt)
#         if M["m00"] == 0:
#             continue
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#
#         bubbles.append({
#             "cx": cx, "cy": cy,
#             "w": w, "h": h,
#             "area": area,
#             "bbox": (x, y, w, h),
#             "circ": circ
#         })
#
#     # opțional: sortare top-to-bottom, left-to-right
#     bubbles.sort(key=lambda b: (b["cy"], b["cx"]))
#
#     if debug_draw:
#         dbg = roi_bgr.copy()
#         for b in bubbles:
#             x, y, w, h = b["bbox"]
#             cv2.rectangle(dbg, (x,y), (x+w, y+h), (0,255,0), 2)
#             cv2.circle(dbg, (b["cx"], b["cy"]), 3, (0,0,255), -1)
#         return bubbles, bw, dbg
#
#     return bubbles, bw, None
#
#
# # =========================
# # OMR read (n-a fost modificat)
# # =========================
#
# def extract_answers_box(roi_bgr, n_questions=20, n_choices=5,
#                         top_margin=0.10, bottom_margin=0.06,
#                         left_margin=0.12, right_margin=0.08,
#                         bubble_pad=0.18, fill_thresh=0.18, ambiguous_delta=0.06):
#     gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     bw = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#         cv2.THRESH_BINARY_INV, 35, 10
#     )
#
#     h, w = bw.shape
#     y0 = int(h * top_margin)
#     y1 = int(h * (1 - bottom_margin))
#     x0 = int(w * left_margin)
#     x1 = int(w * (1 - right_margin))
#
#     grid = bw[y0:y1, x0:x1]
#     gh, gw = grid.shape
#     row_h = gh / n_questions
#     col_w = gw / n_choices
#
#     answers = []
#     for i in range(n_questions):
#         ratios = []
#         for j in range(n_choices):
#             cy0 = int(i * row_h); cy1 = int((i + 1) * row_h)
#             cx0 = int(j * col_w); cx1 = int((j + 1) * col_w)
#             cell = grid[cy0:cy1, cx0:cx1]
#
#             pad_y = int(cell.shape[0] * bubble_pad)
#             pad_x = int(cell.shape[1] * bubble_pad)
#             inner = cell[pad_y:cell.shape[0] - pad_y, pad_x:cell.shape[1] - pad_x]
#
#             if inner.size == 0:
#                 ratios.append(0.0)
#                 continue
#
#             fill_ratio = float(np.count_nonzero(inner)) / float(inner.size)
#             ratios.append(fill_ratio)
#
#         best = int(np.argmax(ratios))
#         best_val = ratios[best]
#         second = sorted(ratios, reverse=True)[1]
#
#         if best_val < fill_thresh:
#             answers.append(None)
#         elif (best_val - second) < ambiguous_delta:
#             answers.append("AMBIG")
#         else:
#             answers.append("ABCDE"[best])
#     return answers
#
# # =========================
# # FLOW
# # =========================
#
# pages = convert_from_path("GrileRezidentiat/farm_C_editat.pdf", 300)
# page_2 = pages[1]
# img = pil_to_bgr(page_2)
#
# boxes, dbg = get_10_boxes_fixedBands(img)
#
# # Citire raspunsuri
# all_answers = {}
# for idx, (x0, y0, x1, y1) in enumerate(boxes):
#     roi = img[y0:y1, x0:x1]
#     answers_20 = extract_answers_box(roi, n_questions=20)
#
#     col = idx % 5
#     block = idx // 5
#     q_start = (1 + col * 20) if block == 0 else (101 + col * 20)
#
#     for i, ans in enumerate(answers_20):
#         all_answers[q_start + i] = ans
#
# print("Exemplu 1..25:", [all_answers[i] for i in range(1, 26)])
# print("Exemplu 101..125:", [all_answers[i] for i in range(101, 126)])
#
# # =========================
# # DEBUG DRAW
# # =========================
#
# debug = img.copy()
# H, W = debug.shape[:2]
# x_off = dbg["x_off"]
# cut_top = dbg["cut_top"]
# cut_bot = dbg["cut_bot"]
# xb = dbg["xb"]
# x_off = dbg["x_off"],
# right_shape = dbg["right_shape"],
# cut_top = dbg["cut_top"],
# cut_mid_bot = dbg["cut_mid_bot"],
# cut_mid_top = dbg["cut_mid_top"],
# cut_bot = dbg["cut_bot"],
# xb = dbg["xb"],
# debug = img.copy()
# H, W = debug.shape[:2]
#
# x_off = dbg["x_off"]
# right_w = dbg["right_shape"][1]
# cut_top = dbg["cut_top"]
# cut_mid_bot = dbg["cut_mid_bot"]
# cut_mid_top = dbg["cut_mid_top"]
# cut_bot = dbg["cut_bot"]
# xb = dbg["xb"]
#
# # dreptunghi bw_mid (zona de unde ai luat proiecțiile)
# # cv2.rectangle(debug, (x_off, cut_top), (x_off + right_w - 1, cut_bot - 1), (255, 0, 0), 2)  # albastru
# #
# # # banda de sus (TOP)
# # cv2.rectangle(debug, (x_off, cut_top), (x_off + right_w - 1, cut_mid_bot - 1), (0, 0, 255), 2)  # roșu
# #
# # # gap-ul (zona sărită)
# # cv2.rectangle(debug, (x_off, cut_mid_bot), (x_off + right_w - 1, cut_mid_top), (0, 255, 255), 2)  # galben
# #
# # # banda de jos (BOT)
# # cv2.rectangle(debug, (x_off, cut_mid_top), (x_off + right_w - 1, cut_bot - 1), (0, 255, 0), 2)  # verde
# #
# # # liniile xb
# # for i, x in enumerate(xb):
# #     X = x_off + int(x)
# #     cv2.line(debug, (X, 0), (X, H - 1), (255, 255, 0), 2)  # cyan-ish
# #
# # box-urile finale
# for i, (x0,y0,x1,y1) in enumerate(boxes):
#     cv2.rectangle(debug, (x0,y0), (x1,y1), (255, 0, 255), 2)  # mov
#     cv2.putText(debug, str(i), (x0+10, y0+40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
#
# cv2.imwrite("debug_boxes.png", debug)
# print("Salvat: debug_boxes.png")
#
# x0,y0,x1,y1 = boxes[0]
# roi = img[y0:y1, x0:x1]
#
# bubbles, bw, dbg = detect_bubbles(roi, debug_draw=True)
#
# print("Buline detectate:", len(bubbles))
# cv2.imwrite("dbg_bubbles_roi.png", dbg)
# cv2.imwrite("dbg_binarized_roi.png", bw)


##############
# Aici incep #
##############
from pdf2image import convert_from_path
import cv2
import numpy as np

# =========================
# Utils
# =========================

def pil_to_bgr(pil_img):
    rgb = np.array(pil_img)  # RGB
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def binarize_for_layout(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
    return bw

def smooth_1d(arr, k=51):
    k = k if k % 2 == 1 else k + 1
    return cv2.GaussianBlur(arr.astype(np.float32).reshape(-1, 1), (1, k), 0).ravel()

def find_valleys_between_peaks(proj, n_sections):
    p = proj.copy().astype(np.float32)
    p = (p - p.min()) / (p.max() - p.min() + 1e-6)

    mask = p > 0.25
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise RuntimeError("Nu găsesc conținut în proiecție. Prag prea mare sau binarizare slabă.")

    start = int(idx[0])
    end = int(idx[-1])

    cuts = [start]
    for i in range(1, n_sections):
        target = int(start + (end - start) * i / n_sections)

        win = max(20, (end - start) // (n_sections * 8))
        a = max(start, target - win)
        b = min(end, target + win)

        valley = a + int(np.argmin(p[a:b + 1]))
        cuts.append(valley)

    cuts.append(end)
    return cuts

# =========================
# Box detection
# =========================

def get_10_boxes_fixedBands(page2_bgr):
    """
    2 benzi pe Y (sus/jos) + 5 coloane pe X => 10 box-uri.
    """
    H, W = page2_bgr.shape[:2]

    x_off = int(W * 0.50)
    right = page2_bgr[:, x_off:W-80]
    bw = binarize_for_layout(right)

    cut_top = int(right.shape[0] * 0.08)
    cut_bot = int(right.shape[0] * 0.90)

    mid = int((cut_top + cut_bot) / 2)
    cut_mid_bot = mid - 40
    cut_mid_top = mid + 40

    cut_mid_bot = max(cut_top + 10, cut_mid_bot)
    cut_mid_top = min(cut_bot - 10, cut_mid_top)

    bw_mid = bw[cut_top:cut_bot, :]
    proj_x = smooth_1d(bw_mid.sum(axis=0), 81)
    xb = find_valleys_between_peaks(proj_x, 5)  # 6

    boxes = []
    bands = [
        ("TOP", cut_top, cut_mid_bot),
        ("BOT", cut_mid_top, cut_bot),
    ]

    for band_name, yA, yB in bands:
        y0 = max(0, yA - 10)
        y1 = min(right.shape[0] - 1, yB + 10)

        for col in range(5):
            x0 = max(0, xb[col] + 90)
            x1 = min(right.shape[1] - 1, xb[col + 1]-20)
            boxes.append((x_off + x0, y0, x_off + x1, y1))

    dbg = {
        "x_off": x_off,
        "right_shape": right.shape,
        "cut_top": cut_top,
        "cut_mid_bot": cut_mid_bot,
        "cut_mid_top": cut_mid_top,
        "cut_bot": cut_bot,
        "xb": xb,
    }
    return boxes, dbg

# =========================
# Detectare buline + umplere
# =========================

def preprocess_bw_for_bubbles(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bw

def detect_bubbles(roi_bgr):
    """
    Detectează contururile de bulă în ROI.
    Returnează: bubbles(list), bw(binarizat)
    bubble: {"cx","cy","bbox","cnt","area","circ"}
    """
    bw = preprocess_bw_for_bubbles(roi_bgr)
    H, W = bw.shape
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    img_area = H * W

    # Praguri mai sănătoase (al tău min_area era prea mare și îți pierdea buline)
    # min_area = img_area * 0.00012
    min_area = img_area * 0.0015
    max_area = img_area * 0.01

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 8 or h < 8:
            continue

        ar = w / float(h)
        if ar < 0.6 or ar > 1.7:
            continue

        per = cv2.arcLength(cnt, True)
        if per <= 0:
            continue
        circ = 4 * np.pi * area / (per * per)
        if circ < 0.28:  # lasă și elipse mai “stricate”
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        bubbles.append({
            "cx": cx, "cy": cy,
            "bbox": (x, y, w, h),
            "cnt": cnt,
            "area": area,
            "circ": float(circ)
        })

    return bubbles, bw

def bubble_fill_ratio(bw, bubble, shrink=0.70):
    """
    Calculează % pixeli negri (activi) în interiorul bulei.
    Folosește o mască eliptică pe bounding box, ușor micșorată.
    """
    x, y, w, h = bubble["bbox"]

    # protecție la margini
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(bw.shape[1], x + w); y1 = min(bw.shape[0], y + h)
    patch = bw[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0

    ph, pw = patch.shape
    mask = np.zeros((ph, pw), dtype=np.uint8)

    cx = pw // 2
    cy = ph // 2
    ax = max(1, int((pw * shrink) / 2))
    ay = max(1, int((ph * shrink) / 2))

    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

    inside = patch[mask == 255]
    if inside.size == 0:
        return 0.0

    # bw e invertit: pixeli activi = 255
    return float(np.count_nonzero(inside)) / float(inside.size)

def group_bubbles_into_20_rows(bubbles, n_rows=20):
    """
    Grupează bulinele în 20 rânduri folosind k-means pe coordonata cy.
    Returnează: listă de rânduri, fiecare rând = listă bubbles sortate pe cx.
    """
    if len(bubbles) < n_rows * 3:
        # prea puține => ceva e greșit (threshold/ROI)
        return []

    ys = np.array([b["cy"] for b in bubbles], dtype=np.float32).reshape(-1, 1)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)
    _, labels, centers = cv2.kmeans(ys, n_rows, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    centers = centers.flatten()
    order = np.argsort(centers)  # top -> bottom

    # map label original -> row index 0..n_rows-1 (ordonat)
    label_to_row = {int(order[i]): i for i in range(n_rows)}

    rows = [[] for _ in range(n_rows)]
    for b, lab in zip(bubbles, labels.flatten()):
        r = label_to_row[int(lab)]
        rows[r].append(b)

    # în fiecare rând păstrăm cele mai “probabile” 5 (după x sort), dacă sunt mai multe
    out = []
    for r in rows:
        r.sort(key=lambda b: b["cx"])
        # uneori intră zgomot; păstrăm 5 cele mai “centrale” dacă sunt >5
        if len(r) > 5:
            # păstrăm 5 cu bbox asemănător medianei (filtru simplu)
            widths = np.array([b["bbox"][2] for b in r])
            medw = np.median(widths)
            r = sorted(r, key=lambda b: abs(b["bbox"][2] - medw))[:5]
            r.sort(key=lambda b: b["cx"])
        out.append(r)

    return out

def read_answers_from_bubbles(roi_bgr, fill_threshold=0.95, n_rows=20, n_choices=5, debug_path=None):
    """
    Returnează:
      answers: listă lungime 20, fiecare element = listă de litere active (ex ['B'] sau ['A','D'] sau [])
    """
    bubbles, bw = detect_bubbles(roi_bgr)
    rows = group_bubbles_into_20_rows(bubbles, n_rows=n_rows)

    answers = []
    dbg_img = roi_bgr.copy() if debug_path else None

    if not rows:
        return [], bw, dbg_img

    for r_idx, row in enumerate(rows):
        # dacă nu avem 5 exact, tot încercăm: sortăm și luăm până la 5
        row = sorted(row, key=lambda b: b["cx"])[:n_choices]

        active = []
        for j, b in enumerate(row):
            ratio = bubble_fill_ratio(bw, b, shrink=0.70)
            if ratio >= fill_threshold:
                active.append("ABCDE"[j])

            if dbg_img is not None:
                x, y, w, h = b["bbox"]
                color = (0, 255, 0) if ratio >= fill_threshold else (0, 0, 255)
                cv2.rectangle(dbg_img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(dbg_img, f"{ratio:.2f}", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        answers.append(active)

    if debug_path:
        cv2.imwrite(debug_path, dbg_img)

    return answers, bw, dbg_img

# =========================
# FLOW
# =========================

pages = convert_from_path("GrileRezidentiat/farm_C_editat.pdf", 300)
page_2 = pages[1]
img = pil_to_bgr(page_2)

boxes, dbg = get_10_boxes_fixedBands(img)

# DEBUG: desenează box-urile
debug_boxes = img.copy()
for i, (x0, y0, x1, y1) in enumerate(boxes):
    cv2.rectangle(debug_boxes, (x0, y0), (x1, y1), (255, 0, 255), 2)
    cv2.putText(debug_boxes, str(i), (x0+10, y0+40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
cv2.imwrite("debug_boxes.png", debug_boxes)
print("Salvat: debug_boxes.png")

# =========================
# Citește răspunsuri pe baza bulinelor (prag 30%)
# =========================

all_answers = {}  # q -> list de litere active ([], ['B'], ['A','D'] etc.)

for idx, (x0, y0, x1, y1) in enumerate(boxes):
    roi = img[y0:y1, x0:x1]

    # pentru debug: scrie o imagine per chenar
    ans20, bw_roi, _ = read_answers_from_bubbles(
        roi,
        fill_threshold=0.80,
        n_rows=20,
        n_choices=5,
        debug_path=f"dbg_bubbles_active_box_{idx}.png"
    )
    cv2.imwrite(f"dbg_binarized_box_{idx}.png", bw_roi)

    col = idx % 5
    block = idx // 5
    q_start = (1 + col * 20) if block == 0 else (101 + col * 20)

    for i, active_letters in enumerate(ans20):
        all_answers[q_start + i] = active_letters

print("Exemplu 1..10 (liste litere active):", [all_answers[i] for i in range(1, 11)])
print("Exemplu 101..110 (liste litere active):", [all_answers[i] for i in range(101, 111)])
print("Debug: dbg_bubbles_active_box_0.png, dbg_binarized_box_0.png etc.")

