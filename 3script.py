
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

def compute_standard_axes(bubbles, shrink=0.90):
    """
    bubbles: list cu bubble["ellipse"] = ((cx,cy),(MA,ma),angle)
    Returnează axes standard (ax, ay) în pixeli (semi-axe), micșorate cu shrink.
    """
    MAs = []
    mas = []
    for b in bubbles:
        (cx, cy), (MA, ma), angle = b["ellipse"]
        if MA > 0 and ma > 0:
            MAs.append(MA)
            mas.append(ma)

    if len(MAs) < 10:
        return None  # prea puține => ceva e greșit

    MA_med = float(np.median(MAs))
    ma_med = float(np.median(mas))

    ax = max(2, int((MA_med * shrink) / 2))
    ay = max(2, int((ma_med * shrink) / 2))
    return (ax, ay)

def fill_ratio_standard(bw, bubble, axes_std):
    """
    bw: binar invert (255 = cerneală)
    axes_std: (ax, ay) semi-axe standard
    """
    (cx, cy), (MA, ma), angle = bubble["ellipse"]
    mask = np.zeros_like(bw, dtype=np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), axes_std, angle, 0, 360, 255, -1)
    inside = bw[mask == 255]
    if inside.size == 0:
        return 0.0
    return float(np.count_nonzero(inside)) / float(inside.size)

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
    Detectează bulele ca INELE (margini) folosind Canny + fitEllipse.
    Returnează bubbles(list), edges(map)
    bubble: {"cx","cy","bbox","ellipse"}
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, k, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    H, W = edges.shape
    img_area = H * W
    bubbles = []

    for cnt in contours:
        if len(cnt) < 25:
            continue

        area = cv2.contourArea(cnt)
        if area < img_area * 0.0002 or area > img_area * 0.02:
            continue

        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (MA, ma), angle = ellipse  # MA/ma = diametre

        if MA <= 0 or ma <= 0:
            continue

        ar = max(MA, ma) / min(MA, ma)
        if ar > 1.8:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue

        bubbles.append({
            "cx": int(cx), "cy": int(cy),
            "bbox": (x, y, w, h),
            "ellipse": ellipse
        })

    return bubbles, edges

def bubble_fill_ratio(bw, bubble, shrink=0.90):
    """
    bw = imagine binară INV (255 = negru/cerneală)
    bubble are bubble["ellipse"]
    """
    ellipse = bubble["ellipse"]
    (cx, cy), (MA, ma), angle = ellipse

    mask = np.zeros_like(bw, dtype=np.uint8)
    axes = (max(1, int((MA * shrink) / 2)), max(1, int((ma * shrink) / 2)))

    cv2.ellipse(mask, (int(cx), int(cy)), axes, angle, 0, 360, 255, -1)

    inside = bw[mask == 255]
    if inside.size == 0:
        return 0.0
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

def read_answers_from_bubbles(roi_bgr, fill_threshold=0.30, n_rows=20, n_choices=5, debug_path=None):
    bubbles, _edges = detect_bubbles(roi_bgr)

    # bw pentru fill (în interior)
    bw = preprocess_bw_for_bubbles(roi_bgr)

    # IMPORTANT: standardizează dimensiunea “chenarului” (oval) pentru toate bulele
    bubbles.sort(key=lambda b: (b["cy"], b["cx"]))
    axes_std = compute_standard_axes(bubbles, shrink=0.90)
    if axes_std is None:
        return [], bw, (roi_bgr.copy() if debug_path else None)

    rows = group_bubbles_into_20_rows(bubbles, n_rows=n_rows)

    answers = []
    dbg_img = roi_bgr.copy() if debug_path else None

    if not rows:
        return [], bw, dbg_img

    for r_idx, row in enumerate(rows):
        row = sorted(row, key=lambda b: b["cx"])[:n_choices]

        active = []
        for j, b in enumerate(row):
            # În loc de bubble_fill_ratio -> folosim o mască standard
            ratio = fill_ratio_standard(bw, b, axes_std)

            if ratio >= fill_threshold:
                active.append("ABCDE"[j])

            if dbg_img is not None:
                # desenăm elipsa standard ca să vezi exact unde măsori
                (cx, cy), (MA, ma), angle = b["ellipse"]
                cv2.ellipse(dbg_img, (int(cx), int(cy)), axes_std, angle, 0, 360,
                            (0, 255, 0) if ratio >= fill_threshold else (0, 0, 255), 2)
                cv2.putText(dbg_img, f"{ratio:.2f}", (int(cx)-15, int(cy)-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0) if ratio >= fill_threshold else (0, 0, 255), 1)

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
print("Exemplu 111..120 (liste litere active):", [all_answers[i] for i in range(111, 121)])
print("Debug: dbg_bubbles_active_box_0.png, dbg_binarized_box_0.png etc.")

