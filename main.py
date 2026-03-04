##############
# Aici incep #
##############
from pdf2image import convert_from_path
import cv2
import numpy as np
import json

# =========================
# Utils
# =========================

def load_template_from_pdf(pdf_path, page_index=1, dpi=300):
    pages = convert_from_path(pdf_path, dpi)
    tpl_bgr = pil_to_bgr(pages[page_index])
    return tpl_bgr

#aliniere poza
def align_photo_to_template(photo_bgr, template_bgr, max_features=5000, good_match_percent=0.15):
    # grayscale
    im1 = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    # ORB keypoints + descriptors
    orb = cv2.ORB_create(max_features)
    k1, d1 = orb.detectAndCompute(im1, None)
    k2, d2 = orb.detectAndCompute(im2, None)

    if d1 is None or d2 is None or len(k1) < 50 or len(k2) < 50:
        raise RuntimeError("Prea puține features. Asigură-te că se vede bine foaia și e focusată.")

    # Match descriptors (Hamming)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    matches = sorted(matches, key=lambda x: x.distance)

    # păstrăm doar cele mai bune
    num_good = max(30, int(len(matches) * good_match_percent))
    matches = matches[:num_good]

    pts1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)  # photo
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)  # template

    # Homography (RANSAC)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Nu am putut calcula homografia. Încearcă altă poză (mai clară, mai puțin blur).")

    h, w = template_bgr.shape[:2]
    aligned = cv2.warpPerspective(photo_bgr, H, (w, h))

    return aligned, H


def pil_to_bgr(pil_img):
    rgb = np.array(pil_img)  # RGB
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def binarize_for_layout(img_bgr): # binarizeaza imaginea pentru layout
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
    return bw
#pentru liniile fine care mai apar scanner best practice
def smooth_1d(arr, k=51): # smootheaza o dimensiune a imaginii
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
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
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
    print("k:" + str(k))
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

    all_cx = [b["cx"] for b in bubbles]
    min_cx = np.percentile(all_cx, 5)
    max_cx = np.percentile(all_cx, 95)
    interval = (max_cx - min_cx) / (n_choices - 1) if n_choices > 1 else 1.0

    for r_idx, row in enumerate(rows):
        active = []
        for b in row:
            # În loc de index j, calculăm coloana din cx
            col_idx = int(round((b["cx"] - min_cx) / interval))
            if col_idx < 0 or col_idx >= n_choices:
                continue

            ratio = fill_ratio_standard(bw, b, axes_std)

            if ratio >= fill_threshold:
                letter = "ABCDE"[col_idx]
                if letter not in active:
                    active.append(letter)

            if dbg_img is not None:
                # desenăm elipsa standard ca să vezi exact unde măsori
                (cx, cy), (MA, ma), angle = b["ellipse"]
                cv2.ellipse(dbg_img, (int(cx), int(cy)), axes_std, angle, 0, 360,
                            (0, 255, 0) if ratio >= fill_threshold else (0, 0, 255), 2)
                cv2.putText(dbg_img, f"{ratio:.2f}", (int(cx)-15, int(cy)-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0) if ratio >= fill_threshold else (0, 0, 255), 1)

        # Sortam literele gasite si le adaugam
        active.sort()
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
        fill_threshold=0.92,
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

# =========================
# Comparatie cu JSON (Mock)
# =========================

try:
    with open("mock_answers.json", "r") as f:
        mock_correct = json.load(f)
        
    print("\n--- COMPARARE REZULTATE CU MOCK JSON ---")
    score_total = 0
    max_score = 0
    
    for q_str, correct_ans in mock_correct.items():
        q_idx = int(q_str)
        if q_idx in all_answers:
            detected = all_answers[q_idx]
            
            # Regulile: 1..50 (25%) sunt simplu complement (4 pct)
            # 51..200 (75%) sunt complement multiplu (5 pct)
            if q_idx <= 50:
                is_simple = True
                points_possible = 4
            else:
                is_simple = False
                points_possible = 5
                
            max_score += points_possible
            
            # Verificare corectitudine conform regulilor
            if is_simple:
                if len(detected) == 1 and detected == correct_ans:
                    points = points_possible
                    status = "✅ CORECT (+4)"
                else:
                    points = 0
                    status = "❌ ANULAT"
            else:
                # Complement multiplu: 2-4 raspunsuri corecte
                if len(detected) >= 2 and len(detected) <= 4 and sorted(detected) == sorted(correct_ans):
                    points = points_possible
                    status = "✅ CORECT (+5)"
                else:
                    points = 0
                    status = "❌ ANULAT"
                    
            score_total += points
            
            print(f"Intrebarea {q_idx:3d} | Asteptat: {str(correct_ans):15s} | Detectat: {str(detected):15s} | {status}")
            
    print(f"\nScor final: {score_total} / {max_score}")
    
except FileNotFoundError:
    print("\nFisierul mock_answers.json nu a fost gasit pentru comparare.")


