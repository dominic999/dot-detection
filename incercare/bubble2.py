import numpy as np
import cv2

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
    _, labels, centers = cv2.kmeans(ys, n_rows, None, criteria, 10, cv2.KMEANS_PP_CENTERS) # pyright: ignore[]

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
