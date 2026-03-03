from pdf2image import convert_from_path
import cv2
import numpy as np

def pil_to_bgr(pil_img):
    rgb = np.array(pil_img)  # RGB
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def find_valleys_between_peaks(proj, n_sections):
    """
    proj: vector 1D (de ex proj_x)
    n_sections: 5 pentru coloane, 2 pentru blocuri
    Returnează boundaries (n_sections+1) index-uri: [start, cut1, ..., end]
    """
    p = proj.copy()
    p = (p - p.min()) / (p.max() - p.min() + 1e-6)

    # găsim zonele "pline" (unde există conținut)
    mask = p > 0.25
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise RuntimeError("Nu găsesc conținut în proiecție. Prag prea mare sau binarizare slabă.")

    start = int(idx[0])
    end   = int(idx[-1])

    # împărțim intervalul [start,end] în n_sections și găsim minimul (valley) în jurul fiecărei tăieturi
    cuts = [start]
    for i in range(1, n_sections):
        target = start + (end - start) * i / n_sections
        target = int(target)

        # caut minimul într-o fereastră în jurul target
        win = max(20, (end - start)//(n_sections*8))
        a = max(start, target - win)
        b = min(end,   target + win)
        valley = a + int(np.argmin(p[a:b+1]))
        cuts.append(valley)

    cuts.append(end)
    return cuts

def binarize_for_layout(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # prinde bine pe fundal alb
    bw = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
    return bw

def kmeans_1d(values, k):
    values = values.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return labels.flatten(), centers.flatten()

def split_bounds_from_centers(centers, min_val, max_val):
    centers = np.sort(centers)
    bounds = [min_val]
    for i in range(len(centers) - 1):
        bounds.append(int((centers[i] + centers[i+1]) / 2))
    bounds.append(max_val)
    return bounds

def find_best_horizontal_cut(proj_y, H, mid_frac=0.52, band=0.18):
    """
    Caută cel mai mare 'gol' (proiecție mică) într-o bandă în jurul mijlocului.
    Returnează y_cut.
    """
    p = proj_y.astype(np.float32)
    p = (p - p.min()) / (p.max() - p.min() + 1e-6)

    mid = int(H * mid_frac)
    a = max(0, int(mid - H * band))
    b = min(H - 1, int(mid + H * band))

    # prag dinamic: "rânduri aproape goale"
    thresh = np.percentile(p[a:b+1], 15)  # 15% cele mai mici
    low = p[a:b+1] <= thresh

    # găsește cea mai lungă secvență de True
    best_len = 0
    best_center = mid
    i = 0
    while i < len(low):
        if not low[i]:
            i += 1
            continue
        j = i
        while j < len(low) and low[j]:
            j += 1
        run_len = j - i
        if run_len > best_len:
            best_len = run_len
            best_center = a + (i + j)//2
        i = j

    return best_center

def get_10_boxes_projection(page2_bgr):
    H, W = page2_bgr.shape[:2]

    x_off = int(W * 0.40)
    right = page2_bgr[:, x_off:W]
    print("right: ")
    bw = binarize_for_layout(right)

    # ignoră barele negre sus/jos pentru proiecții
    cut_top = int(right.shape[0] * 0.08)
    cut_bot = int(right.shape[0] * 0.92)
    bw_mid = bw[cut_top:cut_bot, :]

    proj_x = smooth_1d(bw_mid.sum(axis=0), 81)
    proj_y_mid = smooth_1d(bw_mid.sum(axis=1), 81)

    xb = find_valleys_between_peaks(proj_x, 5)

    y_cut_mid = find_best_horizontal_cut(proj_y_mid, bw_mid.shape[0])
    y_cut = cut_top + y_cut_mid
    yb = [0, y_cut, right.shape[0]-1]

    boxes = []
    for block in range(2):
        if block == 0:
            y0 = max(0, yb[0] - 25)
            y1 = max(0, yb[1] - 5)     # nu trece de cut
        else:
            y0 = min(H-1, yb[1] + 5)
            y1 = min(H-1, yb[2] + 25)

        for col in range(5):
            x0 = max(0, xb[col] - 15)
            x1 = min(right.shape[1]-1, xb[col+1] + 15)
            boxes.append((x_off + x0, y0, x_off + x1, y1))

    return boxes
def smooth_1d(arr, k=51):
    # k impar
    k = k if k % 2 == 1 else k + 1
    return cv2.GaussianBlur(arr.astype(np.float32).reshape(-1,1), (1,k), 0).ravel()

def extract_answers_box(roi_bgr, n_questions=20, n_choices=5,
                        top_margin=0.10, bottom_margin=0.06,
                        left_margin=0.12, right_margin=0.08,
                        bubble_pad=0.18, fill_thresh=0.18, ambiguous_delta=0.06):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 35, 10)

    h, w = bw.shape
    y0 = int(h * top_margin)
    y1 = int(h * (1 - bottom_margin))
    x0 = int(w * left_margin)
    x1 = int(w * (1 - right_margin))

    grid = bw[y0:y1, x0:x1]
    gh, gw = grid.shape
    row_h = gh / n_questions
    col_w = gw / n_choices

    answers = []
    for i in range(n_questions):
        ratios = []
        for j in range(n_choices):
            cy0 = int(i * row_h); cy1 = int((i + 1) * row_h)
            cx0 = int(j * col_w); cx1 = int((j + 1) * col_w)
            cell = grid[cy0:cy1, cx0:cx1]

            pad_y = int(cell.shape[0] * bubble_pad)
            pad_x = int(cell.shape[1] * bubble_pad)
            inner = cell[pad_y:cell.shape[0]-pad_y, pad_x:cell.shape[1]-pad_x]

            if inner.size == 0:
                ratios.append(0.0)
                continue

            fill_ratio = float(np.count_nonzero(inner)) / float(inner.size)
            ratios.append(fill_ratio)

        best = int(np.argmax(ratios))
        best_val = ratios[best]
        second = sorted(ratios, reverse=True)[1]

        if best_val < fill_thresh:
            answers.append(None)
        elif (best_val - second) < ambiguous_delta:
            answers.append("AMBIG")
        else:
            answers.append("ABCDE"[best])
    return answers

# =========================
# FLOW-ul tău:
# =========================
pages = convert_from_path("GrileRezidentiat/farm_C.pdf", 300)
page_2 = pages[1]  # page_2 (index 1)
img = pil_to_bgr(page_2)

boxes = get_10_boxes_projection(img)

all_answers = {}  # cheie: nr intrebare (1..200) -> 'A'/'B'/.../None/'AMBIG'
for idx, (x0, y0, x1, y1) in enumerate(boxes):
    roi = img[y0:y1, x0:x1]
    answers_20 = extract_answers_box(roi, n_questions=20)

    col = idx % 5
    block = idx // 5  # 0 sus, 1 jos

    if block == 0:
        q_start = 1 + col * 20
    else:
        q_start = 101 + col * 20

    for i, ans in enumerate(answers_20):
        all_answers[q_start + i] = ans

print("Exemplu 1..25:", [all_answers[i] for i in range(1, 26)])
print("Exemplu 101..125:", [all_answers[i] for i in range(101, 126)])

debug = img.copy()
boxes = get_10_boxes_projection(img)

for i, (x0,y0,x1,y1) in enumerate(boxes):
    cv2.rectangle(debug, (x0,y0), (x1,y1), (0,255,0), 3)
    cv2.putText(debug, str(i), (x0+10, y0+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

cv2.imwrite("debug_boxes.png", debug)
