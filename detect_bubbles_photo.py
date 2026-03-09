"""
detect_bubbles_photo.py — Detectare buline din poze reale

Flow:
  1. Detectează markeri pe imagine scalată (find_grid.py)
  2. Scalează coordonatele markerilor înapoi la rezoluția originală
  3. Perspective warp cu padding generos → grilă mare, clară
  4. Împarte grila în 10 box-uri pe baza gap-urilor reale din proiecție
  5. Detectare buline per box + vizualizare
"""

import cv2
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from find_grid import detect_black_squares
from main import detect_bubbles, read_answers_from_bubbles


def order_corners(candidates):
    """Ordonează cei 4 markeri: TL, TR, BL, BR."""
    centers = np.array([c["center"] for c in candidates], dtype=np.float32)
    s = centers.sum(axis=1)
    d = np.diff(centers, axis=1).flatten()
    tl = centers[np.argmin(s)]
    br = centers[np.argmax(s)]
    tr = centers[np.argmin(d)]
    bl = centers[np.argmax(d)]
    return np.array([tl, tr, bl, br], dtype=np.float32)


def warp_grid(img_bgr, corners_4, pad_pct_x=0.12, pad_pct_y=0.04):
    """
    Perspective warp cu padding generos.
    Padding-ul este calculat în spațiul destinație, nu prin extrapolare liniară
    în spațiul sursă (care distorsionează perspectiva).
    """
    tl, tr, bl, br = corners_4

    # Dimensiuni brute (distanța între markeri în imaginea originală)
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)

    raw_w = max(w_top, w_bot)
    raw_h = max(h_left, h_right)

    # Minim 1200px lățime pentru calitatea detecției (distanța dintre markeri intern)
    scale = max(1.0, 1200.0 / raw_w)
    rw = int(raw_w * scale)
    rh = int(raw_h * scale)

    # Calculăm padding-ul în pixeli (în spațiul destinației, deci e ortogonal și rectiliniu)
    pad_x = int(rw * pad_pct_x)
    pad_y = int(rh * pad_pct_y)

    tw = rw + 2 * pad_x
    th = rh + 2 * pad_y

    src = np.array([tl, tr, bl, br], dtype=np.float32)

    # Destinația = markerii trebuie să ajungă exact la coordonatele cu padding
    dst = np.array([
        [pad_x, pad_y],
        [tw - 1 - pad_x, pad_y],
        [pad_x, th - 1 - pad_y],
        [tw - 1 - pad_x, th - 1 - pad_y]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, M, (tw, th))
    return warped, M



def warped_to_original(pts_warped, M):
    """
    Mapează puncte din spațiul warped înapoi în spațiul imaginii originale.
    pts_warped: array Nx2 de puncte (x,y) în spațiul warped
    M: matricea de perspectivă (original→warped)
    Returnează: array Nx2 de puncte în spațiul original
    """
    M_inv = np.linalg.inv(M)
    pts = np.array(pts_warped, dtype=np.float64).reshape(-1, 1, 2)
    pts_orig = cv2.perspectiveTransform(pts, M_inv)
    return pts_orig.reshape(-1, 2).astype(np.int32)


def find_5_columns(warped_bgr):
    """
    Găsește cele 5 coloane de buline folosind proiecția verticală.
    Caută 4 gap-uri mari (spații între grupuri de coloane) în proiecția X.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # Binarizare
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Proiecție pe X (sumă pe fiecare coloană verticală)
    proj_x = bw.sum(axis=0).astype(np.float64)
    # Smooth
    ksize = max(3, W // 50) | 1
    proj_x_smooth = cv2.GaussianBlur(proj_x.reshape(1, -1), (1, ksize), 0).ravel()

    # Normalizare
    proj_norm = proj_x_smooth / (proj_x_smooth.max() + 1e-6)

    # Găsim regiunile cu conținut (>10% din max) vs gap-uri (<10%)
    threshold = 0.10
    is_content = proj_norm > threshold

    # Găsim tranzițiile content→gap și gap→content
    transitions = []
    for i in range(1, len(is_content)):
        if is_content[i - 1] and not is_content[i]:
            transitions.append(("end", i))
        elif not is_content[i - 1] and is_content[i]:
            transitions.append(("start", i))

    # Găsim gap-urile (end → start) și le sortăm după lățime
    gaps = []
    for i in range(len(transitions) - 1):
        if transitions[i][0] == "end" and transitions[i + 1][0] == "start":
            gap_start = transitions[i][1]
            gap_end = transitions[i + 1][1]
            gap_width = gap_end - gap_start
            gap_center = (gap_start + gap_end) // 2
            gaps.append((gap_width, gap_center, gap_start, gap_end))

    # Sortăm gap-urile după lățime descrescător, luăm cele mai mari 4
    gaps.sort(key=lambda g: g[0], reverse=True)
    if len(gaps) >= 4:
        col_separators = sorted([g[1] for g in gaps[:4]])
    else:
        # Fallback: împărțire egală
        col_separators = [W * (i + 1) // 5 for i in range(4)]

    # Bordurile coloanelor
    col_borders = [0] + col_separators + [W]
    return col_borders


def find_band_split(warped_bgr):
    """
    Găsește separarea orizontală între cele 2 benzi (sus/jos).
    Folosim smoothing agresiv (kernel=101) + rolling average ca să ignorăm
    mini-gap-urile dintre rânduri individuale și să găsim separatorul REAL.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    proj_y = bw.sum(axis=1).astype(np.float64)

    # Smoothing agresiv ca să "topim" gap-urile individuale dintre rânduri
    kbig = min(101, H // 5) | 1
    proj_smooth = cv2.GaussianBlur(proj_y.reshape(-1, 1), (kbig, 1), 0).ravel()
    proj_norm = proj_smooth / (proj_smooth.max() + 1e-6)

    # Rolling average cu window mare — smoothează și mai mult
    window = max(15, H // 100)
    kernel = np.ones(window) / window
    proj_rolling = np.convolve(proj_norm, kernel, mode='same')

    # Căutăm minimul în zona centrală (38%-62% din H)
    search_start = int(H * 0.38)
    search_end = int(H * 0.62)
    region = proj_rolling[search_start:search_end]
    mid_split = search_start + int(np.argmin(region))

    # Margini sus/jos ale conținutului
    content_mask = proj_norm > 0.05
    content_idx = np.where(content_mask)[0]
    if len(content_idx) > 0:
        y_top = max(0, content_idx[0] - 5)
        y_bot = min(H, content_idx[-1] + 5)
    else:
        y_top, y_bot = 0, H

    return y_top, mid_split, y_bot


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

input_dir = "detect-poze-test/"
output_dir = "detect-poze-output/"
os.makedirs(output_dir, exist_ok=True)

photos = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
photos.sort()

colors = [
    (255, 0, 255), (0, 255, 255), (255, 128, 0),
    (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (128, 0, 255), (0, 128, 255), (255, 0, 128), (128, 255, 0),
]

for photo_path in photos:
    name = os.path.splitext(os.path.basename(photo_path))[0]
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    img_full = cv2.imread(photo_path)
    if img_full is None:
        print(f"  EROARE: nu pot citi {photo_path}")
        continue

    # Detectare markeri pe imagine scalată
    scale = 2000 / max(img_full.shape[:2])
    img_small = cv2.resize(img_full, None, fx=scale, fy=scale)

    candidates = detect_black_squares(img_small)
    print(f"  Markeri: {len(candidates)}")
    if len(candidates) < 3:
        print(f"  ❌ Prea puțini markeri")
        continue

    # Scalăm coordonatele înapoi la full-res
    inv_scale = 1.0 / scale
    for c in candidates:
        c["center"] = (c["center"][0] * inv_scale, c["center"][1] * inv_scale)

    corners = order_corners(candidates)
    print(f"  Colțuri (full-res): TL={corners[0].astype(int)}, TR={corners[1].astype(int)}")
    print(f"                      BL={corners[2].astype(int)}, BR={corners[3].astype(int)}")

    # Warp cu padding generos din full-res
    warped, M = warp_grid(img_full, corners, pad_pct_x=0.16, pad_pct_y=0.04)
    print(f"  Warped: {warped.shape[1]}x{warped.shape[0]}")
    cv2.imwrite(os.path.join(output_dir, f"{name}_warped.jpg"), warped)

    # Găsim cele 5 coloane
    col_borders = find_5_columns(warped)
    print(f"  Coloane (borduri X): {col_borders}")

    # Găsim separarea benzilor
    y_top, mid_split, y_bot = find_band_split(warped)
    print(f"  Benzi: sus=[{y_top}, {mid_split}], jos=[{mid_split}, {y_bot}]")

    # 10 box-uri (coordonate în spațiul warped)
    boxes = []
    bands = [(y_top, mid_split), (mid_split, y_bot)]
    for by0, by1 in bands:
        for ci in range(5):
            boxes.append((col_borders[ci], by0, col_borders[ci + 1], by1))

    # Desenare pe warped (debug) + pe original (cu perspectivă)
    dbg_warped = warped.copy()
    dbg_original = img_full.copy()

    # =========================
    # Citește răspunsuri (grading)
    # =========================
    all_answers = {}
    
    # Scalare grosime linii și fonturi în funcție de rezoluția imaginii
    orig_diag = np.sqrt(img_full.shape[0]**2 + img_full.shape[1]**2)
    orig_thick = max(2, int(orig_diag / 800))   # ~9 pt 5700x4300
    orig_font_scale = orig_diag / 4000           # ~1.8 pt 5700x4300
    orig_font_thick = max(1, int(orig_diag / 2000))

    warp_diag = np.sqrt(warped.shape[0]**2 + warped.shape[1]**2)
    warp_thick = max(2, int(warp_diag / 800))
    warp_font_scale = warp_diag / 4000
    warp_font_thick = max(1, int(warp_diag / 2000))

    for i, (x0, y0, x1, y1) in enumerate(boxes):
        color = colors[i]

        band = i // 5  # 0 = sus, 1 = jos
        col = i % 5
        q_start = (1 + col * 20) if band == 0 else (101 + col * 20)
        q_end = q_start + 19
        label = f"Q{q_start}-{q_end}"

        # ── Desenare pe warped (dreptunghiuri drepte) ──
        cv2.rectangle(dbg_warped, (x0, y0), (x1, y1), color, warp_thick)
        cv2.putText(dbg_warped, label, (x0 + 5, y0 + int(30 * warp_font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, warp_font_scale, color, warp_font_thick)

        # ── Desenare pe original (quadrilaterale cu perspectivă) ──
        # Cele 4 colțuri ale box-ului din warped → original via inversul M
        box_corners_warped = np.array([
            [x0, y0], [x1, y0], [x1, y1], [x0, y1]
        ], dtype=np.float64)
        box_corners_orig = warped_to_original(box_corners_warped, M)
        cv2.polylines(dbg_original, [box_corners_orig], True, color, orig_thick)
        # Label la colțul TL al box-ului pe original
        label_x = box_corners_orig[0][0] + int(10 * orig_font_scale)
        label_y = box_corners_orig[0][1] + int(30 * orig_font_scale)
        cv2.putText(dbg_original, label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, orig_font_scale, color, orig_font_thick)

        # Extract region of interest for this box
        roi = warped[y0:y1, x0:x1]
        if roi.size == 0:
            continue
            
        # Folosim metoda de grading din main.py
        ans20, bw_roi, _ = read_answers_from_bubbles(
            roi,
            fill_threshold=0.50, # Un prag rezonabil pentru poze
            n_rows=20,
            n_choices=5
        )
        
        # Salvăm un debug pentru primul box
        if band == 0 and col == 0:
            cv2.imwrite(os.path.join(output_dir, f"{name}_debug_bw_box0.png"), bw_roi)

        # Salvăm răspunsurile
        for r_idx, active_letters in enumerate(ans20):
            all_answers[q_start + r_idx] = active_letters
            
        # Păstrăm desenul cu bulele doar pentru vizualizare (detect_bubbles extrage doar contururile tuturor bulelor)
        bubbles, _ = detect_bubbles(roi)
        total_bubbles = len(bubbles)
        
        # Desenăm bulinele detectate pe warped
        for b in bubbles:
            (cx, cy), (MA, ma), angle = b["ellipse"]
            cv2.ellipse(dbg_warped,
                        (int(cx) + x0, int(cy) + y0),
                        (int(MA / 2), int(ma / 2)),
                        angle, 0, 360, color, max(1, warp_thick // 2))

        # Desenăm bulinele și pe original (proiecție cu perspectivă)
        for b in bubbles:
            (cx, cy), (MA, ma), angle = b["ellipse"]
            bcx_w = cx + x0
            bcy_w = cy + y0
            n_pts = 32
            theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            a_half = MA / 2
            b_half = ma / 2
            ang_rad = np.radians(angle)
            cos_a, sin_a = np.cos(ang_rad), np.sin(ang_rad)
            ell_pts_w = np.zeros((n_pts, 2), dtype=np.float64)
            for k in range(n_pts):
                px = a_half * np.cos(theta[k])
                py = b_half * np.sin(theta[k])
                ell_pts_w[k, 0] = bcx_w + px * cos_a - py * sin_a
                ell_pts_w[k, 1] = bcy_w + px * sin_a + py * cos_a
            ell_pts_orig = warped_to_original(ell_pts_w, M)
            cv2.polylines(dbg_original, [ell_pts_orig], True, color,
                          max(1, orig_thick // 2))

        print(f"    {label}: evaluate {len(ans20)} rânduri (detectate {total_bubbles} cercuri)")

    cv2.imwrite(os.path.join(output_dir, f"{name}_boxes.jpg"), dbg_warped)
    cv2.imwrite(os.path.join(output_dir, f"{name}_boxes_original.jpg"), dbg_original)
    
    # Debug print:
    print(f"  Exemplu Q1..10 (liste litere active):", [all_answers.get(i, []) for i in range(1, 11)])
    
    # ── Comparare rezultate ──
    try:
        with open("mock_answers.json", "r") as f:
            mock_correct = json.load(f)
            
        score_total = 0
        max_score = 0
        
        # Contorizare erori pt sumare
        correct_count = 0
        wrong_count = 0
        
        for q_str, correct_ans in mock_correct.items():
            q_idx = int(q_str)
            if q_idx in all_answers:
                detected = all_answers[q_idx]
                if q_idx <= 50:
                    points_pos = 4
                    if len(detected) == 1 and detected == correct_ans:
                        score_total += points_pos
                        correct_count += 1
                    else:
                        wrong_count += 1
                else:
                    points_pos = 5
                    if 2 <= len(detected) <= 4 and sorted(detected) == sorted(correct_ans):
                        score_total += points_pos
                        correct_count += 1
                    else:
                        wrong_count += 1
                max_score += points_pos
                
        print(f"  ✅ Rezultate Q1-Q200:")
        print(f"     Corecte: {correct_count} | Gresite/Anulate: {wrong_count}")
        print(f"     SCOR FINAL: {score_total} / {max_score}")
        
    except FileNotFoundError:
        print(f"  Fisierul mock_answers.json lipseste.")
    print(f"  ✅ Salvat: {name}_warped.jpg, {name}_boxes.jpg, {name}_boxes_original.jpg")

print(f"\n✅ Rezultate în {output_dir}")
