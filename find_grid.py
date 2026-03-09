"""
find_grid.py — Detectare markeri grilă din poze reale (dual-path)

Traseul DARK: fundal întunecat → izolăm foaia (convex hull + mască),
              vopsim restul cu alb, căutăm markeri doar pe foaie.
Traseul LIGHT: fundal luminos → căutăm markeri global,
              validăm geometric (paralelogram) + similaritate.
"""

import cv2
import numpy as np
import os
import itertools


# ─────────────────────────────────────────────
# 1. IZOLARE FOAIE (doar pentru fundal întunecat)
# ─────────────────────────────────────────────

def find_paper_mask(gray):
    """
    Găsește foaia albă pe un fundal închis la culoare.
    Folosește blur masiv (distruge textura covorului) + Otsu + convex hull.
    Returnează (mask, bbox) unde mask e 255 pe foaie, 0 în rest.
    bbox = (x, y, w, h) = bounding box-ul foii.
    Returnează (None, None) dacă nu găsește nimic valid.
    """
    H, W = gray.shape
    img_area = H * W

    # Blur masiv — "miopie" care dizolvă textura covorului dar păstrează forma mare a foii
    blur = cv2.GaussianBlur(gray, (25, 25), 0)

    # Otsu — separă automat alb (foaie) de întunecat (fundal)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morfologie agresivă: eliminăm granulații mici, păstrăm doar masa solidă
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < img_area * 0.10:
        return None, None

    # Convex hull = formă solidă care acoperă exact foaia
    hull = cv2.convexHull(best)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, -1)


    x, y, w, h = cv2.boundingRect(hull)
    # Padding foarte generos — pe imagini subexpuse, markerii pot fi chiar
    # în afara zonei Otsu (sunt la marginea foii, pe zona subexpusă)
    pad = 150
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(W - x, w + 2 * pad)
    h = min(H - y, h + 2 * pad)
    h = min(H - y, h + 2 * pad)

    return mask, (x, y, w, h)


# ─────────────────────────────────────────────
# 2. DETECȚIE CANDIDAȚI MARKER
# ─────────────────────────────────────────────

def detect_black_squares(img_bgr, relaxed_validation=False):
    """
    Detectează markerii negri ai grilei.
    Decide automat traseul (dark vs light) pe baza luminozității marginilor.
    Returnează lista de max 4 markeri validați geometric.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    H, W = gray.shape

    # ── Decizia de traseu: analizăm luminozitatea marginilor ──
    pad_h = max(1, int(H * 0.02))
    pad_w = max(1, int(W * 0.02))
    border_pixels = np.concatenate([
        gray[:pad_h, :].flatten(),
        gray[-pad_h:, :].flatten(),
        gray[:, :pad_w].flatten(),
        gray[:, -pad_w:].flatten()
    ])
    border_mean = float(np.mean(border_pixels))
    is_dark_bg = border_mean < 85

    traseu = "DARK" if is_dark_bg else "LIGHT"
    print(f"  Border luma: {border_mean:.0f} → Traseu: {traseu}")

    # ── Traseu DARK: crop generos pe zona foii, fără a albi (markeri pot fi pe margine subexpusă) ──
    if is_dark_bg:
        mask, bbox = find_paper_mask(gray_blur)
        if mask is not None:
            rx, ry, rw, rh = bbox
            gray_roi = gray_blur[ry:ry+rh, rx:rx+rw]
            offset = (rx, ry)
            print(f"  Foaie izolată (crop): ({rx},{ry}) {rw}x{rh}")
        else:
            gray_roi = gray_blur
            offset = (0, 0)
            rw, rh = W, H
            print(f"  [WARN] Nu am găsit foaia, mergem global")
        # CLAHE: egalizare adaptivă de contrast (recuperează markeri subexpuși)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_roi = clahe.apply(gray_roi)
        min_contrast = 40  # prag mai scăzut pe fundal dark (CLAHE + paralelogram compensează)
    else:
        # ── Traseu LIGHT: căutăm global, lăsăm geometria să filtreze ──
        gray_roi = gray_blur
        offset = (0, 0)
        rw, rh = W, H
        min_contrast = 40 if relaxed_validation else 100  # prag mai indulgent pe STAS (elimină taste, cabluri)

    # ── Extragere candidați din ROI ──
    roi_h, roi_w = gray_roi.shape
    roi_area = roi_h * roi_w
    roi_mean = float(np.mean(gray_roi))

    all_candidates = []

    # Multi-threshold: adaptive, Otsu, și praguri relative
    masks = []
    masks.append(cv2.adaptiveThreshold(
        gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 10
    ))
    _, otsu = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(otsu)
    for delta in [-60, -40, -20]:
        thr = max(10, int(roi_mean + delta))
        _, bw = cv2.threshold(gray_roi, thr, 255, cv2.THRESH_BINARY_INV)
        masks.append(bw)

    for mask_idx, bw in enumerate(masks):
        k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        bw = cv2.morphologyEx(bw, cv2.MORPH_ERODE, k_erode, iterations=1)
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close, iterations=1)

        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 200 or area > roi_area * 0.05:
                continue

            x, y, w, h = cv2.boundingRect(c)
            patch = gray_roi[y:y+h, x:x+w]
            if patch.size == 0:
                continue

            patch_mean = float(np.mean(patch))
            ar = max(w, h) / (min(w, h) + 1e-6)

            # ── Contrast local: zona din jurul markerului trebuie să fie ALBĂ ──
            # Markeri reali: negri pe foaie albă → contrast > 100
            # Taste/covor: negri pe fundal negru → contrast < 60
            pad_s = 40
            sy1 = max(0, y - pad_s)
            sy2 = min(roi_h, y + h + pad_s)
            sx1 = max(0, x - pad_s)
            sx2 = min(roi_w, x + w + pad_s)
            surround = gray_roi[sy1:sy2, sx1:sx2]
            smask = np.ones(surround.shape, dtype=bool)
            smask[y - sy1:y - sy1 + h, x - sx1:x - sx1 + w] = False
            if smask.any():
                surround_mean = float(np.mean(surround[smask]))
            else:
                surround_mean = patch_mean
            local_contrast = surround_mean - patch_mean

            hull = cv2.convexHull(c)
            solidity = area / (cv2.contourArea(hull) + 1e-6)

            # Filtru final: ar >= 1.5, negru, pe fundal alb (contrast > min_contrast), solid
            if (ar >= 1.5 and local_contrast > min_contrast
                    and min(w, h) >= 10 and max(w, h) >= 20
                    and max(w, h) < 180 and solidity >= 0.4):
                cx = (x + offset[0]) + w / 2
                cy = (y + offset[1]) + h / 2
                all_candidates.append({
                    "bbox": (x + offset[0], y + offset[1], w, h),
                    "center": (cx, cy),
                    "area": area,
                    "patch_mean": patch_mean,
                    "aspect_ratio": ar,
                    "solidity": solidity,
                    "thr_used": mask_idx,
                })

    candidates = _deduplicate(all_candidates)
    print(f"  Candidați unici după dedup: {len(candidates)}")
    for i, c in enumerate(candidates):
        print(f"    C{i}: {c['bbox']} ar={c['aspect_ratio']:.1f} area={c['area']:.0f} solidity={c['solidity']:.2f}")

    # ── Validare geometrică (paralelogram) ──
    candidates = _find_best_grid(candidates, rw, rh)
    return candidates


# ─────────────────────────────────────────────
# 3. VALIDARE GEOMETRICĂ (PARALELOGRAM)
# ─────────────────────────────────────────────

def _find_best_grid(candidates, rw, rh):
    """
    Caută setul optim de 3 sau 4 markeri care formează un patrulater valid
    sub deformare de perspectivă.
    """
    # Funcție utilitară pentru a verifica dacă 4 puncte formează un poligon convex
    def is_convex(tl, tr, br, bl):
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        c1 = cross(tl, tr, br)
        c2 = cross(tr, br, bl)
        c3 = cross(br, bl, tl)
        c4 = cross(bl, tl, tr)
        return (c1 > 0 and c2 > 0 and c3 > 0 and c4 > 0) or (c1 < 0 and c2 < 0 and c3 < 0 and c4 < 0)

    # Funcție pentru a sorta 4 puncte: TL, TR, BR, BL
    def order_4_points(pts):
        pts = sorted(pts, key=lambda p: (p[1], p[0])) # top to bottom
        top = sorted(pts[:2], key=lambda p: p[0])
        bottom = sorted(pts[2:], key=lambda p: p[0])
        return top[0], top[1], bottom[1], bottom[0]

    if len(candidates) < 3:
        return candidates

    # Scor de calitate per candidat
    for c in candidates:
        pm = max(1.0, c["patch_mean"])
        c["score"] = (c["area"] * c["solidity"]) / pm
    
    # Daca avem prea multi candidati (cazul STAS pe foaia gata printata, unde vede toate bulele)
    # limitam puternic pool-ul cautand doar punctele care s-ar califica geografic drept colturi
    if len(candidates) > 20:
        candidates_tl = sorted(candidates, key=lambda c: (c["center"][0] ** 2 + c["center"][1] ** 2))[:10]
        candidates_tr = sorted(candidates, key=lambda c: ((rw - c["center"][0]) ** 2 + c["center"][1] ** 2))[:10]
        candidates_br = sorted(candidates, key=lambda c: ((rw - c["center"][0]) ** 2 + (rh - c["center"][1]) ** 2))[:10]
        candidates_bl = sorted(candidates, key=lambda c: (c["center"][0] ** 2 + (rh - c["center"][1]) ** 2))[:10]
        
        pool_dict = {id(c): c for subset in [candidates_tl, candidates_tr, candidates_br, candidates_bl] for c in subset}
        pool = list(pool_dict.values())
        pool.sort(key=lambda c: c["score"], reverse=True)
        pool = pool[:25] # Hard cap absolut la 25 = 12650 combinatii maxime (ruleaza instant)
    else:
        candidates.sort(key=lambda c: c["score"], reverse=True)
        pool = candidates[:40]

    best_grid_4 = None
    best_score_4 = -1

    # Funcție utilitară pentru a verifica dacă 4 puncte formează un poligon convex
    def is_convex(tl, tr, br, bl):
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        c1 = cross(tl, tr, br)
        c2 = cross(tr, br, bl)
        c3 = cross(br, bl, tl)
        c4 = cross(bl, tl, tr)
        return (c1 > 0 and c2 > 0 and c3 > 0 and c4 > 0) or (c1 < 0 and c2 < 0 and c3 < 0 and c4 < 0)

    # Funcție pentru a sorta 4 puncte: TL, TR, BR, BL
    def order_4_points(pts):
        pts = sorted(pts, key=lambda p: (p[1], p[0])) # top to bottom
        top = sorted(pts[:2], key=lambda p: p[0])
        bottom = sorted(pts[2:], key=lambda p: p[0])
        return top[0], top[1], bottom[1], bottom[0]

    # 1. Căutăm seturi de câte 4 markeri
    for quad in itertools.combinations(pool, 4):
        pts = [np.array(c["center"]) for c in quad]
        tl, tr, br, bl = order_4_points(pts)
        
        # Orientare folosind Y (tl sus, bl jos) și X (tl stânga, tr dreapta)
        if not is_convex(tl, tr, br, bl):
            continue
            
        w_top = np.linalg.norm(tr - tl)
        w_bot = np.linalg.norm(br - bl)
        h_left = np.linalg.norm(bl - tl)
        h_right = np.linalg.norm(br - tr)

        # Filtre de perspectivă
        if min(w_top, w_bot) < rw * 0.15 or min(h_left, h_right) < rh * 0.15:
            continue
            
        # Distorsionarea de perspectivă nu ar trebui să fie extremă (ex. de 3 ori mai mic la un capăt)
        if w_top / (w_bot + 1e-6) > 3 or w_bot / (w_top + 1e-6) > 3:
            continue
        if h_left / (h_right + 1e-6) > 3 or h_right / (h_left + 1e-6) > 3:
            continue

        # Proporția grilei standard OMR e mai înaltă decât lată (între 1.0 și 3.0)
        avg_w = (w_top + w_bot) / 2
        avg_h = (h_left + h_right) / 2
        if not (0.8 < avg_h / (avg_w + 1e-6) < 3.5):
            continue
            
        # Unghiurile dintr-un patrulater valid de perspectivă n-ar trebui să fie prea ascuțite (<45 grade)
        def calc_angle(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            return np.arccos(np.clip(cosine, -1.0, 1.0)) * 180 / np.pi
            
        ang_tl = calc_angle(bl, tl, tr)
        ang_tr = calc_angle(tl, tr, br)
        ang_br = calc_angle(tr, br, bl)
        ang_bl = calc_angle(br, bl, tl)
        if min(ang_tl, ang_tr, ang_br, ang_bl) < 45 or max(ang_tl, ang_tr, ang_br, ang_bl) > 135:
            continue

        score = sum(c["score"] for c in quad)
        if score > best_score_4:
            best_score_4 = score
            best_grid_4 = list(quad)

    if best_grid_4:
        best_grid_4.sort(key=lambda c: (c["center"][1], c["center"][0]))
        return best_grid_4

    # 2. Dacă nu găsim 4, căutăm un T/L bun de 3 markeri și estimăm al 4-lea
    best_grid_3 = None
    best_score_3 = -1
    for trio in itertools.combinations(pool, 3):
        pts = [np.array(c["center"]) for c in trio]
        
        # Găsim colțul din mijloc (unghiul de ~90 grade dintr-o perspectivă)
        d01 = np.linalg.norm(pts[0] - pts[1])
        d12 = np.linalg.norm(pts[1] - pts[2])
        d20 = np.linalg.norm(pts[2] - pts[0])
        dists = sorted([(d01, 0, 1, 2), (d12, 1, 2, 0), (d20, 2, 0, 1)])
        _, idx_A, idx_B, idx_C = dists[2] # B este opus celei mai lungi laturi
        
        pA, pB, pC = pts[idx_A], pts[idx_B], pts[idx_C]
        vBA, vBC = pA - pB, pC - pB
        nBA, nBC = np.linalg.norm(vBA), np.linalg.norm(vBC)

        if nBA < rw * 0.15 or nBC < rw * 0.15:
            continue
            
        cos_B = np.dot(vBA, vBC) / (nBA * nBC + 1e-6)
        if abs(cos_B) > 0.4: # toleranță mai mare pentru perspectivă (70-110 grade)
            continue
            
        gw, gh = min(nBA, nBC), max(nBA, nBC)
        if not (0.8 < gh / (gw + 1e-6) < 3.5):
            continue

        pD = pB + vBA + vBC # estimare paralelogram (nu e perfectă în perspectivă, dar e fallback)
        score = sum(c["score"] for c in trio)
        
        if score > best_score_3:
            best_score_3 = score
            wD = int(np.median([c["bbox"][2] for c in trio]))
            hD = int(np.median([c["bbox"][3] for c in trio]))
            best_grid_3 = list(trio)
            best_grid_3.append({
                "bbox": (int(pD[0] - wD/2), int(pD[1] - hD/2), wD, hD),
                "center": (float(pD[0]), float(pD[1])),
                "area": wD * hD,
                "patch_mean": 0,
                "aspect_ratio": 1.0,
                "solidity": 1.0,
                "thr_used": -1,
                "score": 0,
                "estimated": True,
            })
            print(f"  [INFO] Al 4-lea marker estimat (fallback) la ({pD[0]:.0f}, {pD[1]:.0f})")

    if best_grid_3:
         best_grid_3.sort(key=lambda c: (c["center"][1], c["center"][0]))
         return best_grid_3

    return pool[:4]


# ─────────────────────────────────────────────
# 4. UTILITARE
# ─────────────────────────────────────────────

def _deduplicate(candidates, iou_thr=0.5):
    """Elimină candidații suprapuși (IoU > prag)."""
    if not candidates:
        return []
    candidates.sort(key=lambda c: c["thr_used"])
    kept = []
    for c in candidates:
        x1, y1, w1, h1 = c["bbox"]
        is_dup = False
        for k in kept:
            x2, y2, w2, h2 = k["bbox"]
            xa, ya = max(x1, x2), max(y1, y2)
            xb, yb = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
            if xa < xb and ya < yb:
                inter = (xb - xa) * (yb - ya)
                union = w1*h1 + w2*h2 - inter
                if inter / (union + 1e-6) > iou_thr:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(c)
    return kept


def draw_markers(img_bgr, candidates):
    """Desenează markerii finali pe imagine (verde = real, galben = estimat)."""
    dbg = img_bgr.copy()
    for i, c in enumerate(candidates):
        x, y, w, h = c["bbox"]
        is_est = c.get("estimated", False)
        color = (0, 200, 255) if is_est else (0, 255, 0)
        cv2.rectangle(dbg, (x, y), (x+w, y+h), color, 3)
        tag = "EST" if is_est else f"m={c['patch_mean']:.0f}"
        label = f"#{i} {w}x{h} {tag}"
        cv2.putText(dbg, label, (x, max(15, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return dbg


# ─────────────────────────────────────────────
# 5. MAIN (test pe cele 3 poze)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    input_dir = "detect-poze-test/"
    output_dir = "detect-poze-output/"
    os.makedirs(output_dir, exist_ok=True)

    photos = [
        os.path.join(input_dir, "IMG_0530.jpg"),
        os.path.join(input_dir, "IMG_0532.jpg"),
        os.path.join(input_dir, "IMG_0536.jpg"),
    ]

    for photo_path in photos:
        name = os.path.splitext(os.path.basename(photo_path))[0]
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")

        img = cv2.imread(photo_path)
        if img is None:
            print(f"  EROARE: nu pot citi {photo_path}")
            continue

        scale = 2000 / max(img.shape[:2])
        if scale < 1.0:
            img_small = cv2.resize(img, None, fx=scale, fy=scale)
        else:
            img_small = img.copy()

        print(f"  Original: {img.shape[:2]}, Scaled: {img_small.shape[:2]}")

        candidates = detect_black_squares(img_small)

        print(f"  ✅ Markeri finali: {len(candidates)}")
        for i, c in enumerate(candidates):
            x, y, w, h = c["bbox"]
            est = " [ESTIMAT]" if c.get("estimated") else ""
            print(f"    #{i}: pos=({x},{y}) size={w}x{h} "
                  f"ar={c['aspect_ratio']:.1f} mean={c['patch_mean']:.0f}{est}")

        dbg = draw_markers(img_small, candidates)
        cv2.imwrite(os.path.join(output_dir, f"{name}_markeri.jpg"), dbg)

    print(f"\n✅ Rezultate salvate în {output_dir}")
