import cv2
import numpy as np
import os
import shutil

# Importăm funcțiile deja existente din proiect
from find_grid import detect_black_squares
from normalize_grid import normalize_orientation


def order_corners(candidates):
    """Ordonează markerii în ordinea TL, TR, BL, BR."""
    centers = np.array([c["center"] for c in candidates], dtype=np.float32)
    s = centers.sum(axis=1)
    d = np.diff(centers, axis=1).flatten()
    tl = centers[np.argmin(s)]
    br = centers[np.argmax(s)]
    tr = centers[np.argmin(d)]
    bl = centers[np.argmax(d)]
    return np.array([tl, tr, bl, br], dtype=np.float32)


def warp_grid(img_bgr, corners_4, pad_pct_x=0.12, pad_pct_y=0.04):
    """Perspective warp cu padding în spațiul destinație."""
    tl, tr, bl, br = corners_4

    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    raw_w = max(w_top, w_bot)
    raw_h = max(h_left, h_right)

    scale = max(1.0, 1200.0 / max(1.0, raw_w))
    rw = int(raw_w * scale)
    rh = int(raw_h * scale)

    pad_x = int(rw * pad_pct_x)
    pad_y = int(rh * pad_pct_y)
    tw = rw + 2 * pad_x
    th = rh + 2 * pad_y

    src = np.array([tl, tr, bl, br], dtype=np.float32)
    dst = np.array(
        [
            [pad_x, pad_y],
            [tw - 1 - pad_x, pad_y],
            [pad_x, th - 1 - pad_y],
            [tw - 1 - pad_x, th - 1 - pad_y],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, M, (tw, th))
    return warped, M


def _grid_black_ratio(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ratio = float(np.count_nonzero(bw)) / float(bw.size + 1e-6)
    return ratio


def _looks_like_grid(img_bgr):
    """
    Validare simplă: o grilă reală are procent de negru într-un interval moderat.
    Evităm warpurile complet greșite (ex: covor).
    """
    if img_bgr is None or img_bgr.size == 0:
        return False
    ratio = _grid_black_ratio(img_bgr)
    return 0.04 <= ratio <= 0.40


def _extract_grid_from_markers(img_bgr, candidates, scale):
    if len(candidates) < 3:
        return None
    inv_scale = 1.0 / scale
    for c in candidates:
        c["center"] = (c["center"][0] * inv_scale, c["center"][1] * inv_scale)
    corners = order_corners(candidates)
    warped, _ = warp_grid(img_bgr, corners, pad_pct_x=0.16, pad_pct_y=0.04)
    if not _looks_like_grid(warped):
        return None
    return warped


def _order_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _find_page_quad(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edge = cv2.Canny(blur, 50, 140)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge = cv2.dilate(edge, k, iterations=1)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, k, iterations=2)

    H, W = gray.shape
    img_area = float(H * W)
    cnts, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < 0.08 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            quad = _order_quad(approx.reshape(4, 2))
            return quad

    c0 = cnts[0]
    if float(cv2.contourArea(c0)) < 0.08 * img_area:
        return None
    rect = cv2.minAreaRect(c0)
    box = cv2.boxPoints(rect)
    return _order_quad(box)


def _warp_from_quad(img_bgr, quad):
    tl, tr, br, bl = quad
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    w = int(max(w_top, w_bot))
    h = int(max(h_left, h_right))
    if w < 300 or h < 300:
        return None

    scale = max(1.0, 1600.0 / max(1.0, float(w)))
    w = int(w * scale)
    h = int(h * scale)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    return cv2.warpPerspective(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _fallback_extract_via_page(photo_name, img_full):
    quad = _find_page_quad(img_full)
    if quad is None:
        print("   [fallback] Nu am găsit contur de pagină.")
        return None

    page = _warp_from_quad(img_full, quad)
    if page is None:
        print("   [fallback] Warp pagină invalid.")
        return None

    # Încercăm din nou detectarea markerilor pe pagina rectificată.
    scale = 2000.0 / max(page.shape[:2])
    if scale < 1.0:
        small = cv2.resize(page, None, fx=scale, fy=scale)
    else:
        small = page.copy()
        scale = 1.0

    cands = detect_black_squares(small, relaxed_validation=False)
    if len(cands) < 3:
        cands = detect_black_squares(small, relaxed_validation=True)

    grid = _extract_grid_from_markers(page, cands, scale)
    if grid is None:
        print("   [fallback] Markerii pe pagina rectificată nu au dat un warp valid.")
        return None

    print("   [fallback] Extragere reușită prin contur de pagină.")
    return grid

def trim_background(img_bgr):
    """
    Finds the white page within an image that might have dark background corners
    and stretches the image so the white page completely fills the frame.
    """
    H, W = img_bgr.shape[:2]
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, bw = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < (W * H) * 0.5:
        return img_bgr
        
    hull = cv2.convexHull(largest_contour)
    
    rect_corners = np.array([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H]
    ], dtype=np.float32)
    
    found_corners = []
    hull_pts = hull[:, 0, :]
    
    for corner in rect_corners:
        dists = np.sum((hull_pts - corner)**2, axis=1)
        closest_idx = np.argmin(dists)
        found_corners.append(hull_pts[closest_idx])
        
    found_corners = np.array(found_corners, dtype=np.float32)

    max_dist_to_corner = np.max(np.linalg.norm(found_corners - rect_corners, axis=1))
    if max_dist_to_corner < 20: 
        return img_bgr 

    M = cv2.getPerspectiveTransform(found_corners, rect_corners)
    trimmed = cv2.warpPerspective(img_bgr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return trimmed

def deskew_image(image_bgr):
    """
    Corectează înclinația fină (skew) a foii prin maximizarea varianței proiecției orizontale.
    Găsește exact unghiul la care rândurile de buline se aliniază perfect orizontal.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    best_angle = 0.0
    max_var = 0.0
    
    # Căutăm cel mai bun unghi între -3 și 3 grade (cu precizie de 0.1) 
    # pentru a corecta imperfecțiunile fine ale warp-ului
    angles = np.arange(-3.0, 3.1, 0.1)
    
    (h, w) = bw.shape
    center = (w // 2, h // 2)

    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_bw = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST)
        
        # Proiecția pe orizontală (rândurile)
        proj_y = np.sum(rotated_bw, axis=1)
        var = np.var(proj_y)
        
        if var > max_var:
            max_var = var
            best_angle = angle

    if abs(best_angle) > 0.05:
        # Deskew final cu unghiul optim
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        corrected = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return corrected, best_angle
    return image_bgr, 0.0

def main():
    input_dir = "detect-poze-test/"
    output_dir = "extracted_grids_only/"
    os.makedirs(output_dir, exist_ok=True)
    for entry in os.listdir(output_dir):
        p = os.path.join(output_dir, entry)
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)

    photos = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    photos.sort()

    print(f"==================================================")
    print(f"Extragere grile in: {output_dir}")
    print(f"==================================================")

    for photo_path in photos:
        name = os.path.splitext(os.path.basename(photo_path))[0]
        print(f"-> Procesez: {name}")

        img_full = cv2.imread(photo_path)
        if img_full is None:
            print(f"   ❌ Eroare: Nu se poate citi imaginea")
            continue

        # Detectare markeri pe imagine scalată
        scale = 2000 / max(img_full.shape[:2])
        if scale < 1.0:
            img_small = cv2.resize(img_full, None, fx=scale, fy=scale)
        else:
            img_small = img_full.copy()
            scale = 1.0

        candidates = detect_black_squares(img_small, relaxed_validation=False)
        if len(candidates) < 3:
            print(f"   [retry] Markeri insuficienți pe strict ({len(candidates)}), încerc relaxed.")
            candidates = detect_black_squares(img_small, relaxed_validation=True)

        warped = _extract_grid_from_markers(img_full, candidates, scale)
        if warped is None:
            print("   [retry] Warp din markeri invalid sau markeri insuficienți, încerc fallback pe contur pagină.")
            warped = _fallback_extract_via_page(name, img_full)
        if warped is None:
            print(f"   ❌ Nu am putut extrage robust grila pentru {name}.")
            continue
        
        # APLICĂM NORMALIZAREA CORECTA PE 4 LATURI
        normalized_warped = normalize_orientation(warped, pad_pct_x=0.16, pad_pct_y=0.04)
        
        # CORECTAM INCLINATIA FINA (Deskew pe baza continutului)
        deskewed_warped, angle_fixed = deskew_image(normalized_warped)
        if abs(angle_fixed) > 0.05:
            print(f"   📐 Aplicat corecție de înclinație (deskew): {angle_fixed:.2f} grade")
        
        # --- Curățare fundal (elimina colturi negre aduse de warp-ul prea generos) ---
        # Acesta lasa doar o foaie ALBA, dreapta, pastrand markerii si marginea extinsa.
        final_processed_image = trim_background(deskewed_warped)
        
        # Salvăm doar imaginea extrasă și binarizată
        out_path = os.path.join(output_dir, f"{name}_grid.jpg")
        cv2.imwrite(out_path, final_processed_image)
        print(f"   ✅ Grilă perfect curățată pe baza logicii anterioare salvată: {name}_grid.jpg")

    print(f"\nProces finalizat! Poti verifica folderul '{output_dir}'.")

if __name__ == "__main__":
    main()
