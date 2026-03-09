import cv2
import numpy as np

def count_markers_strict(strip, side_name="DEBUG"):
    """
    V4: Detectează strict markerii de orientare (pătrate sau bare negre).
    Folosește binarizare deja aplicată (negativ).
    Filtrează agresiv orice are solidity < 0.92 (bulele au ~0.8).
    """
    if strip.size == 0:
        return 0
    
    cnts, _ = cv2.findContours(strip.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = 0
    
    for c in cnts:
        area = cv2.contourArea(c)
        # Markerii sunt mari (pătrate ~1000px, bare ~3000px la 2000px rezoluție)
        if 250 < area < 40000:
            x, y, w, h = cv2.boundingRect(c)
            ar = float(w) / (h + 1e-5)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-5)

            # 1. Bară (AR extrem) - Solidity poate fi puțin mai mic dacă e zgomotoasă
            if ar > 3.5 or ar < 0.28:
                if solidity > 0.85:
                    valid += 1
                continue
            
            # 2. Pătrat solid (Solidity FOARTE mare!)
            if solidity > 0.92:
                # Verificăm aspect ratio să fie ~pătrat
                if 0.5 < ar < 2.0:
                    valid += 1
                    
    return valid

def normalize_orientation(img_bgr, pad_pct_x=0.16, pad_pct_y=0.04):
    """
    Rotește grila warpată punând TOP (3 markeri) sus și BOTTOM (4 markeri) jos.
    V4: Super-strict markers + priority for (3,4) axis + smaller strips.
    """
    H, W = img_bgr.shape[:2]
    
    # 1. Binarizare pe elemente NEGRE (threshold fix este mai sigur pentru markeri)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Folosim threshold fix fiindcă markerii sunt negru pur
    _, bw = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
    
    # Grosime fâșii (subțiri pt a evita bulele, cca 4%)
    th_x = int(W * 0.045)
    th_y = int(H * 0.045)
    
    top_strip    = bw[0 : th_y, 0 : W]
    bot_strip    = bw[H - th_y : H, 0 : W]
    left_strip   = bw[0 : H, 0 : th_x]
    right_strip  = bw[0 : H, W - th_x : W]
    
    counts = {
        "TOP": count_markers_strict(top_strip, "TOP"),
        "BOTTOM": count_markers_strict(bot_strip, "BOTTOM"),
        "LEFT": count_markers_strict(left_strip, "LEFT"),
        "RIGHT": count_markers_strict(right_strip, "RIGHT")
    }
    
    print(f"  Orientation signature: {counts}")

    def calculate_axis_score(c1, c2):
        # Penalizare totală pentru bule (>10)
        if c1 > 10 or c2 > 10: return -1000
        # Semnătură perfectă (3, 4) sau (4, 3)
        if (c1 == 3 and c2 == 4) or (c1 == 4 and c2 == 3): return 5000
        # Aproximație bună (2-5, 3-6)
        if (2 <= c1 <= 5) and (3 <= c2 <= 6): return 1000 + (c1 + c2)
        return c1 + c2

    v_score = calculate_axis_score(counts["TOP"], counts["BOTTOM"])
    h_score = calculate_axis_score(counts["LEFT"], counts["RIGHT"])
    
    # Căutăm axa cu cel mai mare scor (validă)
    if v_score < 0 and h_score < 0:
        print("  [SKIP] No clear marker axis detected.")
        return img_bgr

    rotate_code = None
    if v_score >= h_score:
        # Axa verticală (Portrait)
        # Vrem 3 SUS (TOP), 4 JOS (BOTTOM).
        # Deci dacă TOP > BOTTOM, rotim.
        if counts["TOP"] > counts["BOTTOM"]:
            rotate_code = cv2.ROTATE_180
    else:
        # Axa orizontală (Landscape)
        # Vrem ca latura cu 3 markeri să ajungă TOP.
        if counts["LEFT"] < counts["RIGHT"]:
            # 3 în stânga -> Stânga devine sus (90 CCW)
            rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            # 3 în dreapta -> Dreapta devine sus (90 CW)
            rotate_code = cv2.ROTATE_90_CLOCKWISE
            
    if rotate_code is not None:
        return cv2.rotate(img_bgr, rotate_code)
    return img_bgr

if __name__ == "__main__":
    pass
