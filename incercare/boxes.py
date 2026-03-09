import cv2
import numpy as np
from helpers import save # pyright: ignore[]

def smooth_1d(arr, k=51):
    k = k if k % 2 == 1 else k + 1
    return cv2.GaussianBlur(arr.astype(np.float32).reshape(-1,1), (1,k), 0).ravel()

def refine_equal_boundaries(proj, n_sections, search_frac=0.10):
    """
    Delimitări robuste:
    1) împărțire uniformă pe lățime
    2) fiecare graniță internă este rafinată local la cea mai mică valoare din proiecție
    """
    p = proj.astype(np.float32)
    W = len(p)
    if W <= 1:
        raise RuntimeError("Proiecție invalidă pe axa X.")

    # Netezire suplimentară pentru stabilitate la zgomot/pete
    k = max(51, (W // 20) | 1)  # impar
    ps = smooth_1d(p, k=k)

    base = [int(round(i * (W - 1) / n_sections)) for i in range(n_sections + 1)]
    refined = [base[0]]

    step = max(1, (W - 1) // n_sections)
    win = max(8, int(step * search_frac))
    for i in range(1, n_sections):
        t = base[i]
        a = max(1, t - win)
        b = min(W - 2, t + win)
        if b <= a:
            refined.append(t)
            continue
        valley = a + int(np.argmin(ps[a:b + 1]))
        refined.append(valley)

    refined.append(base[-1])

    # Monotonicitate strictă, evită box-uri cu lățime 0.
    for i in range(1, len(refined)):
        if refined[i] <= refined[i - 1]:
            refined[i] = refined[i - 1] + 1
    refined[-1] = min(refined[-1], W - 1)
    return refined

def find_valleys_between_peaks(proj, n_sections, mask_thr=0.25):
    """
    proj: vector 1D
    n_sections: 5 pentru coloane
    întoarce boundaries: listă de lungime n_sections+1
    """
    p = proj.astype(np.float32)
    p = (p - p.min()) / (p.max() - p.min() + 1e-6)

    mask = p > mask_thr
    idx = np.where(mask)[0]
    if len(idx) == 0:
        # Fallback robust: dacă proiecția e prea plată, împărțim uniform.
        w = len(p)
        if w <= 1:
            raise RuntimeError("Nu găsesc conținut în proiecție.")
        return [int(round(i * (w - 1) / n_sections)) for i in range(n_sections + 1)]

    start = int(idx[0])
    end = int(idx[-1])

    cuts = [start]
    for i in range(1, n_sections):
        target = int(start + (end - start) * i / n_sections)
        win = max(30, (end - start)//(n_sections*6))
        a = max(start, target - win)
        b = min(end,   target + win)
        valley = a + int(np.argmin(p[a:b+1]))
        cuts.append(valley)
    cuts.append(end)
    return cuts

def find_best_horizontal_gap(proj_y, mid_frac=0.52, band=0.20):
    """
    Găsește cea mai mare zonă "goală" în jurul mijlocului (split sus/jos).
    Returnează y_cut.
    """
    p = proj_y.astype(np.float32)
    H = len(p)
    if H <= 2:
        return max(0, H // 2)

    # Netezire puternică pe Y: elimină efectul textului/punctelor locale.
    k = max(101, (H // 10) | 1)  # impar
    ps = smooth_1d(p, k=k)
    ps = (ps - ps.min()) / (ps.max() - ps.min() + 1e-6)

    mid = int(H * mid_frac)
    a = max(1, int(mid - H * band))
    b = min(H - 2, int(mid + H * band))
    if b <= a:
        return mid

    win = ps[a:b + 1]
    center = mid - a
    # Cost compus: vale mică + apropiere de mijlocul benzii căutate.
    dist = np.abs(np.arange(len(win), dtype=np.float32) - float(center))
    dist /= (len(win) / 2.0 + 1e-6)
    score = win + 0.35 * dist
    idx_local = int(np.argmin(score))

    # Dacă există un platou în jurul minimului, alegem centrul platoului.
    minv = win[idx_local]
    eps = 0.02
    left = idx_local
    right = idx_local
    while left > 0 and win[left - 1] <= minv + eps:
        left -= 1
    while right < len(win) - 1 and win[right + 1] <= minv + eps:
        right += 1
    return a + (left + right) // 2

def get_10_boxes_from_warp(warp_bgr,
                           top_crop=0.06, bottom_crop=0.90,
                           pad_x=10, pad_y=10):
    """
    warp_bgr: imaginea ta mare (îndreptată) care conține doar grila.
    Returnează 10 box-uri (x0,y0,x1,y1) în coordonate warp.
    """
    if warp_bgr is None or warp_bgr.size == 0:
        raise RuntimeError("warp_bgr este gol.")

    img = warp_bgr
    if img.dtype != np.uint8:
        imgf = img.astype(np.float32)
        # Dacă datele sunt normalizate în [0,1], le aducem în [0,255].
        if imgf.max() <= 1.5:
            imgf *= 255.0
        img = np.clip(imgf, 0, 255).astype(np.uint8)

    H, W = img.shape[:2]
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # binarizare robustă pentru imagini cu iluminare/contrast variabil.
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Elimină puncte mici (noise, pixeli izolați) care destabilizează proiecțiile.
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    # ignoră barele negre margini sus/jos ca să nu strice proiecțiile
    yA = int(H * top_crop)
    yB = int(H * bottom_crop)
    bw_mid = bw[yA:yB, :]

    # Delimitări verticale:
    # - căutăm doar în [0.001W, 0.99W]
    # - împărțim în 5 zone egale
    # - pentru fiecare separator intern alegem coloana cu cei mai mulți pixeli negri
    xL = int(0.001 * W)
    xR = int(0.99 * W)
    if xR <= xL + 5:
        xL, xR = 0, W - 1

    gray_mid = gray[yA:yB, :]
    otsu_thr_x = cv2.threshold(gray_mid, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    black_x = (gray_mid < otsu_thr_x).sum(axis=0).astype(np.float32)
    black_x = smooth_1d(black_x, 81)

    span = xR - xL
    step = span / 5.0
    x_inner = []
    for i in range(1, 5):
        target = xL + i * step
        a = int(max(xL + 1, round(target - 0.5 * step)))
        b = int(min(xR - 1, round(target + 0.5 * step)))
        if b <= a:
            x_inner.append(int(round(target)))
            continue
        # alegem coloana cu maxim de pixeli negri în zona separatorului
        x_best = a + int(np.argmax(black_x[a:b + 1]))
        x_inner.append(x_best)

    xb = [xL] + x_inner + [xR]
    for i in range(1, len(xb)):
        if xb[i] <= xb[i - 1]:
            xb[i] = xb[i - 1] + 1
    xb[-1] = min(xb[-1], W - 1)

    # Proiecție Y doar din zona de bule (interiorul fiecărei coloane),
    # evităm marginile cu numere și chenare.
    strips_bw = []
    strips_gray = []
    for c in range(5):
        x0 = int(xb[c])
        x1 = int(xb[c + 1])
        w = max(1, x1 - x0)
        xi0 = x0 + int(0.28 * w)
        xi1 = x0 + int(0.82 * w)
        if xi1 > xi0:
            strips_bw.append(bw_mid[:, xi0:xi1])
            strips_gray.append(gray_mid[:, xi0:xi1])
    bw_for_y = np.concatenate(strips_bw, axis=1) if strips_bw else bw_mid
    gray_for_y = np.concatenate(strips_gray, axis=1) if strips_gray else gray_mid

    # Cerință: căutăm între 45%-55% din înălțimea utilă și alegem rândul
    # cu cei mai mulți pixeli negri.
    a = int(0.45 * bw_for_y.shape[0])
    b = int(0.55 * bw_for_y.shape[0])
    if b <= a:
        a, b = 0, bw_for_y.shape[0]

    otsu_thr = cv2.threshold(gray_for_y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    black_counts = (gray_for_y < otsu_thr).sum(axis=1)
    black_win = black_counts[a:b]
    if black_win.size == 0:
        proj_y = smooth_1d(bw_for_y.sum(axis=1), 121)
        y_cut_mid = find_best_horizontal_gap(proj_y, mid_frac=0.50, band=0.05)
    else:
        y_cut_mid = a + int(np.argmax(black_win))
    y_cut = yA + y_cut_mid

    # două benzi: sus + jos
    bands = [
        (0, y_cut),
        (y_cut, H)
    ]

    boxes = []
    for (yy0, yy1) in bands:
        y0 = max(0, yy0 - pad_y)
        y1 = min(H, yy1 + pad_y)
        for c in range(5):
            x0 = max(0, xb[c] - pad_x)
            x1 = min(W, xb[c+1] + pad_x)
            boxes.append((x0, y0, x1, y1))

    return boxes, xb, y_cut

def save_10_boxes(warp_bgr, out_dir="output_boxes"):

    boxes, xb, y_cut = get_10_boxes_from_warp(warp_bgr, pad_x=5, pad_y=8)

    # debug desen
    dbg = warp_bgr.copy()
    for i, (x0,y0,x1,y1) in enumerate(boxes):
        cv2.rectangle(dbg, (x0,y0), (x1,y1), (0,255,0), 2)
        cv2.putText(dbg, str(i), (x0+10, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    cv2.line(dbg, (0, int(y_cut)), (warp_bgr.shape[1]-1, int(y_cut)), (255,0,0), 2)
    save("debug_boxes.png", dbg)


    # salvează crop-urile
    for i, (x0,y0,x1,y1) in enumerate(boxes):
        roi = warp_bgr[y0+100:y1-30, x0+32:x1-100]
        save(f"box_{i}.png", roi)


    return boxes
