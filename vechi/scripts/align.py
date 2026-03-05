from pdf2image import convert_from_path
import cv2
import numpy as np

def pil_to_bgr(pil_img):
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def load_template_from_pdf(pdf_path, page_index=1, dpi=300):
    pages = convert_from_path(pdf_path, dpi)
    tpl_bgr = pil_to_bgr(pages[page_index])
    return tpl_bgr

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

# ======== Exemplu de utilizare ========

template = load_template_from_pdf("GrileRezidentiat/farm_C.pdf", page_index=1, dpi=300)  # pagina 2 din PDF :contentReference[oaicite:1]{index=1}

photo = cv2.imread("distorted-image.png")  # poza ta
aligned, H = align_photo_to_template(photo, template)

cv2.imwrite("aligned.png", aligned)
print("Salvat: aligned.png")
