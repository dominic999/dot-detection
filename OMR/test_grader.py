import cv2
from imutils import contours
import argparse
import numpy as np
from four_point_transform import four_point_transform
import imutils
import os

def increase_contrast(img_bgr, method="clahe", strength=2.5, tile=8):
    """
    Crește contrastul unei imagini BGR.

    method:
      - "clahe"  : contrast local (recomandat pentru poze de telefon)
      - "linear" : contrast global (alpha/beta)

    strength:
      - clahe: clipLimit (ex 1.5 .. 4.0)
      - linear: alpha (ex 1.2 .. 2.0)

    tile:
      - clahe: tileGridSize (ex 6..12)

    Returnează imagine BGR.
    """
    if method == "clahe":
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=float(strength), tileGridSize=(int(tile), int(tile)))
        L2 = clahe.apply(L)

        lab2 = cv2.merge([L2, A, B])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    elif method == "linear":
        alpha = float(strength)  # contrast
        beta = 0                 # luminozitate (poți pune ex 10/-10)
        return cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

    else:
        raise ValueError("method trebuie să fie 'clahe' sau 'linear'.")

# create argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="path to the image")
ap.add_argument('-c', '--crop', type=int, help="whether to crop image 1 if yes 0 if no")
ap.add_argument('-o', '--output', default="output", help="output folder for debug images")
args = vars(ap.parse_args())

os.makedirs(args["output"], exist_ok=True)

def save(name, img):
    path = os.path.join(args["output"], name)
    ok = cv2.imwrite(path, img)
    if not ok:
        print(f"[WARN] Nu am putut salva: {path}")
    else:
        print(f"[SAVED] {path}")

# define the answer key (zero-based)
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

# preprocess the image
if args["crop"] == 1:
    image = cv2.imread(args["image"])
    if image is None:
        raise RuntimeError(f"Nu pot citi imaginea: {args['image']}")
    image = imutils.resize(image, width=700)
else:
    image = cv2.imread(args["image"])
    if image is None:
        raise RuntimeError(f"Nu pot citi imaginea: {args['image']}")

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8,8))
gray2 = clahe.apply(gray)
# blurred = cv2.GaussianBlur(gray2, (11, 11), 10)
blurred = cv2.bilateralFilter(gray2, 9, 75, 75)
save("00_gri_edged.png", gray)
save("00_gri2_edged.png", gray2)
save("gassioan.png", blurred)
ret, inverted = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
save("inverted.png", inverted)
edged = cv2.Canny(gray, 0, 500)

save("01_preprocessed_edged.png", edged)

# find contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# contour of the document i.e. answer sheet
docCnt = None
paper_contours = image.copy()

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for cnt in cnts:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        if len(approx) == 4:
            docCnt = approx
            cv2.drawContours(paper_contours, [cnt], -1, (0, 0, 255), 3)
            break

save("02_paper_contours_detected.png", paper_contours)

if docCnt is None:
    save("ERROR_no_doc_contour_edged.png", edged)
    raise RuntimeError("Nu am găsit conturul foii (4 colțuri). Verifică 01_preprocessed_edged.png")

paper = four_point_transform(image, docCnt.reshape((4, 2)))
warped = four_point_transform(gray, docCnt.reshape((4, 2)))

save("03_paper_extracted_color.png", paper)
save("04_paper_extracted_gray.png", warped)

thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
save("05_threshold.png", thresh)

# after thresholding image, find bubbles
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

questionContours = []

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    aspect_ratio = w / float(h + 1e-6)

    if w >= 20 and h >= 20 and 0.9 <= aspect_ratio <= 1.1:
        questionContours.append(c)

if len(questionContours) == 0:
    save("ERROR_debug_thresh.png", thresh)
    save("ERROR_debug_warped.png", warped)
    print("[ERROR] Nu am detectat nicio bulina. Verifică imaginile ERROR_*.png")
    raise SystemExit(1)

# draw detected bubbles (all)
paper_bubbles = paper.copy()
for cnt in questionContours:
    cv2.drawContours(paper_bubbles, [cnt], 0, (0, 0, 255), 3)

save("06_detected_bubbles_all.png", paper_bubbles)

question_cnts = contours.sort_contours(questionContours, method='top-to-bottom')[0]

correct = 0
graded = paper.copy()

for (q, i) in enumerate(np.arange(0, len(question_cnts), 5)):
    cnts_row = contours.sort_contours(question_cnts[i: i + 5])[0]
    bubbled = None

    for (j, c) in enumerate(cnts_row):
        m = np.zeros(thresh.shape, dtype='uint8')
        cv2.drawContours(m, [c], -1, 255, -1)

        masked = cv2.bitwise_and(thresh, thresh, mask=m)
        total = cv2.countNonZero(masked)

        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    color = (0, 0, 255)
    k = ANSWER_KEY.get(q, None)

    if k is not None and k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1

    if k is not None:
        cv2.drawContours(graded, [cnts_row[k]], -1, color, 3)

    # debug per question (optional): uncomment if you want one image per question
    # save(f"q_{q:03d}.png", graded)

print(correct)
score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))

cv2.putText(graded, "{:.2f}%".format(score), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

save("07_final_graded.png", graded)
