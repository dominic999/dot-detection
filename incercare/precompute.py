import cv2
import numpy as np
from helpers import save # pyright: ignore[reportImplicitRelativeImport, reportUnknownVariableType]


def turnGray(img):
    #NOTE aici fac imaginea gri
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save("imagine_gri.png", gray_image)
    # gray_f = cv2.medianBlur(gray_image, 7)
    gray_f = cv2.bilateralFilter(gray_image, d=21, sigmaColor=50, sigmaSpace=50)
    save("imagine_blurata.png", gray_f)
    # save("imagine_blurata_2.png", gray_f_2)
    return gray_f 

def findShapes(img):
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    save("reversed.png", thresh)
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return img, contours 

def pick_bars(img, contours):

    H, W = img.shape[:2]
    bar_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    min_area = 3000
    kept = []
    aspect_min = 3.0         # bară lungă
    rect_score_min = 0.90    # cât de "dreptunghi plin" e
    mean_gray_max = 120      # cât de negru e în interior (mai mic=mai negru)
    max_h_ratio = 0.2       # bara nu trebuie să fie foarte înaltă

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if area < min_area:
            continue

        rect = cv2.minAreaRect(contour)          # ((cx,cy),(rw,rh),angle)
        (cx, cy), (rw, rh), angle = rect
        if rw < 1 or rh < 1:
            continue

        long_side = max(rw, rh)
        short_side = min(rw, rh)
        aspect = long_side / (short_side + 1e-6)

        if aspect < aspect_min:
            continue
        if short_side > max_h_ratio * H:   # prea gros -> nu e bară
            continue
        if long_side > 0.3 * W:
            continue

        # rectangularity: cât de bine umple conturul dreptunghiul
        rect_area = rw * rh
        rect_score = float(area) / float(rect_area + 1e-6)
        if rect_score < rect_score_min:
            continue
        #
        # cât de negru e în interiorul conturului (nu bbox!)
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        vals = img[mask == 255]
        if vals.size == 0:
            continue
        mean_gray = float(np.mean(vals))
        if mean_gray > mean_gray_max:
            continue

        kept.append((contour, area, aspect, rect_score, mean_gray, rect))

        cv2.drawContours(bar_img, [contour], -1, (0, 255, 0), 2)
        cv2.putText(bar_img, "Rectangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,0))
        cv2.putText(bar_img, str(area), (x-50, y-50), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,0))


    save("contours.png", bar_img)

    if kept != 4:
        print("POZA NU ESTE BUNA!")

    return kept
