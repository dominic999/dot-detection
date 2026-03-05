import cv2
from helpers import save

def turnGray(img):
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
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    print(len(contours));

    for contour in contours:
        # cv2.drawContours(out, contour, -1, (0, 255, 0), 2)
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if len(approx) == 4:
            cv2.drawContours(out, contour, -1, (0, 255, 0), 2)
            cv2.putText(out, "Rectangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
        else:
            # cv2.drawContours(out, contour, -1, (255, 0, 0), 2)
            # cv2.putText(out, "nr-laturi: " + str(len(approx)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
            continue
    save("contours.png", out)
    return out 

