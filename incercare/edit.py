import cv2
import numpy as np

def warp_crop_from_quad(img, quad):
    """
    img: imagine BGR sau Gray
    quad: (4,2) float32 în ordine TL,TR,BR,BL
    Return: imaginea decupată și îndreptată (warp perspective)
    """
    quad = np.asarray(quad, dtype=np.float32)
    (tl, tr, br, bl) = quad

    # lățime = max(dist TR-TL, BR-BL)
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    # înălțime = max(dist BL-TL, BR-TR)
    heightA = np.linalg.norm(bl - tl)
    heightB = np.linalg.norm(br - tr)
    maxH = int(max(heightA, heightB))

    # destinația: dreptunghi perfect
    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH))

    return warped
