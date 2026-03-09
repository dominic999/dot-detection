import cv2
import numpy as np

from boxes import get_10_boxes_from_warp as base_get_10_boxes_from_warp  # pyright: ignore
from helpers import save  # pyright: ignore


def get_10_boxes_from_warp(
    warp_bgr,
    top_crop=0.05,
    bottom_crop=0.90,
    left_crop=0.05,
    right_crop=1,
    pad_x=10,
    pad_y=10,
):
    """
    Decupeaza mai intai imaginea mare si apoi ruleaza algoritmul original din boxes.py
    pe ROI-ul rezultat. Parametrii crop afecteaza doar ROI-ul initial:
    - top_crop: cat se taie din sus
    - bottom_crop: cat se taie din jos
    - left_crop: cat se taie din stanga
    - right_crop: cat se taie din dreapta
    """
    if warp_bgr is None or warp_bgr.size == 0:
        raise RuntimeError("warp_bgr este gol.")

    img = warp_bgr
    if img.dtype != np.uint8:
        imgf = img.astype(np.float32)
        if imgf.max() <= 1.5:
            imgf *= 255.0
        img = np.clip(imgf, 0, 255).astype(np.uint8)

    H, W = img.shape[:2]

    y0 = int(H*top_crop)
    y1 = int(H*bottom_crop)
    x0 = int(W*left_crop)
    x1 = int(W*right_crop)

    print(H, top_crop, bottom_crop)
    print(y0, y1, x0, x1)

    cropped = img[y0:y1, x0:x1]
    if cropped.size == 0:
        raise RuntimeError("Crop-ul calculat este gol.")

    boxes_local, xb_local, y_cut_local = base_get_10_boxes_from_warp(
        cropped
    )

    boxes = []
    for bx0, by0, bx1, by1 in boxes_local:
        boxes.append((bx0 + x0, by0 + y0, bx1 + x0, by1 + y0))

    xb = [x + x0 for x in xb_local]
    y_cut = y_cut_local + y0
    return boxes, xb, y_cut


def save_10_boxes(
    warp_bgr,
    out_dir="output_boxes",
    top_crop=0.05,
    bottom_crop=0.90,
    left_crop=0.05,
    right_crop=0.10,
    pad_x=5,
    pad_y=8,
):
    boxes, xb, y_cut = get_10_boxes_from_warp(
        warp_bgr
    )

    dbg = warp_bgr.copy()
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            dbg,
            str(i),
            (x0 + 10, y0 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
    cv2.line(dbg, (0, int(y_cut)), (warp_bgr.shape[1] - 1, int(y_cut)), (255, 0, 0), 2)
    save("debug_boxes.png", dbg)

    for i, (x0, y0, x1, y1) in enumerate(boxes):
        roi = warp_bgr[y0:y1, x0:x1]
        save(f"box_{i}.png", roi)

    return boxes
