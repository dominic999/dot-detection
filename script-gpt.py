import cv2
from pdf2image import convert_from_path
from typing import cast
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
def to_uint8(arr: NDArray[np.generic]) -> Image:
  if arr.dtype == np.uint8:
      return cast(Image, arr)
  return arr.astype(np.uint8, copy=False)

def binarize(img) -> NDArray[np.uint8]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # adaptive e bun la iluminare neuniformă
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 35, 10)
    return bw

def getColumn(img, col_number):
    

def extract_answers(
    roi: NDArray[np.generic],
    n_questions: int = 200,
    n_choices: int = 5,
    top_margin: float = 0.06,
    bottom_margin: float = 0.04,
    left_margin: float = 0.08,
    right_margin: float = 0.06,
    bubble_pad: float = 0.15,
    fill_thresh: float = 0.18,
    ambiguous_delta: float = 0.06,
) -> list[str | None]:
    """
    roi: imaginea deja aliniată care conține DOAR grila cu elipse
    marginile sunt proporții din ROI (le ajustezi 1-2 minute și apoi rămân fixe)
    """
    bw = binarize(roi)
    h, w = bw.shape

    y0 = int(h * top_margin)
    y1 = int(h * (1 - bottom_margin))
    x0 = int(w * left_margin)
    x1 = int(w * (1 - right_margin))

    grid = bw[y0:y1, x0:x1]
    gh, gw = grid.shape

    row_h = gh / n_questions
    col_w = gw / n_choices

    answers = []
    for i in range(n_questions):
        ratios = []
        for j in range(n_choices):
            cy0 = int(i * row_h)
            cy1 = int((i + 1) * row_h)
            cx0 = int(j * col_w)
            cx1 = int((j + 1) * col_w)

            cell = grid[cy0:cy1, cx0:cx1]

            # "taie" din margini ca să nu numeri conturul elipsei prea mult
            pad_y = int(cell.shape[0] * bubble_pad)
            pad_x = int(cell.shape[1] * bubble_pad)
            inner = cell[pad_y:cell.shape[0]-pad_y, pad_x:cell.shape[1]-pad_x]
            if inner.size == 0:
                ratios.append(0.0)
                continue

            fill_ratio = float(np.count_nonzero(inner)) / float(inner.size)
            ratios.append(fill_ratio)

        best = int(np.argmax(ratios))
        best_val = ratios[best]
        sorted_vals = sorted(ratios, reverse=True)
        second = sorted_vals[1] if len(sorted_vals) > 1 else 0.0

        if best_val < fill_thresh:
            answers.append(None)  # necompletat
        elif (best_val - second) < ambiguous_delta:
            answers.append("AMBIG")  # posibil 2 bifări / murdărie
        else:
            answers.append("ABCDE"[best])

    return answers

pages = convert_from_path('GrileRezidentiat/farm_C.pdf', 300)
img = cv2.imread("page_2.png")
img = np.array(img)
h, w = img.shape[:2]
x, x2 = int(0.55 * w), int(0.98 * w)  # coloana din dreapta (ajustezi)
y, y2 = int(0.05 * h), int(0.95 * h)
#
roi: NDArray[np.uint8] = img[y:y2, x:x2]  

# aici bagi crop-ul pe coloana din dreapta
# print(roi);
answers = extract_answers(roi)
print(answers[:20])
