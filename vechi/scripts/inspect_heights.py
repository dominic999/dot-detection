import cv2
import numpy as np
from pdf2image import convert_from_path
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("script3", "3script.py")
script3 = importlib.util.module_from_spec(spec)
sys.modules["script3"] = script3
spec.loader.exec_module(script3)

pages = convert_from_path("GrileRezidentiat/farm_C_editat.pdf", 300)
img = script3.pil_to_bgr(pages[1])
boxes, dbg = script3.get_10_boxes_fixedBands(img)

print(f"Old cut_top was {int(img.shape[0] * 0.08)}")
print(f"New cut_top is {dbg['cut_top']}")

# Check what bubbles are detected
x0, y0, x1, y1 = boxes[0]
roi = img[y0:y1, x0:x1]
bubbles, _ = script3.detect_bubbles(roi)
rows = script3.group_bubbles_into_20_rows(bubbles, n_rows=20)

for i, row in enumerate(rows[:5]):
    cy_avg = np.mean([b["cy"] for b in row])
    print(f"Row {i} cy_avg: {cy_avg:.1f}, count: {len(row)}")
