from pdf2image import convert_from_path
import importlib.util
import sys

# Load 3script.py module
spec = importlib.util.spec_from_file_location("script3", "3script.py")
script3 = importlib.util.module_from_spec(spec)
sys.modules["script3"] = script3
spec.loader.exec_module(script3)

pages = convert_from_path("GrileRezidentiat/farm_C_editat.pdf", 300)
img = script3.pil_to_bgr(pages[1])
boxes, _ = script3.get_10_boxes_fixedBands(img)

x0, y0, x1, y1 = boxes[0]
roi = img[y0:y1, x0:x1]
bubbles, _edges = script3.detect_bubbles(roi)
rows = script3.group_bubbles_into_20_rows(bubbles, n_rows=20)

for r_idx, row in enumerate(rows[:5]):
    row = sorted(row, key=lambda b: b["cx"])[:5]
    print(f"Row {r_idx+1} cx: {[b['cx'] for b in row]}")

x0, y0, x1, y1 = boxes[5]
roi = img[y0:y1, x0:x1]
bubbles, _edges = script3.detect_bubbles(roi)
rows = script3.group_bubbles_into_20_rows(bubbles, n_rows=20)

print("--- Box 5 ---")
for r_idx, row in enumerate(rows[:5]):
    row = sorted(row, key=lambda b: b["cx"])[:5]
    print(f"Row {r_idx+101} cx: {[b['cx'] for b in row]}")
