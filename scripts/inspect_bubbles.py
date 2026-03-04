from pdf2image import convert_from_path
import cv2
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

def print_ratios(idx, title):
    print(f"\n=== {title} ===")
    x0, y0, x1, y1 = boxes[idx]
    roi = img[y0:y1, x0:x1]
    
    bubbles, _edges = script3.detect_bubbles(roi)
    bw = script3.preprocess_bw_for_bubbles(roi)
    bubbles.sort(key=lambda b: (b["cy"], b["cx"]))
    axes_std = script3.compute_standard_axes(bubbles, shrink=0.90)
    
    rows = script3.group_bubbles_into_20_rows(bubbles, n_rows=20)
    
    for r_idx, row in enumerate(rows[:5]): # only first 5 rows for brevity
        row = sorted(row, key=lambda b: b["cx"])[:5]
        
        info = []
        for j, b in enumerate(row):
            ratio = script3.fill_ratio_standard(bw, b, axes_std)
            info.append(f"{ratio:.2f}")
        print(f"Row {r_idx+1}: {', '.join(info)}")

print_ratios(0, "Box 0 (Questions 1-5)")
print_ratios(5, "Box 5 (Questions 101-105)")
