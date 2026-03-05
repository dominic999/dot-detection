import cv2
import numpy as np

def only_very_black_to_white_norm_mask(img_bgr, t=70):
    img = img_bgr.astype(np.int16)
    dist = np.sqrt((img[:,:,0]**2 + img[:,:,1]**2 + img[:,:,2]**2).astype(np.float32))
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    mask[dist < t] = 255
    return mask

def cluster_white_regions(mask, connect=8, min_area=600, 
                          merge_kernel=7, max_area=40000):
    mask2 = mask.copy()

    # (opțional) scoate puncte izolate înainte să lipești
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33,3))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, k3, iterations=1)

    if merge_kernel and merge_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_kernel, merge_kernel))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, k, iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask2, connectivity=connect)

    clusters = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        cx, cy = centroids[i]
        clusters.append({
            "id": i,
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": int(area),
            "centroid": (float(cx), float(cy))
        })

    clusters.sort(key=lambda c: -c["area"])
    return clusters, labels, mask2

input_dir = "/home/domi/soft31/detect-poze-test/"
output_dir = "/home/domi/soft31/detect-poze-output/"

img = cv2.imread(input_dir + "IMG_0537.jpg")

mask = only_very_black_to_white_norm_mask(img, t=50)
cv2.imwrite(output_dir + "only_very_black.png", mask)

clusters, labels, mask2 = cluster_white_regions(mask, connect=8, 
                                                min_area=5000, 
                                                merge_kernel=3)

dbg = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
for c in clusters:
    x,y,w,h = c["bbox"]
    cv2.rectangle(dbg, (x,y), (x+w, y+h), (0,255,0), 5)
    cv2.putText(dbg, str(c["area"]), (x, max(0,y-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 2)

cv2.imwrite(output_dir + "clusters.png", dbg)
print("nr clustere:", len(clusters))
print("top 10 areas:", [c["area"] for c in clusters[:10]])
