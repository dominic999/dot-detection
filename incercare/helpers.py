import os
import cv2

def save(name, img):
    path = os.path.join("/home/domi/soft31/incercare/output/", name)
    ok = cv2.imwrite(path, img)
    if not ok:
        print(f"[WARN] Nu am putut salva: {path}")
    else:
        print(f"[SAVED] {path}")
