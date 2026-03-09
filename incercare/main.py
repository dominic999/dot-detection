import cv2
from pdf2image import convert_from_path
from precompute import pick_bars, turnGray, findShapes # pyright: ignore[reportImplicitRelativeImport, reportUnknownVariableType]
from boxes import save_10_boxes #pyright: ignore
from bubbles import find_bubbles #pyright: ignore

INPUT_DIR = "/home/domi/soft31/incercare/pozele-mele/"
OUTPUT_DIR = "/home/domi/soft31/incercare/output/"
file_path = "IMG_0545.jpg"

image = cv2.imread(INPUT_DIR + file_path)
gray_image = turnGray(image)
shapes_img, contours = findShapes(gray_image)
points, cropped_image = pick_bars(shapes_img, contours) 
boxes = save_10_boxes(cropped_image)

for i in range(10):
    path = OUTPUT_DIR + "box_" + str(i) + ".png"
    print("path: " , path)
    img = cv2.imread(OUTPUT_DIR + "box_" + str(i) + ".png")
    find_bubbles(img, i)


