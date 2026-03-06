import cv2
from pdf2image import convert_from_path
from precompute import pick_bars, turnGray, findShapes # pyright: ignore[reportImplicitRelativeImport, reportUnknownVariableType]

INPUT_DIR = "/home/domi/soft31/incercare/pozele-mele/"
OUTPUT_DIR = "output/"
file_path = "IMG_0545.jpg"

image = cv2.imread(INPUT_DIR + file_path)
gray_image = turnGray(image)
shapes_img, contours = findShapes(gray_image)
pick_bars(shapes_img, contours) 
