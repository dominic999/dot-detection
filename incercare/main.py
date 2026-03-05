import cv2
from pdf2image import convert_from_path
# from precompute import printCeva  
from precompute import turnGray, findShapes

INPUT_DIR = "/home/domi/soft31/incercare/pozele-mele/"
OUTPUT_DIR = "output/"
file_path = "IMG_3611.jpg"

image = cv2.imread(INPUT_DIR + file_path)
gray_image = turnGray(image)
findShapes(gray_image)
