from PIL import Image
from numpy import asarray

img = Image.open('/home/domi/soft31/test-negru.png') # Load image
a = asarray(img)

print(type(a))
print(a.shape)     
# print(a)
