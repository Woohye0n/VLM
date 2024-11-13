import cv2

img = cv2.imread('/home/aidas_intern_1/woohyeon/llava1.6/val2017/000000095069.jpg')
if img is not None:
    cv2.imwrite('/home/aidas_intern_1/woohyeon/llava1.6/val2017/000000095069_fixed.jpg', img)
    print("Image saved as fixed version.")
else:
    print("Image is corrupted or in an unsupported format.")
    
from PIL import Image

try:
    img_fixed = Image.open('/home/aidas_intern_1/woohyeon/llava1.6/val2017/000000095069_fixed.jpg')
    img_fixed.show()
except (IOError, PIL.UnidentifiedImageError) as e:
    print(f"Cannot open fixed image: {e}")