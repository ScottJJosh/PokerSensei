import cv2
import os

file_name = os.path.join(os.path.dirname(__file__), '/test/images/6_jpg.rf.0f0fe570e17badaab037f36ec3be1c0d.jpg')
image = cv2.imread(file_name)

print(image.shape)