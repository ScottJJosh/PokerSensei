import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
#import os


cap = cv2.VideoCapture(0)
cap.set(3, 1000)
cap.set(4, 1000)



if cap.isOpened():
        # Load image, grayscale, blur, Otsu's threshold
        # file_name = os.path.join(os.path.dirname(__file__), 'cards.jpg')
        # assert os.path.exists(file_name)
            while True:
                ret, image = cap.read()
                frame = image
                cv2.imshow('WebCam', frame)
                image = imutils.resize(image, width=500)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                # Find contours and filter for cards using contour area
                cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                threshold_min_area = 400
                threshold_max_area = 200000
                number_of_contours = 0
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area > threshold_min_area and area < threshold_max_area:
                        #cv2.drawContours(image, [c], 0, (36,255,12), 3)
                        x,y,w,h = cv2.boundingRect(c)
                        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 255), 3)
                        number_of_contours += 1

                print("Contours detected:", number_of_contours)
                #cv2.imshow('thresh', thresh)
                cv2.imshow('WebCam', image)
                if cv2.waitKey(1) == ord('q'): 
                    break

cap.release() 
cv2.destroyAllWindows() 