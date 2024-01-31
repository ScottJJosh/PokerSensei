from ultralytics import YOLO
import cv2
import math 
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1000)
cap.set(4, 1000)

# model
model = YOLO("runs/detect/yolov8n_playingcardsV2/weights/best.pt")

# object classes
classNames = ["10 of clubs", "10 of diamonds", "10 of hearts", "10 of spades", "2 of clubs", "2 of diamonds",
               "2 of hearts", "2 of spades", "3 of clubs", "3 of diamonds", "3 of hearts", "3 of spades", "4 of clubs",
                 "4 of diamonds", "4 of hearts", "4 of spades", "5 of clubs", "5 of diamonds", "5 of hearts", "5 of spades",
                   "6 of clubs", "6 of diamonds", "6 of hearts", "6 of spades", "7 of clubs", "7 of diamonds", "7 of hearts",
                     "7 of spades", "7 of spades", "8 of clubs", "8 of diamonds", "8 of hearts", "8 of spades", "9 of clubs",
                       "9 of diamonds", "9 of hearts", "9 of spades", "Ace of clubs", "Ace of diamonds", "Ace of hearts", "Ace of spades",
                         "Jack of clubs", "Jack of diamonds", "Jack of hearts", "Jack of spades", "Joker", "King of clubs", "King of diamonds",
                           "King of hearts", "King of spades", "Queen of clubs", "Queen of diamonds", "Queen of hearts", "Queen of spades",
                             "black chip", "blue chip", "card back", "chips", "green chip", "red chip", "white chip"]


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()