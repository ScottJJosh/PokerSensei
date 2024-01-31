from ultralytics import YOLO
import cv2
import math 
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1000)
cap.set(4, 1000)

# models
modelPlayingCards = YOLO("runs/detect/externalYoloPlayingCards/yolov8s_playing_cards.pt")
modelPokerChips = YOLO("runs/detect/yolov8s_pokerChipsV1/weights/best.pt")
# object classes

classNames = ['10C','10D','10H','10S','2C','2D','2H','2S','3C','3D','3H','3S','4C','4D','4H','4S','5C','5D','5H','5S','6C','6D','6H','6S','7C','7D','7H','7S','8C','8D','8H','8S','9C','9D','9H','9S','AC','AD','AH','AS','JC',
              'JD','JH','JS','KC','KD','KH','KS','QC','QD','QH','QS']

pokerChips = ["-", "Chip"]
while True:
    success, img = cap.read()
    results = modelPlayingCards(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            org2 = [x1+50, y1+50]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            confString = str(confidence)
            # put box and other info in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img, confString, org2, font, fontScale, color, thickness)
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()