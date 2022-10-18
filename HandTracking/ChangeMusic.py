import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import pyautogui
import numpy as np
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.5, maxHands=1)

x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C
while True:
    success, img = cap.read()
    hands = detector.findHands(img, draw=False)

    if hands:
        lmList = hands[0]["lmList"]
        bbox = hands[0]["bbox"]
        x1, y1 = lmList[5][0:2]
        x2, y2 = lmList[17][0:2]
        #centerPoint1 = hands[0]["center"]
        #handType1 = hands[0]["type"]
        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        A, B, C = coff
        distanceCM = A * distance ** 2 + B * distance + C

        if distanceCM <45:
            if lmList[12][0] >900:
                success, img = cap.read()
                pyautogui.press("prevtrack")
                success, img = cap.read()
                time.sleep(0.2)
                success, img = cap.read()
            elif lmList[12][0] < 400:
                success, img = cap.read()
                pyautogui.press("nexttrack")
                success, img = cap.read()
                time.sleep(0.2)
                success, img = cap.read()
        cv2.rectangle(img, bbox, (255, 0, 255), 3)
        cv2.putText(img, f'{int(distanceCM)} cm', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()