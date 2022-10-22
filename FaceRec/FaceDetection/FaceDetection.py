import cv2
from cvzone.FaceDetectionModule import FaceDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = FaceDetector(minDetectionCon=0.5)

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)

    #if bboxs:
        #center = bboxs[0]["center"]
        #cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()