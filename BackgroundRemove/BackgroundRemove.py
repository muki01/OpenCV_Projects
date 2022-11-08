import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

segmentor = SelfiSegmentation()

img_arkaplan = cv2.imread("./BackgroundRemove/background2.png")

while True:
    success, img = cap.read()

    #imgOut = segmentor.removeBG(img, (0, 0, 0), threshold=0.8)
    imgOut = segmentor.removeBG(img, img_arkaplan, threshold=0.8)

    #cv2.imshow("Image", img)
    cv2.imshow("Image-Out", imgOut)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

