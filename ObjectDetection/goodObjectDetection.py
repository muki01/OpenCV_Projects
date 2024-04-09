import cv2
import numpy as np

# Initialize camera
input1 = input("Enter cam number: ")
if input1 == "0":
    cam=0
elif input1 == "9":
    cam = "http://***"
else:
    cam="rtsp://***&channel="+input1+ "&stream=0"
cap = cv2.VideoCapture(cam)
cap.set(3, 1280)
cap.set(4, 720)
whT = 320
confThreshold =0.5
nmsThreshold= 0.3

## Model Files
net = cv2.dnn.readNetFromDarknet("E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/yolov4-tiny.cfg", "E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/yolov4-tiny.weights")
#net = cv2.dnn.readNetFromDarknet("E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/yolov3.cfg", "E:\Muki\Github Projects\CamDetection\ItemDetection\dnn_model/yolov3.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

classes = []
#with open("E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/classes-en.txt", "r") as file_object:
with open("E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/classes-tr.txt", "r") as file_object:
    classes = file_object.read().rstrip('\n').split('\n')

def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
        cv2.putText(img,f'{classes[classIds[i]].upper()} {int(confs[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        print(classes[classIds[i]].upper())

while True:
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs,img)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cv2.destroyAllWindows()