import cv2
import numpy as np
import pyautogui
import time

# Opencv DNN
#net = cv2.dnn.readNet("E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/yolov4-tiny.weights", "E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/yolov4-tiny.cfg")
net = cv2.dnn.readNet("E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/yolov3.weights", "E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/yolov3.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(220, 220), scale=1/255)

# Load class lists
classes = []
#with open("E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/classes-en.txt", "r") as file_object:
with open("E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/classes-tr.txt", "r") as file_object:
    classes = file_object.read().rstrip('\n').split('\n')

colors = np.random.uniform(0, 255, size=(len(classes), 3))

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

start_time = time.time()
frame_count = 0

while True:
    if input1=='10':
        img = pyautogui.screenshot()
        screen =  np.array(img)
        frame = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cap.read()
        
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.5, nmsThreshold=0.3)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        color = colors[class_id]
        #print(class_name)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,128,0), 2)
        cv2.putText(frame, f'{class_name.upper()} {int(score*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,128,0), 2)

    # FPS değerini hesapla
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    # FPS değerini kareye yazdır
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()

