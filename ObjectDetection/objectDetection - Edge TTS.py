import cv2
import time
import threading
import os
from playsound import playsound
import asyncio
import edge_tts

VOICE = "tr-TR-AhmetNeural"
#VOICE = "tr-TR-EmelNeural"
OUTPUT_FILE = "voice.mp3"
def Speak(audio):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_main(audio))
    playsound("voice.mp3", False)
    os.remove("voice.mp3")

async def _main(TEXT) -> None:
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_FILE)

# Opencv DNN
net = cv2.dnn.readNet("E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/yolov4-tiny.weights", "E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/yolov4-tiny.cfg")
#net = cv2.dnn.readNet("E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/yolov3.weights", "E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/yolov3.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class lists
classes = []
#with open("E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/classes-en.txt", "r") as file_object:
with open("E:/Muki/Github Projects/CamDetection/ItemDetection/dnn_model/classes-tr.txt", "r") as file_object:
    classes = file_object.read().rstrip('\n').split('\n')

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

start_time = 6;
while True:
    current_time = time.time()
    ret, frame = cap.read()
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.5, nmsThreshold=0.3)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]

        if class_name=="insan" or class_name =="araba":
            print(class_name)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,255), 2)
            cv2.putText(frame, f'{class_name.upper()} {int(score*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.imshow("Frame", frame)
    current_time = time.time()
    elapsed_time = current_time - start_time
    #print(elapsed_time)
    if "insan" in class_name and elapsed_time > 5:
        class_name=""
        t1 = threading.Thread(target=Speak, args=("birinci kamerada bir insan algılandı",))
        t1.start()
        start_time = time.time()

    if "araba" in class_name and elapsed_time > 5:
        class_name=""
        t1 = threading.Thread(target=Speak, args=("birinci kamerada bir araba algılandı",))
        t1.start()
        start_time = time.time()
        
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()