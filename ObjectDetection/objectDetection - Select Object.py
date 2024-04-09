import cv2
from gui_buttons import Buttons

# Initialize Buttons
button = Buttons()
button.add_button("insan", 20, 20)
button.add_button("araba", 20, 60)
button.add_button("telefon", 20, 100)

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

def click_button(event, x, y, flags, params):
    global buton_insan
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)

# Frame isimli pencerede fareye tıklandığında click_button isimli metod çağırılacak
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)

while True:
    ret, frame = cap.read()

    # Get active buttons list
    active_buttons = button.active_buttons_list()
    #print("Active buttons", active_buttons)

    # Object Detection
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.5, nmsThreshold=0.3)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        #print(class_name)

        if class_name in active_buttons:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,255), 2)
            cv2.putText(frame, f'{class_name.upper()} {int(score*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Display buttons
    button.display_buttons(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

