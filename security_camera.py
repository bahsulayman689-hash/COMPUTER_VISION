import cv2
import numpy as np
import datetime
import os
import winsound
from getpass import getpass

print(os.getcwd())
os.makedirs("screenshots", exist_ok=True)
last_screenshot_time = None
last_alarm_time = None
screenshot_delay = 5
alarm_delay = 4
password = 1234
while True:
    try:
        user_password = int(getpass("Enter password: "))
        break
    except ValueError:
        print("Invalid input. Please enter a numeric password.")
if user_password != password:
    print("Wrong password")
    exit()

print("Login successful")

net = cv2.dnn.readNetFromCaffe(
"MobileNetSSD_deploy.prototxt",
"MobileNetSSD_deploy.caffemodel"
)
CLASSES = ["background","aeroplane","bicycle","bird","boat",
"bottle","bus","car","cat","chair","cow","diningtable",
"dog","horse","motorbike","person","pottedplant","sheep",
"sofa","train","tvmonitor"]

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not camera.isOpened():
    print("camera not detected")
    exit()

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("security_record.avi", fourcc, 20.0, (640,480))

while True:
    ret, frame = camera.read()

    if not ret:
        print("frame not received")
        break

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
    net.setInput(blob)
    #detections = net.forward()
    detection= net.forward()

    for i in range(detection.shape[2]):

        confidence = detection[0,0,i,2]

        if confidence > 0.5:

            idx = int(detection[0,0,i,1])
            label = CLASSES[idx]

            if label == "person":

                print("PERSON DETECTED!")

                box = detection[0,0,i,3:7] * np.array(
                [frame.shape[1],frame.shape[0],frame.shape[1],frame.shape[0]]
                )

                (startX,startY,endX,endY) = box.astype("int")

                cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),4)

                timestamp = datetime.datetime.now()

                if (last_screenshot_time is None or
                    (timestamp - last_screenshot_time).total_seconds() >= screenshot_delay):
                    filename = timestamp.strftime("screenshot/person_%Y-%m-%d_%H-%M-%S.jpg")
                    cv2.imwrite(filename, frame)
                    print("Screenshot saved:", filename)
                    last_screenshot_time = timestamp
                if (last_alarm_time is None or
                    (timestamp - last_alarm_time).total_seconds() >= alarm_delay):
                    winsound.Beep(1000, 800)
                    print("ALARM! Person detected")
                    last_alarm_time = timestamp
                cv2.putText(frame,str(timestamp),
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2)

                out.write(frame)
    cv2.namedWindow("AI Security Camera", cv2.WINDOW_NORMAL)   
    cv2.resizeWindow("AI Security Camera", 1000, 700)         
    cv2.imshow("AI Security Camera", frame)


    if cv2.waitKey(1) == ord("q"):
        break
camera.release()
out.release()
cv2.destroyAllWindows()                
