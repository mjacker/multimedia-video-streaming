from ultralytics import YOLO
import cv2
import math 
from flask import Flask, Response
import time


def generate_frames():

    # start webcam
    cap = cv2.VideoCapture('thieves.mp4')
    # cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # model
    model = YOLO("yolo-Weights/yolov8n.pt")

    # object classes
    classNames = ["person"]

    
    prev_frame_time = 0
    new_frame_time = time.time()
    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        new_frame_time = time.time()


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
                # cls = int(box.cls[0])
                # print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
    
                gray = img
                gray = cv2.resize(gray, (600, 600))
                font = cv2.FONT_HERSHEY_SIMPLEX

                fps = 1 / (new_frame_time - prev_frame_time + 1)

                # if (int(new_frame_time - prev_frame_time)) > 5:
                # if (fps) > 60:
                    # sleep_second = sleep_second + 1
                    # time.sleep(sleep_second)

                new_frame_time = time.time()

                prev_frame_time = new_frame_time
                fps = int(fps)
                fps = str(fps)
                cv2.putText(gray, fps, (50, 100), font, 3, (100, 255, 0), 3, cv2.LINE_AA)


                ret, buffer = cv2.imencode('.jpg', gray)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



app = Flask(__name__)



@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)

