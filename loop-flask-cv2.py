from flask import Flask, Response
import cv2
import time

app = Flask(__name__)

IMAGE_RESOLUTION_WIDTH = 400
IMAGE_RESOLUTION_HEIGHT = 400

video_path = 'meatzoo.mp4'  # Replace 'your_video
camera = cv2.VideoCapture(video_path)  # Use 0 for the default camera (usually the Raspberry Pi camera)


def generate_frames():
    prev_frame_time = time.time()
    new_frame_time = time.time()
    sleep_second = 0
    while camera.isOpened():
        camera.set(cv2.CAP_PROP_FPS, 5)
        ret, frame = camera.read()
        if not ret:
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:

            gray = frame
            gray = cv2.resize(gray, (IMAGE_RESOLUTION_HEIGHT, IMAGE_RESOLUTION_WIDTH))
            font = cv2.FONT_HERSHEY_SIMPLEX
            new_frame_time = time.time()


            fps = 1 / (new_frame_time - prev_frame_time)

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




@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)

