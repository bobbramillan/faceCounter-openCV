import cv2
import time
from flask import Flask, render_template, Response

app = Flask(__name__)

# Face and Palm Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
palm_cascade = cv2.CascadeClassifier('palm.xml')

if face_cascade.empty() or palm_cascade.empty():
    raise IOError("Failed to load cascade classifiers.")

def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not start camera.")

    prev_time = time.time()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face Detection
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            face_count = len(faces)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Palm Detection
            palms = palm_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            for (x, y, w, h) in palms:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, 'PALM DETECTED', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Overlay text
            cv2.putText(frame, f'Faces: {face_count}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
