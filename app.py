from flask import Flask, render_template_string, Response
import cv2
from deepface import DeepFace
import os

app = Flask(__name__)

# Load Haar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Load webcam
camera = cv2.VideoCapture(0)

# Name of person to verify against
target_name = "john"
reference_path = f"dataset/{target_name}.jpg"

# HTML template with live stream
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Face Verification</title>
</head>
<body>
    <h1>Live Face Verification for '{{ name }}'</h1>
    <img src="{{ url_for('video_feed') }}" width="640" height="480" />
    <h2 id="result">{{ result }}</h2>

    <script>
        const evtSource = new EventSource("/result");
        evtSource.onmessage = function(e) {
            document.getElementById("result").textContent = e.data;
        };
    </script>
</body>
</html>
"""

# Global result variable to be updated live
result_text = "Waiting for input..."

@app.route('/')
def index():
    return render_template_string(HTML, name=target_name, result=result_text)

def generate_frames():
    global result_text
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            if os.path.exists(reference_path):
                try:
                    result = DeepFace.verify(img1_path=rgb, img2_path=reference_path, enforce_detection=False)
                    if result['verified']:
                        result_text = f"Match (Distance: {result['distance']:.2f})"
                    else:
                        result_text = f"No Match (Distance: {result['distance']:.2f})"
                except Exception as e:
                    result_text = f"Error: {str(e)}"
            else:
                result_text = "Reference image not found."

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result')
def result_stream():
    def event_stream():
        global result_text
        last_result = ""
        while True:
            if result_text != last_result:
                last_result = result_text
                yield f"data: {result_text}\n\n"
    return Response(event_stream(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
