import cv2
import time
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Load Haar cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Timing setup
prev_time = time.time()

# Precompute reference embedding with Facenet
ref_img_path = r"C:\Users\Dell\Desktop\CV\Zepcruit\Dataset\WIN_20250428_11_43_40_Pro.jpg"
ref_embedding = DeepFace.represent(
    img_path=ref_img_path,
    model_name='Facenet',
    detector_backend='opencv',
    enforce_detection=False
)[0]['embedding']

# Open webcam
cap = cv2.VideoCapture(0)

frame_count = 0
analyze_every_n_frames = 10

# Holds the last computed result until the next recompute
last_txt = "No face"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Convert to grayscale for detection
    frame = cv2.resize(frame, (192, 144))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Compute FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    if len(boxes) > 0:
        # Only recompute embedding every Nth frame
        if frame_count % analyze_every_n_frames == 0:
            x, y, w, h = boxes[0]
            face_roi = frame[y:y+h, x:x+w]
            try:
                live_embedding = DeepFace.represent(
                    img_path=face_roi,
                    model_name='Facenet',
                    detector_backend='opencv',
                    enforce_detection=False
                )[0]['embedding']
                sim = 1 - cosine(ref_embedding, live_embedding)
                last_txt = f"{sim:.2f} – {'Same' if sim > 0.7 else 'Diff'}"
            except Exception:
                last_txt = "Embedding failed"
        txt = last_txt
    else:
        # No face detected at all — reset and hold this
        last_txt = "No face"
        txt = last_txt

    # Draw the text and FPS
    color = (0, 255, 0) if 'Same' in txt else (0, 0, 255)
    cv2.putText(frame, txt, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (1, 77, 220), 2)

    cv2.imshow("Fast Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
