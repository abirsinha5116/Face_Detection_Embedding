import cv2
import time
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Load face cascade
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Precompute reference embedding
ref_img_path = r"C:\Users\Dell\Desktop\CV\Zepcruit\Dataset\WIN_20250428_11_43_40_Pro.jpg"
ref_embedding = DeepFace.represent(
    img_path=ref_img_path,
    model_name='OpenFace',
    detector_backend='opencv',
    enforce_detection=False
)[0]['embedding']

# Video capture
cap = cv2.VideoCapture(0)
            
# Timing & frame control
prev_time = time.time()
frame_count = 0
analyze_every_n_frames = 10

# Holds the last comparison result until updated
last_result = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Resize for speed
    frame = cv2.resize(frame, (192, 144))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Compute FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # If we detected at least one face
    if len(boxes) > 0:
        x, y, w, h = boxes[0]

        # Only re-compute embedding every N frames
        if frame_count % analyze_every_n_frames == 0:
            face_roi = frame[y:y+h, x:x+w]
            try:
                live_embedding = DeepFace.represent(
                    img_path=face_roi,
                    model_name='OpenFace',
                    detector_backend='opencv',
                    enforce_detection=False
                )[0]['embedding']
                sim = 1 - cosine(ref_embedding, live_embedding)
                last_result = f"{sim:.2f} – {'Same' if sim > 0.9 else 'Diff'}"
            except Exception:
                last_result = "Embedding failed"
    else:
        # No face found at all
        last_result = "No face detected"

    # Display the last result and FPS
    color = (0, 255, 0) if 'Same' in last_result else (0, 0, 255)
    cv2.putText(frame, last_result, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (1, 77, 220), 2)

    cv2.imshow("Fast Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
