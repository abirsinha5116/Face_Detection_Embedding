import os
import time
import threading
import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

REF_FOLDER = r"C:\Users\Dell\Desktop\CV\Zepcruit\Dataset"
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')
REF_CAPTURE_COUNT = 15       # Number of frames to capture
CAPTURE_TIMEOUT = 30         # Seconds to wait for captures
ANALYZE_EVERY_N_FRAMES = 1   # Frequency of analysis in main loop
SIMILARITY_THRESHOLD = 0.65  # Face match threshold
# FRAME_SIZE = (192, 144)      # Resize for speed

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()

stream = VideoStream()

os.makedirs(REF_FOLDER, exist_ok=True)

# STEP 1: AUTO-CAPTURE REFERENCE IMAGES

captured = 0
start_time = time.time()
print(f"[INFO] Capturing up to {REF_CAPTURE_COUNT} reference images...")

while captured < REF_CAPTURE_COUNT and (time.time() - start_time) < CAPTURE_TIMEOUT:
    ret, frame = stream.read()
    if not ret:
        continue

    # small_frame = cv2.resize(frame, FRAME_SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = frame[y:y+h, x:x+w]
        filename = os.path.join(REF_FOLDER, f"ref_face_{captured+1}.jpg")
        cv2.imwrite(filename, face_img)
        print(f"[INFO] Saved {filename}")
        captured += 1

    time.sleep(0.3)

if captured == 0:
    print("[ERROR] No faces captured. Exiting.")
    stream.stop()
    cv2.destroyAllWindows()
    exit()

print(f"[INFO] Captured {captured} reference images.")

# STEP 2: COMPUTE EMBEDDINGS

ref_paths = [
    os.path.join(REF_FOLDER, f)
    for f in os.listdir(REF_FOLDER)
    if f.lower().endswith(VALID_EXTENSIONS)
]

ref_embeddings = []
for path in ref_paths:
    try:
        rep = DeepFace.represent(
            path,
            model_name='Facenet',
            detector_backend='opencv',
            enforce_detection=False
        )
        embedding = rep[0]['embedding']
        ref_embeddings.append(embedding)
    except Exception as e:
        print(f"[WARN] Could not process {path}: {e}")

if not ref_embeddings:
    print("[ERROR] No valid reference embeddings. Exiting.")
    stream.stop()
    cv2.destroyAllWindows()
    exit()

# STEP 3: REAL-TIME VERIFICATION LOOP

frame_count = 0
last_txt = "No face"

try:
    while True:
        ret, frame = stream.read()
        if not ret:
            break

        frame_count += 1
        # small_frame = cv2.resize(frame, FRAME_SIZE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        if len(faces) > 0:
            if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
                x, y, w, h = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                try:
                    rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    live_rep = DeepFace.represent(
                        rgb,
                        model_name='Facenet',
                        detector_backend='opencv',
                        enforce_detection=False
                    )
                    live_emb = live_rep[0]['embedding']

                    avg_ref = np.mean(ref_embeddings, axis=0)
                    sim = 1 - cosine(avg_ref, live_emb)
                    status = 'Same' if sim > SIMILARITY_THRESHOLD else 'Diff'
                    last_txt = f"{sim:.2f} – {status}"
                except Exception as e:
                    print(f"[WARN] Live embedding failed: {e}")
                    last_txt = "Embedding failed"

            txt = last_txt
        else:
            last_txt = "No face"
            txt = last_txt

        print(f"The face is: {txt}")

finally:
    stream.stop()
    cv2.destroyAllWindows()