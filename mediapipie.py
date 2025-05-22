import os
import cv2
import threading
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Load reference image folder
ref_folder = r"C:\Users\Dell\Desktop\CV\Zepcruit\Dataset"
valid_extensions = ('.jpg', '.jpeg', '.png')

ref_image_paths = [
    os.path.join(ref_folder, fname)
    for fname in os.listdir(ref_folder)
    if fname.lower().endswith(valid_extensions)
]

ref_embeddings = []

for path in ref_image_paths:
    try:
        embedding = DeepFace.represent(
            path,
            model_name='Facenet',
            detector_backend='mediapipe',
            enforce_detection=True
        )[0]['embedding']
        ref_embeddings.append(embedding)
    except Exception as e:
        print(f"Failed to process {path}: {e}")

if not ref_embeddings:
    print("No reference embeddings found. Exiting.")
    exit()

# Threaded camera capture
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

# Initialize video stream
stream = VideoStream()

# Variables for logic
frame_count = 0
analyze_every_n_frames = 1
last_txt = "No face"

try:
    while True:
        ret, frame = stream.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % analyze_every_n_frames == 0:
            try:
                resized_frame = cv2.resize(frame, (192, 144))
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

                # MediaPipe handles face detection here
                live_embedding = DeepFace.represent(
                    rgb_frame,
                    model_name='Facenet',
                    detector_backend='mediapipe',
                    enforce_detection=True
                )[0]['embedding']

                avg_ref_embedding = np.mean(ref_embeddings, axis=0)
                sim = 1 - cosine(avg_ref_embedding, live_embedding)
                last_txt = f"{sim:.2f} – {'Same' if sim > 0.60 else 'Diff'}"

            except Exception as e:
                print(f"Live embedding failed: {e}")
                last_txt = "No face"

        print(f"The face is: {last_txt}")

        # Optional display:
        # color = (0, 255, 0) if 'Same' in last_txt else (0, 0, 255)
        # cv2.putText(resized_frame, last_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # cv2.imshow("Face Verification", resized_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

finally:
    stream.stop()
    cv2.destroyAllWindows()
