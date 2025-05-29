import cv2
from deepface import DeepFace
import tf_gpu_init
tf_gpu_init.enable_memory_growth()


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            detector_backend='retinaface',
            enforce_detection=False
        )

        if isinstance(results, list):
            if len(results) == 1:
                print("Single face detected")
            else:
                print("Multiple faces detected")

    except Exception as e:
        print("Detection error:", e)

cap.release()
cv2.destroyAllWindows()
