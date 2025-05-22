# #----------------Version 1: Output printed in terminal-----------------------#
# from deepface import DeepFace
# import cv2

# # Reference image and its embedding
# ref_img_path = "D:\Gesture_Detection\WIN_20250425_12_50_10_Pro.jpg"
# ref_embedding = DeepFace.represent(img_path=ref_img_path, model_name='Facenet')[0]['embedding']
# #print(ref_embedding)

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     try:
#         # Live embeddings of the current frame
#         live_embedding = DeepFace.represent(img_path=frame, model_name='Facenet')[0]['embedding']

#         # Compare embeddings using cosine similarity
#         from scipy.spatial.distance import cosine
#         similarity = 1 - cosine(ref_embedding, live_embedding)

#         print(f"Similarity: {similarity:.2f}")
#         if similarity > 0.7:
#             print("Same Person")
#         else:
#             print("Different Person")

#     except Exception as e:
#         print("No face detected")

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()

# #----------------Version 2: Output Showing in cv2 feed -----------------------#
from deepface import DeepFace
import cv2
import time
from scipy.spatial.distance import cosine

prev_time = time.time()
ref_img_path = r"C:\Users\Dell\Desktop\CV\New Project\Dataset\WIN_20250428_11_43_40_Pro.jpg"
ref_embedding = DeepFace.represent(img_path=ref_img_path, model_name='Facenet')[0]['embedding']
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        live_embedding = DeepFace.represent(img_path=frame, model_name='Facenet')[0]['embedding']
        similarity = 1 - cosine(ref_embedding, live_embedding)

        if similarity > 0.7:
            similarity_text = f"Similarity: {similarity:.2f} - ✅ Same Person"
        else:
            similarity_text = f"Similarity: {similarity:.2f} - ❌ Different Person"
            
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(frame, similarity_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(frame, f'FPS: {int(fps)}', (10,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (1, 77, 220), 2)

    except Exception as e:
        similarity_text = "No face detected"
        cv2.putText(frame, similarity_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
