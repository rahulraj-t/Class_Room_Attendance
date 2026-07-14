import cv2
import numpy as np
from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import os

# ------------------ LOAD ENV ------------------
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ------------------ QDRANT ------------------
client = QdrantClient(
      url=QDRANT_URL,
      api_key=QDRANT_API_KEY
)

COLLECTION_NAME = "student_faces"

# ------------------ INSIGHTFACE ------------------
model_root = "model"

face_analyzer = FaceAnalysis(
      name="antelopev2",
      root=model_root,
      providers=["CPUExecutionProvider"]
)

face_analyzer.prepare(ctx_id=0, det_size=(640,640))

# ------------------ IMAGE ------------------

IMAGE_PATH = "class.jpg"         # Replace with your classroom image

img = cv2.imread(IMAGE_PATH)

if img is None:
      print("Image not found")
      exit()

faces = face_analyzer.get(img)

print(f"\nDetected {len(faces)} faces\n")

present_students = []

for i, face in enumerate(faces):

      embedding = face.embedding.astype(np.float32)

      result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding.tolist(),
            limit=1,
            with_payload=True
      )

      if len(result) == 0:
            continue

      match = result[0]

      student_id = match.payload["student_id"]
      similarity = match.score

      if similarity > 0.40:

            present_students.append(student_id)

            print(f"Face {i+1}")
            print("Student :", student_id)
            print("Similarity :", round(similarity,4))
            print("------------------------")

            box = face.bbox.astype(int)

            cv2.rectangle(
                  img,
                  (box[0],box[1]),
                  (box[2],box[3]),
                  (0,255,0),
                  2
            )

            cv2.putText(
                  img,
                  student_id,
                  (box[0],box[1]-10),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.8,
                  (0,255,0),
                  2
            )

print("\nPresent Students")
print(present_students)

cv2.imshow("Attendance", img)
cv2.waitKey(0)
cv2.destroyAllWindows()