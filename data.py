from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
      url=os.getenv("QDRANT_URL"),
      api_key=os.getenv("QDRANT_API_KEY")
)

student_id = input("Enter Student ID: ")

result = client.scroll(
      collection_name="student_faces",
      scroll_filter=models.Filter(
            must=[
                  models.FieldCondition(
                        key="student_id",
                        match=models.MatchValue(value=student_id)
                  )
            ]
      ),
      limit=1,
      with_payload=True,
      with_vectors=False
)

if result[0]:
      payload = result[0][0].payload

      print("Student ID :", payload["student_id"])
      print("Image Path :", payload["image_path"])

else:
      print("Student not found")