import cv2
import uuid
import os
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ---------------- LOAD ENV ----------------
load_dotenv(override=True)

# 1. CLEAR ALL PROXY ENVIRONMENT VARIABLES
# This prevents the OS from trying to route traffic through a blocked corporate proxy
for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "NO_PROXY", "no_proxy"]:
    os.environ.pop(var, None)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print("-" * 50)
print(f"🔍 DEBUG: QDRANT_URL = '{QDRANT_URL}'")
print(f"🔍 DEBUG: QDRANT_API_KEY = {'[SET]' if QDRANT_API_KEY else '[NOT SET]'}")
print("-" * 50)

if not QDRANT_URL:
    print("❌ ERROR: QDRANT_URL is missing! Check your .env file.")
    exit()

COLLECTION_NAME = "student_faces"

# ---------------- QDRANT CONNECTION TEST ----------------
try:
    # 2. PASS trust_env=False DIRECTLY TO QdrantClient
    # This tells the underlying httpx library to ignore system proxy settings,
    # bypassing the corporate proxy interception that causes WinError 10013.
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60.0,
        trust_env=False  # <--- THIS IS THE KEY FIX
    )
    
    # Test connection
    client.get_collections()
    print("✅ Successfully connected to Qdrant!\n")
    
except Exception as e:
    print(f"❌ Failed to connect to Qdrant: {e}")
    print("\n⚠️  NEXT STEP: If you still see 'WinError 10013', your Antivirus is blocking python.exe.")
    print("You must add 'python.exe' to your Antivirus/Windows Defender exclusion list.")
    exit()

# ---------------- INSIGHTFACE ----------------
print("Loading InsightFace model (this may take a few seconds)...")
face_analyzer = FaceAnalysis(
    name="antelopev2",
    root="model",
    providers=["CPUExecutionProvider"]
)
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
print("✅ InsightFace loaded.\n")

# ---------------- INPUT ----------------
student_id = input("Enter Student ID: ")
name = input("Enter Full Name: ")
department = input("Enter Department: ")
year = input("Enter Year: ")
image_path = input("Enter Image Path: ")

# ---------------- READ IMAGE ----------------
img = cv2.imread(image_path)
if img is None:
    print("❌ Image not found. Check the file path.")
    exit()

# ---------------- DETECT FACE ----------------
faces = face_analyzer.get(img)

if len(faces) == 0:
    print("❌ No face detected.")
    exit()

if len(faces) > 1:
    print("⚠️ Multiple faces detected. Using the largest face.")

# Select the largest face
face = max(
    faces,
    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
)

embedding = face.embedding.tolist()

# ---------------- STORE ----------------
try:
    # Check if collection exists, create it if it doesn't
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_NAME not in collection_names:
        print(f"⚠️ Collection '{COLLECTION_NAME}' not found. Creating it...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE)
        )
        print("✅ Collection created successfully!\n")

    # Upsert the student data
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "student_id": student_id,
                    "name": name,
                    "department": department,
                    "year": year
                }
            )
        ]
    )
    print("\n✅ Student Registered Successfully in Qdrant!")
    print(f"Student ID: {student_id}")
    print(f"Name: {name}")
except Exception as e:
    print(f"\n❌ Failed to save to Qdrant: {e}")