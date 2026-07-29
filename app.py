from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
import shutil
import os
import uuid

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(title="RPW AI Detection API")

# -----------------------------
# Initialize Firebase
# -----------------------------
# Local:
cred = credentials.Certificate("serviceAccountKey.json")

# Render:
# cred = credentials.Certificate("/etc/secrets/serviceAccountKey.json")

firebase_admin.initialize_app(cred)

db = firestore.client()

# -----------------------------
# Load YOLO Model
# -----------------------------
print("Loading YOLO model...")
model = YOLO("best.pt")
print("YOLO model loaded successfully!")

# -----------------------------
# Upload Folder
# -----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allow images to be accessed from browser
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# -----------------------------
# Home API
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "RPW AI Server Running Successfully"
    }

# -----------------------------
# Prediction API
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Generate unique filename
    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    # Save uploaded image
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO prediction
    results = model.predict(
        source=image_path,
        conf=0.25,
        save=False
    )

    detected = False

    for result in results:
        if len(result.boxes) > 0:
            detected = True

        # Save image with bounding boxes
        annotated = result.plot()

        import cv2
        cv2.imwrite(image_path, annotated)

    status = "RPW Detected" if detected else "No RPW"

    # Your Render URL
    image_url = f"https://rpw-ai-server-1.onrender.com/uploads/{filename}"

    # Save to Firestore
    firestore_data = {
        "filename": filename,
        "status": status,
        "imageUrl": image_url,
        "timestamp": firestore.SERVER_TIMESTAMP
    }

    db.collection("detections").add(firestore_data)

    return {
        "success": True,
        "status": status,
        "imageUrl": image_url
    }