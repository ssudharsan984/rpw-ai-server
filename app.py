from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
import shutil
import os

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(title="RPW AI Detection API")

# -----------------------------
# Initialize Firebase
# -----------------------------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# -----------------------------
# Load YOLO Model
# -----------------------------
print("Loading YOLO model...")
model = YOLO("best.pt")
print("Model loaded successfully!")

# -----------------------------
# Create Upload Folder
# -----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

    # Save uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO prediction
    results = model.predict(
        source=image_path,
        conf=0.25,
        save=True
    )

    detections = []

    for result in results:
        for box in result.boxes:

            detections.append({
                "class": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "xmin": float(box.xyxy[0][0]),
                "ymin": float(box.xyxy[0][1]),
                "xmax": float(box.xyxy[0][2]),
                "ymax": float(box.xyxy[0][3])
            })

    status = "RPW Detected" if len(detections) > 0 else "No RPW"

    # Save to Firestore
    firestore_data = {
        "filename": file.filename,
        "status": status,
        "count": len(detections),
        "detections": detections,
        "timestamp": firestore.SERVER_TIMESTAMP
    }

    db.collection("detections").add(firestore_data)

    return {
        "success": True,
        "filename": file.filename,
        "status": status,
        "count": len(detections),
        "detections": detections
    }