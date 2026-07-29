from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
import shutil
import os
import uuid
import cv2

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(
    title="RPW AI Detection API",
    description="Red Palm Weevil Detection System",
    version="1.0"
)

# -----------------------------
# Firebase
# -----------------------------
db = None

try:
    cred = credentials.Certificate("/etc/secrets/serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase connected successfully!")

except Exception as e:
    print("Firebase initialization failed:", e)

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

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_FOLDER),
    name="uploads"
)

# -----------------------------
# Home
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "RPW AI Server Running Successfully",
        "status": "online"
    }

# -----------------------------
# Predict
# -----------------------------
# ---------------------------------
# Prediction API (Temporary Test)
# ---------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    print("TEST: Predict endpoint reached")

    filename = f"{uuid.uuid4()}.jpg"

    image_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    print("Image saved successfully")

    return {
        "success": True,
        "message": "Upload successful",
        "filename": filename
    }

# -----------------------------
# Local Run
# -----------------------------
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=10000,
        reload=False
    )