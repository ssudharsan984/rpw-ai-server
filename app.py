from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
import shutil
import os
import uuid
import cv2
import traceback
import cloudinary
import cloudinary.uploader

# ---------------------------------
# FastAPI
# ---------------------------------

app = FastAPI(
    title="RPW AI Detection API",
    description="Red Palm Weevil Detection System",
    version="1.0"
)

# ---------------------------------
# Cloudinary
# ---------------------------------

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# ---------------------------------
# Firebase
# ---------------------------------

db = None

try:
    cred = credentials.Certificate("/etc/secrets/serviceAccountKey.json")

    firebase_admin.initialize_app(cred)

    db = firestore.client()

    print("Firebase connected successfully!")

except Exception as e:

    print("Firebase initialization failed")
    print(e)

# ---------------------------------
# Load YOLO Model
# ---------------------------------

print("Loading YOLO model...")

model = YOLO("best.pt")

print("YOLO model loaded successfully!")

# ---------------------------------
# Upload Folder
# ---------------------------------

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------
# Home
# ---------------------------------

@app.get("/")
def home():

    return {
        "message": "RPW AI Server Running Successfully",
        "status": "online"
    }

# ---------------------------------
# Predict
# ---------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        print("STEP 1 : File Received")

        filename = f"{uuid.uuid4()}.jpg"

        image_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("STEP 2 : Image Saved")

        print("STEP 3 : Running YOLO")

        results = model.predict(
            source=image_path,
            conf=0.25,
            imgsz=640,
            save=False,
            verbose=False
        )

        print("STEP 4 : YOLO Finished")

        detected = False
        confidence = 0

        for result in results:

            if len(result.boxes) > 0:

                detected = True

                confidence = round(
                    float(result.boxes.conf.max()) * 100,
                    2
                )

            annotated = result.plot()

            cv2.imwrite(
                image_path,
                annotated
            )

        status = "RPW Detected" if detected else "No RPW"

        print("Uploading image to Cloudinary...")

        upload_result = cloudinary.uploader.upload(
            image_path,
            folder="rpw-detections"
        )

        image_url = upload_result["secure_url"]

        print("Cloudinary Upload Successful")

        # ---------------------------------
        # Save Firestore
        # ---------------------------------

        if db:

            db.collection("traps").add({

                "trapId": "TRAP001",

                "status": status,

                "confidence": confidence,

                "imageUrl": image_url,

                "location": "Palm Plantation",

                "active": True,

                "timestamp": firestore.SERVER_TIMESTAMP

            })

            print("Firestore Saved Successfully")

        # Delete temporary file
        if os.path.exists(image_path):
            os.remove(image_path)

        print("Response Sent")

        return {

            "success": True,

            "status": status,

            "confidence": confidence,

            "imageUrl": image_url

        }

    except Exception:

        print(traceback.format_exc())

        return {

            "success": False,

            "error": traceback.format_exc()

        }

# ---------------------------------
# Local Run
# ---------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=10000,
        reload=False
    )