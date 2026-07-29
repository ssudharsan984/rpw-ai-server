from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
import shutil
import os
import uuid
import cv2
import traceback

# ---------------------------------
# FastAPI
# ---------------------------------

app = FastAPI(
    title="RPW AI Detection API",
    description="Red Palm Weevil Detection System",
    version="1.0"
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

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_FOLDER),
    name="uploads"
)

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

            shutil.copyfileobj(
                file.file,
                buffer
            )

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

        image_url = (
            "https://rpw-ai.onrender.com/uploads/"
            + filename
        )

        print(status)

        # ---------------------------------
        # Save to Firestore
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