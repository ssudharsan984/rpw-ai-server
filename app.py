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
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        print("STEP 1 : File received")

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

        print("STEP 2 : Image saved")

        image = cv2.imread(image_path)

        if image is None:
            raise Exception("Unable to read uploaded image")

        print("STEP 3 : Starting YOLO")

        results = model(
            image,
            conf=0.25,
            imgsz=320,
            verbose=False
        )

        print("STEP 4 : YOLO completed")

        detected = False

        for result in results:

            if len(result.boxes) > 0:
                detected = True

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

        print("STEP 5 : Saving Firestore")

        if db is not None:

            db.collection("detections").add({

                "filename": filename,

                "status": status,

                "imageUrl": image_url,

                "timestamp": firestore.SERVER_TIMESTAMP

            })

            print("Firestore saved")

        print("STEP 6 : Sending Response")

        return {

            "success": True,

            "status": status,

            "imageUrl": image_url

        }

    except Exception as e:

        print("ERROR:", repr(e))

        return {

            "success": False,

            "error": str(e)

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