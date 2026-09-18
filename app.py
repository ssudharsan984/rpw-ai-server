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

# ==========================================
# FASTAPI
# ==========================================

app = FastAPI(
    title="RPW AI Detection API",
    description="Red Palm Weevil Detection System",
    version="1.0"
)

# ==========================================
# CLOUDINARY
# ==========================================

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# ==========================================
# FIREBASE
# ==========================================

db = None

try:
    cred = credentials.Certificate(
        "/etc/secrets/serviceAccountKey.json"
    )

    firebase_admin.initialize_app(cred)

    db = firestore.client()

    print("Firebase connected successfully!")

except Exception as e:
    print("Firebase initialization failed")
    print(e)

# ==========================================
# LOAD YOLO
# ==========================================

print("Loading YOLO model...")

model = YOLO("best.pt")

print("YOLO model loaded successfully!")

# ==========================================
# TEMPORARY UPLOAD FOLDER
# ==========================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "message": "RPW AI Server Running Successfully",
        "status": "online"
    }

# ==========================================
# PREDICT
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_path = None

    try:

        print("\n==============================")
        print("NEW IMAGE RECEIVED")
        print("==============================")

        # ----------------------------------
        # Save temporary image
        # ----------------------------------

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

        print("Image saved:", image_path)

        # ----------------------------------
        # YOLO DETECTION
        # ----------------------------------

        print("Running YOLO...")

        results = model.predict(
            source=image_path,
            conf=0.40,
            imgsz=640,
            save=False,
            verbose=False
        )

        print("YOLO finished")

        detected = False
        confidence = 0

        # ----------------------------------
        # CHECK DETECTION
        # ----------------------------------

        for result in results:

            if result.boxes is not None:

                if len(result.boxes) > 0:

                    detected = True

                    confidence = float(
                        result.boxes.conf.max()
                    ) * 100

                    confidence = round(
                        confidence,
                        2
                    )

        # ==================================
        # NO RPW
        # ==================================

        if not detected:

            print("NO RPW DETECTED")

            # Delete temporary image
            if os.path.exists(image_path):

                os.remove(image_path)

            # IMPORTANT:
            # Do NOT upload to Cloudinary
            # Do NOT save to Firestore

            return {

                "success": True,

                "status": "No RPW",

                "detected": False,

                "confidence": 0

            }

        # ==================================
        # RPW DETECTED
        # ==================================

        print("!!!!!!!!!!!!!!!!!!!!!!!!")
        print("RPW DETECTED")
        print(
            "Confidence:",
            confidence,
            "%"
        )
        print("!!!!!!!!!!!!!!!!!!!!!!!!")

        # ----------------------------------
        # Create annotated image
        # ----------------------------------

        for result in results:

            annotated = result.plot()

            cv2.imwrite(
                image_path,
                annotated
            )

        # ----------------------------------
        # Upload ONLY DETECTED IMAGE
        # ----------------------------------

        print(
            "Uploading detected image..."
        )

        upload_result = cloudinary.uploader.upload(
            image_path,
            folder="rpw-detections"
        )

        image_url = upload_result[
            "secure_url"
        ]

        print(
            "Cloudinary upload successful"
        )

        # ==================================
        # FIRESTORE
        # ==================================

        if db:

            db.collection(
                "traps"
            ).add({

                "trapId": "TRAP001",

                "status": "RPW Detected",

                "detected": True,

                "confidence": confidence,

                "imageUrl": image_url,

                "location": "Palm Plantation",

                "active": True,

                "timestamp":
                    firestore.SERVER_TIMESTAMP

            })

            print(
                "Firestore record saved"
            )

        # ----------------------------------
        # Delete temporary image
        # ----------------------------------

        if os.path.exists(image_path):

            os.remove(image_path)

        print("Response sent")

        return {

            "success": True,

            "status": "RPW Detected",

            "detected": True,

            "confidence": confidence,

            "imageUrl": image_url

        }

    # ======================================
    # ERROR
    # ======================================

    except Exception:

        print(
            traceback.format_exc()
        )

        # Delete temporary file
        if image_path:

            if os.path.exists(
                image_path
            ):

                os.remove(
                    image_path
                )

        return {

            "success": False,

            "detected": False,

            "error":
                traceback.format_exc()

        }


# ==========================================
# LOCAL RUN
# ==========================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=10000,
        reload=False
    )
