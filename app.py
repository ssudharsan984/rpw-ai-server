from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO

import firebase_admin
from firebase_admin import credentials, firestore

import cloudinary
import cloudinary.uploader

import shutil
import os
import uuid
import cv2
import traceback


# ==========================================
# FASTAPI
# ==========================================

app = FastAPI(
    title="RPW AI Detection API",
    description="Red Palm Weevil Detection System",
    version="2.1"
)


# ==========================================
# SETTINGS
# ==========================================

# IMPORTANT:
# Your current model has low validation performance,
# so 0.70 is too strict.
#
# Start with 0.40.
YOLO_CONFIDENCE = 0.40

YOLO_IMAGE_SIZE = 640

MODEL_VERSION = "RPW_MODEL_2.1_CONF_40"

TRAP_ID = "TRAP001"

LOCATION = "Palm Plantation"


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
# LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO("best.pt")

print("YOLO model loaded successfully!")

print("Model classes:")
print(model.names)

print("Confidence threshold:")
print(YOLO_CONFIDENCE)


# ==========================================
# TEMPORARY FOLDER
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

        "message":
            "RPW AI Server Running Successfully",

        "status":
            "online",

        "model_version":
            MODEL_VERSION,

        "confidence_threshold":
            YOLO_CONFIDENCE

    }


# ==========================================
# PREDICT
# ==========================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    image_path = None

    try:

        print()
        print("==============================")
        print("NEW IMAGE RECEIVED")
        print("==============================")

        # ==================================
        # SAVE TEMPORARY IMAGE
        # ==================================

        filename = (
            str(uuid.uuid4())
            + ".jpg"
        )

        image_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        with open(
            image_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(
            "Temporary image saved"
        )

        # ==================================
        # RUN YOLO
        # ==================================

        print()
        print("Running YOLO...")

        results = model.predict(

            source=image_path,

            conf=YOLO_CONFIDENCE,

            imgsz=YOLO_IMAGE_SIZE,

            save=False,

            verbose=False,

            # Prevent excessive detections
            max_det=10,

            agnostic_nms=False

        )

        print(
            "YOLO prediction finished"
        )

        # ==================================
        # FIND BEST RPW DETECTION
        # ==================================

        detected = False

        best_confidence = 0.0

        best_result = None

        for result in results:

            if (
                result.boxes is None
                or len(result.boxes) == 0
            ):

                continue

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                class_id = int(
                    box.cls[0]
                )

                class_name = model.names[
                    class_id
                ]

                print(
                    "Detection:",
                    class_name,
                    "Confidence:",
                    round(
                        confidence * 100,
                        2
                    ),
                    "%"
                )

                # ==================================
                # ONLY ACCEPT OUR RPW CLASS
                # ==================================

                if (
                    class_id == 0
                    and confidence >= YOLO_CONFIDENCE
                ):

                    if (
                        confidence
                        > best_confidence
                    ):

                        best_confidence = (
                            confidence
                        )

                        best_result = result

                        detected = True


        # ==================================
        # NO RPW
        # ==================================

        if not detected:

            print()
            print("------------------------------")
            print("NO RPW DETECTED")
            print("------------------------------")

            # Delete temporary image
            if os.path.exists(
                image_path
            ):

                os.remove(
                    image_path
                )

            return {

                "success": True,

                "detected": False,

                "status": "No RPW",

                "confidence": 0,

                "imageUrl": None,

                "model_version":
                    MODEL_VERSION,

                "threshold":
                    YOLO_CONFIDENCE

            }


        # ==================================
        # RPW DETECTED
        # ==================================

        confidence_percent = round(
            best_confidence * 100,
            2
        )

        print()
        print("==============================")
        print("RPW DETECTED")
        print(
            "Confidence:",
            confidence_percent,
            "%"
        )
        print("==============================")


        # ==================================
        # CREATE ANNOTATED IMAGE
        # ==================================

        annotated_image = (
            best_result.plot()
        )

        cv2.imwrite(
            image_path,
            annotated_image
        )


        # ==================================
        # CLOUDINARY
        # ONLY RPW IMAGES COME HERE
        # ==================================

        print()
        print(
            "Uploading RPW image..."
        )

        upload_result = (
            cloudinary.uploader.upload(
                image_path,
                folder="rpw-detections"
            )
        )

        image_url = (
            upload_result["secure_url"]
        )

        print(
            "Cloudinary upload successful"
        )


        # ==================================
        # FIRESTORE
        # ONLY RPW IMAGES COME HERE
        # ==================================

        if db:

            db.collection(
                "traps"
            ).add({

                "trapId":
                    TRAP_ID,

                "status":
                    "RPW Detected",

                "detected":
                    True,

                "confidence":
                    confidence_percent,

                "imageUrl":
                    image_url,

                "location":
                    LOCATION,

                "active":
                    True,

                "modelVersion":
                    MODEL_VERSION,

                "threshold":
                    YOLO_CONFIDENCE,

                "timestamp":
                    firestore.SERVER_TIMESTAMP

            })

            print(
                "Firestore record saved"
            )


        # ==================================
        # DELETE TEMPORARY IMAGE
        # ==================================

        if os.path.exists(
            image_path
        ):

            os.remove(
                image_path
            )


        # ==================================
        # RESPONSE
        # ==================================

        print()
        print(
            "RPW result sent"
        )

        return {

            "success": True,

            "detected": True,

            "status":
                "RPW Detected",

            "confidence":
                confidence_percent,

            "imageUrl":
                image_url,

            "model_version":
                MODEL_VERSION,

            "threshold":
                YOLO_CONFIDENCE

        }


    # ======================================
    # ERROR
    # ======================================

    except Exception:

        error_message = (
            traceback.format_exc()
        )

        print(
            error_message
        )

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

            "status":
                "Detection Error",

            "error":
                error_message

        }
