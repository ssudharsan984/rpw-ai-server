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
    version="1.0"
)


# ==========================================
# SETTINGS
# ==========================================

# YOLO confidence threshold
# Start with 0.55
# If real RPW is missed, try 0.45
# If false detections continue, try 0.60
YOLO_CONFIDENCE = 0.55

YOLO_IMAGE_SIZE = 640

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

    print("===================================")
    print("Firebase connected successfully!")
    print("===================================")

except Exception as e:

    print("Firebase initialization failed")
    print(e)


# ==========================================
# LOAD YOLO MODEL
# ==========================================

print("===================================")
print("Loading YOLO model...")
print("===================================")

model = YOLO("best.pt")

print("YOLO model loaded successfully!")

print(
    "YOLO confidence threshold:",
    YOLO_CONFIDENCE
)


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

        "detection_confidence":
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
        print("===================================")
        print("NEW IMAGE RECEIVED")
        print("===================================")


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
            "Temporary image:",
            image_path
        )


        # ==================================
        # CHECK IMAGE
        # ==================================

        image = cv2.imread(
            image_path
        )

        if image is None:

            print(
                "ERROR: Invalid image"
            )

            if os.path.exists(
                image_path
            ):
                os.remove(
                    image_path
                )

            return {

                "success": False,

                "detected": False,

                "status": "Invalid image"

            }


        # ==================================
        # YOLO DETECTION
        # ==================================

        print()
        print(
            "Running YOLO..."
        )

        print(
            "Confidence threshold:",
            YOLO_CONFIDENCE
        )

        results = model.predict(

            source=image_path,

            conf=YOLO_CONFIDENCE,

            imgsz=YOLO_IMAGE_SIZE,

            save=False,

            verbose=False

        )


        print(
            "YOLO finished"
        )


        # ==================================
        # FIND BEST DETECTION
        # ==================================

        detected = False

        best_confidence = 0

        best_result = None


        for result in results:

            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

                current_confidence = float(
                    result.boxes.conf.max()
                )

                print(
                    "YOLO detection:",
                    round(
                        current_confidence * 100,
                        2
                    ),
                    "%"
                )


                if (
                    current_confidence
                    > best_confidence
                ):

                    best_confidence = (
                        current_confidence
                    )

                    best_result = result


        # ==================================
        # FINAL CONFIDENCE CHECK
        # ==================================

        if (
            best_result is not None
            and best_confidence
            >= YOLO_CONFIDENCE
        ):

            detected = True


        # ==================================
        # NO RPW
        # ==================================

        if not detected:

            print()
            print("-----------------------------------")
            print("NO RPW")
            print("-----------------------------------")

            print(
                "Image will NOT be uploaded"
            )

            print(
                "Image will NOT be stored"
            )

            print(
                "Firestore record will NOT be created"
            )


            # --------------------------------
            # DELETE TEMP IMAGE
            # --------------------------------

            if os.path.exists(
                image_path
            ):

                os.remove(
                    image_path
                )

                print(
                    "Temporary image deleted"
                )


            # --------------------------------
            # RETURN ONLY RESULT
            # --------------------------------

            return {

                "success": True,

                "detected": False,

                "status": "No RPW",

                "confidence": 0,

                "imageUrl": None

            }


        # ==================================
        # RPW DETECTED
        # ==================================

        confidence = round(

            best_confidence * 100,

            2

        )


        print()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("       RPW DETECTED")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        print(
            "Confidence:",
            confidence,
            "%"
        )


        # ==================================
        # CREATE ANNOTATED IMAGE
        # ==================================

        print(
            "Creating detection image..."
        )

        annotated_image = (
            best_result.plot()
        )

        cv2.imwrite(

            image_path,

            annotated_image

        )


        # ==================================
        # CLOUDINARY
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

        print(
            "Image URL:",
            image_url
        )


        # ==================================
        # FIRESTORE
        # ==================================

        if db is not None:

            print()
            print(
                "Saving RPW detection to Firestore..."
            )


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
                    confidence,

                "imageUrl":
                    image_url,

                "location":
                    LOCATION,

                "active":
                    True,

                "timestamp":
                    firestore.SERVER_TIMESTAMP

            })


            print(
                "Firestore record saved"
            )

        else:

            print(
                "Firestore unavailable"
            )


        # ==================================
        # DELETE TEMPORARY FILE
        # ==================================

        if os.path.exists(
            image_path
        ):

            os.remove(
                image_path
            )

            print(
                "Temporary file deleted"
            )


        # ==================================
        # RESPONSE
        # ==================================

        print()
        print(
            "RPW detection response sent"
        )

        print(
            "==================================="
        )


        return {

            "success": True,

            "detected": True,

            "status":
                "RPW Detected",

            "confidence":
                confidence,

            "imageUrl":
                image_url

        }


    # ======================================
    # ERROR
    # ======================================

    except Exception as e:

        print()
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )

        print(
            "SERVER ERROR"
        )

        print(
            traceback.format_exc()
        )

        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )


        # ----------------------------------
        # DELETE TEMP IMAGE
        # ----------------------------------

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

            "confidence": 0,

            "imageUrl": None,

            "error":
                str(e)

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
