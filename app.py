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

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="RPW AI Detection API",
    description="Red Palm Weevil Detection System",
    version="2.0"
)

# =========================================================
# SETTINGS
# =========================================================

# IMPORTANT:
# Increase this to reduce false detections.
YOLO_CONFIDENCE = 0.70

YOLO_IMAGE_SIZE = 640

TRAP_ID = "TRAP001"

LOCATION = "Palm Plantation"

MODEL_VERSION = "RPW_MODEL_2.0_CONF_70"

# =========================================================
# CLOUDINARY
# =========================================================

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# =========================================================
# FIREBASE
# =========================================================

db = None

try:

    cred = credentials.Certificate(
        "/etc/secrets/serviceAccountKey.json"
    )

    firebase_admin.initialize_app(cred)

    db = firestore.client()

    print("====================================")
    print("Firebase connected successfully")
    print("====================================")

except Exception as e:

    print("Firebase initialization failed")
    print(e)

# =========================================================
# LOAD YOLO MODEL
# =========================================================

print("====================================")
print("Loading YOLO model...")
print("====================================")

model = YOLO("best.pt")

print("YOLO model loaded successfully")

print("Model classes:")
print(model.names)

print("Confidence threshold:")
print(YOLO_CONFIDENCE)

# =========================================================
# TEMPORARY FOLDER
# =========================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "RPW AI Server Running Successfully",
        "status": "online",
        "model_version": MODEL_VERSION,
        "confidence_threshold": YOLO_CONFIDENCE,
        "model_classes": model.names
    }


# =========================================================
# PREDICT
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    image_path = None

    try:

        print()
        print("====================================")
        print("NEW IMAGE RECEIVED")
        print("====================================")

        # =================================================
        # SAVE TEMPORARY IMAGE
        # =================================================

        filename = str(uuid.uuid4()) + ".jpg"

        image_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        with open(image_path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        print("Temporary image saved:")
        print(image_path)

        # =================================================
        # CHECK IMAGE
        # =================================================

        image = cv2.imread(image_path)

        if image is None:

            print("Invalid image")

            if os.path.exists(image_path):
                os.remove(image_path)

            return {
                "success": False,
                "detected": False,
                "status": "Invalid image",
                "confidence": 0,
                "imageUrl": None
            }

        height, width = image.shape[:2]

        print(
            "Image size:",
            width,
            "x",
            height
        )

        # =================================================
        # YOLO DETECTION
        # =================================================

        print()
        print("Running YOLO...")
        print(
            "Confidence threshold:",
            YOLO_CONFIDENCE
        )
        print(
            "Image size:",
            YOLO_IMAGE_SIZE
        )

        results = model.predict(

            source=image_path,

            conf=YOLO_CONFIDENCE,

            imgsz=YOLO_IMAGE_SIZE,

            save=False,

            verbose=False,

            # Don't merge different classes
            agnostic_nms=False,

            # Limit number of detections
            max_det=10
        )

        print("YOLO finished")

        # =================================================
        # FIND BEST VALID RPW DETECTION
        # =================================================

        best_result = None

        best_confidence = 0.0

        best_class_id = None

        detection_count = 0

        for result in results:

            if result.boxes is None:
                continue

            for i in range(len(result.boxes)):

                confidence = float(
                    result.boxes.conf[i]
                )

                class_id = int(
                    result.boxes.cls[i]
                )

                class_name = model.names.get(
                    class_id,
                    str(class_id)
                )

                print(
                    "Detection:",
                    class_name,
                    "confidence:",
                    round(
                        confidence * 100,
                        2
                    ),
                    "%"
                )

                # =================================================
                # ONLY ACCEPT RPW CLASS
                # =================================================

                if class_name.lower() not in [
                    "rpw",
                    "red palm weevil",
                    "red_palm_weevil",
                    "redpalmweevil"
                ]:

                    print(
                        "Rejected class:",
                        class_name
                    )

                    continue

                # =================================================
                # CONFIDENCE CHECK
                # =================================================

                if confidence < YOLO_CONFIDENCE:

                    print(
                        "Rejected low confidence:",
                        round(
                            confidence * 100,
                            2
                        ),
                        "%"
                    )

                    continue

                # =================================================
                # KEEP BEST DETECTION
                # =================================================

                if confidence > best_confidence:

                    best_confidence = confidence

                    best_result = result

                    best_class_id = class_id

                    detection_count += 1

        # =================================================
        # NO VALID RPW
        # =================================================

        if best_result is None:

            print()
            print("------------------------------------")
            print("NO VALID RPW DETECTED")
            print("------------------------------------")

            print(
                "Temporary image will be deleted"
            )

            if os.path.exists(image_path):

                os.remove(image_path)

            return {

                "success": True,

                "detected": False,

                "status": "No RPW",

                "confidence": 0,

                "imageUrl": None,

                "model_version": MODEL_VERSION,

                "threshold": YOLO_CONFIDENCE

            }

        # =================================================
        # RPW DETECTED
        # =================================================

        confidence_percent = round(
            best_confidence * 100,
            2
        )

        print()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("       RPW DETECTED")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        print(
            "Confidence:",
            confidence_percent,
            "%"
        )

        print(
            "Class:",
            model.names.get(
                best_class_id,
                str(best_class_id)
            )
        )

        # =================================================
        # CREATE ANNOTATED IMAGE
        # =================================================

        annotated_image = best_result.plot()

        cv2.imwrite(
            image_path,
            annotated_image
        )

        print(
            "Detection image created"
        )

        # =================================================
        # UPLOAD ONLY RPW IMAGE
        # =================================================

        print()
        print(
            "Uploading RPW detection image..."
        )

        upload_result = cloudinary.uploader.upload(

            image_path,

            folder="rpw-detections",

            resource_type="image"
        )

        image_url = upload_result[
            "secure_url"
        ]

        print(
            "Cloudinary upload successful"
        )

        # =================================================
        # FIRESTORE
        # =================================================

        if db is not None:

            print()
            print(
                "Saving RPW detection to Firestore..."
            )

            db.collection(
                "traps"
            ).add({

                "trapId": TRAP_ID,

                "status": "RPW Detected",

                "detected": True,

                "confidence":
                    confidence_percent,

                "imageUrl":
                    image_url,

                "location":
                    LOCATION,

                "active": True,

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

        # =================================================
        # DELETE TEMPORARY FILE
        # =================================================

        if os.path.exists(image_path):

            os.remove(image_path)

            print(
                "Temporary image deleted"
            )

        # =================================================
        # RESPONSE
        # =================================================

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

    # =====================================================
    # ERROR
    # =====================================================

    except Exception as e:

        print()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("SERVER ERROR")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        print(
            traceback.format_exc()
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

            "confidence": 0,

            "imageUrl": None,

            "error":
                str(e)

        }


# =========================================================
# LOCAL RUN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "app:app",

        host="0.0.0.0",

        port=10000,

        reload=False
    )
