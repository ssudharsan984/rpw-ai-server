
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
    version="2.2"
)


# ==========================================
# SETTINGS
# ==========================================

YOLO_CONFIDENCE = 0.40
YOLO_DEBUG_CONFIDENCE = 0.01
YOLO_IMAGE_SIZE = 640

MODEL_VERSION = "RPW_MODEL_2.2_DEBUG"

TRAP_ID = "TRAP001"
LOCATION = "Palm Plantation"

# Save unsuccessful camera images to Cloudinary
SAVE_DEBUG_IMAGES = True


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
    if not firebase_admin._apps:
        cred = credentials.Certificate(
            "/etc/secrets/serviceAccountKey.json"
        )
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    print("Firebase connected successfully!")

except Exception as e:
    print("Firebase initialization failed:", str(e))


# ==========================================
# LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO("best.pt")

print("YOLO model loaded successfully!")
print("Model classes:", model.names)
print("Final acceptance threshold:", YOLO_CONFIDENCE)
print("Diagnostic prediction threshold:", YOLO_DEBUG_CONFIDENCE)


# ==========================================
# TEMPORARY FOLDER
# ==========================================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():
    return {
        "message": "RPW AI Server Running Successfully",
        "status": "online",
        "model_version": MODEL_VERSION,
        "confidence_threshold": YOLO_CONFIDENCE,
        "debug_confidence": YOLO_DEBUG_CONFIDENCE
    }


# ==========================================
# PREDICT
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_path = None
    cloudinary_debug_url = None

    try:
        print("\n==============================")
        print("NEW IMAGE RECEIVED")
        print("Filename:", file.filename)
        print("Content type:", file.content_type)
        print("==============================")

        # ==================================
        # SAVE RECEIVED IMAGE
        # ==================================

        filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("Temporary image saved:", image_path)

        # Verify that OpenCV can read the image
        image = cv2.imread(image_path)

        if image is None:
            return {
                "success": False,
                "detected": False,
                "status": "Invalid image",
                "error": "Received file could not be decoded as an image"
            }

        height, width = image.shape[:2]
        print(f"Received image dimensions: {width} x {height}")

        # ==================================
        # RUN YOLO AT A LOW DIAGNOSTIC THRESHOLD
        # ==================================

        print("Running YOLO diagnostic prediction...")

        results = model.predict(
            source=image,
            conf=YOLO_DEBUG_CONFIDENCE,
            imgsz=YOLO_IMAGE_SIZE,
            save=False,
            verbose=False,
            max_det=10,
            agnostic_nms=False
        )

        print("YOLO prediction finished")

        best_confidence = 0.0
        best_result = None
        best_box = None
        best_class_id = None

        candidate_count = 0

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                print("YOLO returned NO candidate boxes")
                continue

            for box in result.boxes:
                candidate_count += 1

                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                class_name = model.names.get(
                    class_id, str(class_id)
                ) if isinstance(model.names, dict) else model.names[class_id]

                coordinates = [
                    round(float(v), 1)
                    for v in box.xyxy[0].tolist()
                ]

                print(
                    f"Candidate {candidate_count}: "
                    f"class_id={class_id}, "
                    f"class={class_name}, "
                    f"confidence={confidence:.4f}, "
                    f"box={coordinates}"
                )

                # Keep the highest-confidence RPW candidate
                if (
                    class_id == 0
                    and confidence > best_confidence
                ):
                    best_confidence = confidence
                    best_result = result
                    best_box = box
                    best_class_id = class_id

        print("Total candidate boxes:", candidate_count)

        # ==================================
        # ACCEPT RPW ONLY AT THE REAL THRESHOLD
        # ==================================

        detected = (
            best_box is not None
            and best_confidence >= YOLO_CONFIDENCE
        )

        # ==================================
        # NO ACCEPTED RPW: SAVE IMAGE FOR DEBUGGING
        # ==================================

        if not detected:
            print("\n------------------------------")
            print("NO RPW PASSED THE THRESHOLD")
            print(
                f"Best RPW candidate confidence: "
                f"{best_confidence:.4f}"
            )
            print(
                f"Required confidence: {YOLO_CONFIDENCE:.2f}"
            )
            print("------------------------------")

            if SAVE_DEBUG_IMAGES:
                try:
                    # Upload the ORIGINAL image, not an annotation.
                    upload_result = cloudinary.uploader.upload(
                        image_path,
                        folder="rpw-debug-images",
                        resource_type="image"
                    )

                    cloudinary_debug_url = upload_result.get(
                        "secure_url"
                    )

                    print(
                        "Debug image URL:",
                        cloudinary_debug_url
                    )

                except Exception as upload_error:
                    print(
                        "Debug image upload failed:",
                        str(upload_error)
                    )

            return {
                "success": True,
                "detected": False,
                "status": "No RPW",
                "confidence": round(best_confidence * 100, 2),
                "confidence_fraction": round(best_confidence, 4),
                "candidate_count": candidate_count,
                "threshold": YOLO_CONFIDENCE,
                "imageUrl": None,
                "debugImageUrl": cloudinary_debug_url,
                "model_version": MODEL_VERSION
            }

        # ==================================
        # RPW DETECTED
        # ==================================

        confidence_percent = round(
            best_confidence * 100, 2
        )

        print("\n==============================")
        print("RPW DETECTED")
        print("Confidence:", confidence_percent, "%")
        print("==============================")

        # Draw the detected box on the original image
        annotated_image = best_result.plot()

        if not cv2.imwrite(image_path, annotated_image):
            raise RuntimeError("Could not save annotated image")

        # ==================================
        # CLOUDINARY: ANNOTATED IMAGE
        # ==================================

        print("Uploading annotated RPW image...")

        upload_result = cloudinary.uploader.upload(
            image_path,
            folder="rpw-detections",
            resource_type="image"
        )

        image_url = upload_result["secure_url"]

        print("Annotated image uploaded:", image_url)

        # ==================================
        # FIRESTORE
        # ==================================

        if db:
            db.collection("traps").add({
                "trapId": TRAP_ID,
                "status": "RPW Detected",
                "detected": True,
                "confidence": confidence_percent,
                "imageUrl": image_url,
                "location": LOCATION,
                "active": True,
                "modelVersion": MODEL_VERSION,
                "threshold": YOLO_CONFIDENCE,
                "timestamp": firestore.SERVER_TIMESTAMP
            })

            print("Firestore record saved")
        else:
            print("Firestore unavailable; detection was not saved")

        return {
            "success": True,
            "detected": True,
            "status": "RPW Detected",
            "confidence": confidence_percent,
            "confidence_fraction": round(best_confidence, 4),
            "candidate_count": candidate_count,
            "imageUrl": image_url,
            "debugImageUrl": None,
            "model_version": MODEL_VERSION,
            "threshold": YOLO_CONFIDENCE
        }

    except Exception:
        error_message = traceback.format_exc()
        print(error_message)

        return {
            "success": False,
            "detected": False,
            "status": "Detection Error",
            "error": error_message
        }

    finally:
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError as e:
                print("Temporary file cleanup failed:", str(e))

        await file.close()
