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


# =========================================
# FASTAPI
# =========================================

app = FastAPI(
    title="RPW AI Detection API",
    description="Red Palm Weevil Detection System",
    version="1.0"
)


# =========================================
# CLOUDINARY
# =========================================

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)


# =========================================
# FIREBASE
# =========================================

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


# =========================================
# LOAD YOLO MODEL
# =========================================

print("Loading YOLO model...")

model = YOLO("best.pt")

print("YOLO model loaded successfully!")


# =========================================
# TEMPORARY UPLOAD FOLDER
# =========================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================
# HOME
# =========================================

@app.get("/")
def home():

    return {
        "message": "RPW AI Server Running Successfully",
        "status": "online"
    }


# =========================================
# PREDICT
# =========================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    image_path = None

    try:

        print("")
        print("==============================")
        print("STEP 1 : File Received")
        print("==============================")


        # ---------------------------------
        # Create temporary filename
        # ---------------------------------

        filename = f"{uuid.uuid4()}.jpg"

        image_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )


        # ---------------------------------
        # Save temporarily
        # ---------------------------------

        with open(
            image_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        print("STEP 2 : Temporary image saved")


        # ---------------------------------
        # YOLO DETECTION
        # ---------------------------------

        print("STEP 3 : Running YOLO...")

        results = model.predict(

            source=image_path,

            # Lower resolution = faster
            imgsz=320,

            # Detection confidence
            conf=0.25,

            save=False,

            verbose=False

        )


        print("STEP 4 : YOLO finished")


        # ---------------------------------
        # CHECK DETECTION
        # ---------------------------------

        detected = False

        confidence = 0

        detected_result = None


        for result in results:

            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

                detected = True

                confidence = round(
                    float(
                        result.boxes.conf.max()
                    ) * 100,
                    2
                )

                detected_result = result

                break


        # =================================
        # NO RPW
        # =================================

        if not detected:

            print("")
            print("NO RPW DETECTED")
            print("Image will NOT be uploaded")
            print("Image will NOT be saved")
            print("")

            # Delete temporary image

            if (
                image_path
                and os.path.exists(image_path)
            ):

                os.remove(image_path)


            return {

                "success": True,

                "status": "No RPW",

                "confidence": 0,

                "imageUrl": None

            }


        # =================================
        # RPW DETECTED
        # =================================

        print("")
        print("******************************")
        print("RPW DETECTED!")
        print("Confidence:",
              confidence,
              "%")
        print("******************************")


        status = "RPW Detected"


        # ---------------------------------
        # Draw bounding box
        # ---------------------------------

        annotated_image = detected_result.plot()


        # ---------------------------------
        # Save annotated image temporarily
        # ---------------------------------

        cv2.imwrite(
            image_path,
            annotated_image
        )


        # =================================
        # CLOUDINARY
        # =================================

        print(
            "Uploading detected image to Cloudinary..."
        )


        upload_result = cloudinary.uploader.upload(

            image_path,

            folder="rpw-detections"

        )


        image_url = upload_result[
            "secure_url"
        ]


        print(
            "Cloudinary upload successful!"
        )


        # =================================
        # FIRESTORE
        # =================================

        if db:

            db.collection(
                "traps"
            ).add({

                "trapId": "TRAP001",

                "status": status,

                "confidence": confidence,

                "imageUrl": image_url,

                "location": "Palm Plantation",

                "active": True,

                "timestamp":
                    firestore.SERVER_TIMESTAMP

            })


            print(
                "Firestore detection saved!"
            )


        # =================================
        # DELETE TEMPORARY IMAGE
        # =================================

        if (
            image_path
            and os.path.exists(image_path)
        ):

            os.remove(image_path)


        # =================================
        # RESPONSE
        # =================================

        print(
            "Detection response sent!"
        )

        return {

            "success": True,

            "status": "RPW Detected",

            "confidence": confidence,

            "imageUrl": image_url

        }


    except Exception as e:

        print("")
        print("==============================")
        print("ERROR")
        print("==============================")

        print(
            traceback.format_exc()
        )


        # ---------------------------------
        # Delete temporary file
        # ---------------------------------

        if (
            image_path
            and os.path.exists(image_path)
        ):

            os.remove(image_path)


        return {

            "success": False,

            "status": "Error",

            "error": str(e)

        }


# =========================================
# LOCAL RUN
# =========================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "app:app",

        host="0.0.0.0",

        port=10000,

        reload=False

    )
