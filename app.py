from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
import shutil
import os
import uuid
import cv2


# ---------------------------------
# Initialize FastAPI
# ---------------------------------
app = FastAPI(
    title="RPW AI Detection API",
    description="Red Palm Weevil Detection System",
    version="1.0"
)


# ---------------------------------
# Initialize Firebase
# ---------------------------------
try:
    cred = credentials.Certificate(
        "/etc/secrets/serviceAccountKey.json"
    )

    firebase_admin.initialize_app(cred)

    db = firestore.client()

    print("Firebase connected successfully!")

except Exception as e:
    print("Firebase initialization error:")
    print(e)
    db = None



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

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_FOLDER),
    name="uploads"
)



# ---------------------------------
# Home Route
# ---------------------------------
@app.get("/")
def home():

    return {
        "message": "RPW AI Server Running Successfully",
        "status": "online"
    }



# ---------------------------------
# Prediction API
# ---------------------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        # Create unique filename
        filename = f"{uuid.uuid4()}.jpg"

        image_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )


        # Save uploaded image
        with open(image_path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        print("Image saved:", image_path)



        # YOLO Detection
        results = model.predict(
            source=image_path,
            conf=0.25,
            save=False
        )


        detected = False


        for result in results:

            # Check detection
            if len(result.boxes) > 0:

                detected = True


            # Draw bounding box
            annotated = result.plot()


            cv2.imwrite(
                image_path,
                annotated
            )



        if detected:

            status = "RPW Detected"

        else:

            status = "No RPW"



        # Correct Render URL
        image_url = (
            "https://rpw-ai.onrender.com/uploads/"
            + filename
        )



        # Save to Firebase Firestore
        if db:

            firestore_data = {

                "filename": filename,

                "status": status,

                "imageUrl": image_url,

                "timestamp": firestore.SERVER_TIMESTAMP

            }


            db.collection(
                "detections"
            ).add(firestore_data)


            print("Firestore data saved")



        return {

            "success": True,

            "status": status,

            "imageUrl": image_url

        }



    except Exception as e:


        return {

            "success": False,

            "error": str(e)

        }



# ---------------------------------
# Run Local
# ---------------------------------
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=10000
    )