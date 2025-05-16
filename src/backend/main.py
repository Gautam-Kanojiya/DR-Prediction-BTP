# main.py - FastAPI backend for QNN DR prediction with macular edema

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import uvicorn
from utils import classifier, preprocessing

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

THRESHOLD = 0.5

# Global model containers
edema_models = []
dr_models = []

def train_models():
    global edema_models, dr_models
    training_data_labels_path = "src/dataset/training_data/training_labels.csv"
    testing_data_labels_path = "src/dataset/testing_data/testing_labels.csv"
    validate = False

    print("\nTraining Macular Edema Models")
    edema_models = [
        classifier.compile_edema_qnn_model(training_data_labels_path, testing_data_labels_path, THRESHOLD, validate, i)
        for i in range(3)
    ]

    print("\nTraining DR Models")
    dr_models = [
        classifier.compile_qnn_model(training_data_labels_path, testing_data_labels_path, THRESHOLD, validate, i, edema_models)
        for i in range(5)
    ]

    print("DONE : Successfully Trained all models\n")

@app.post("/upload/")
async def upload_image(file: UploadFile):
    try:
        file_extension = file.filename.split(".")[-1]
        file_name = f"uploaded_image.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, file_name)
        with open(file_path, "wb") as image_file:
            image_file.write(await file.read())

        img = cv2.imread(file_path)
        processed_image = preprocessing.preprocess_image(img)

        predicted_grade = classifier.classify_with_edema(
            THRESHOLD, processed_image, edema_models, dr_models
        )

        response_dict = {
            "predicted_grade": str(predicted_grade),
            "message": "Successfully predicted the Diabetic Retinopathy Grade with Macular Edema context!"
        }
        return JSONResponse(content=response_dict, status_code=200)

    except Exception as err:
        print(f"ERROR : {err}")
        return JSONResponse(content={"message": "An error occurred"}, status_code=500)

if __name__ == "__main__":
    print("\n\n<--- DR + Macular Edema Classification using Quantum Neural Networks --->\n")
    train_models()
    uvicorn.run(app, port=8000)