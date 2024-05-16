import cv2
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from pyspark.ml.linalg import Vectors
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from datetime import datetime, timedelta
from pyspark.sql import SparkSession

app = FastAPI()
spark = SparkSession.builder \
    .appName("GenderDetectionAPI") \
    .getOrCreate()

# Load the trained PySpark model
model_path = "model/model_32_4_layers"
model = MultilayerPerceptronClassificationModel.load(model_path)

def preprocess_image(image):
    resized = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    flattened = gray.flatten().tolist()
    return Vectors.dense(flattened)

class RateLimitingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit=3, duration=60):
        super().__init__(app)
        self.limit = limit
        self.duration = duration
        self.request_counts = defaultdict(lambda: (0, datetime.now()))

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        count, last_request_time = self.request_counts[client_ip]

        if count >= self.limit and (datetime.now() - last_request_time).total_seconds() < self.duration:
            return JSONResponse(
                status_code=429,
                content={"error": f"Rate limit exceeded for IP {client_ip}. Please try again later."}
            )

        self.request_counts[client_ip] = (count + 1, datetime.now())
        response = await call_next(request)
        return response

app.add_middleware(RateLimitingMiddleware)

@app.post("/predict")
async def predict_gender(file: UploadFile = File(...)):
    # Read the uploaded image file
    img = Image.open(file.file)

    # Preprocess the image
    features = preprocess_image(img)

    # Make predictions using the loaded model
    prediction = model.predict(features)

    # Map the prediction to gender
    gender = 'female' if prediction == 0 else 'male'

    # Get the client IP address
    client_ip = request.client.host

    return {"gender": gender, "client_ip": client_ip}
