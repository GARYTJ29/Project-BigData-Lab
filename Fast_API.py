import numpy as np
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest
from starlette.responses import Response

REQUESTS = Counter('http_requests_total', 'Total HTTP Requests', ['method', 'endpoint', 'status_code', 'client_ip'])
REQUEST_LATENCY = Histogram('http_request_latency_seconds', 'HTTP Request Latency', ['method', 'endpoint'])


app = FastAPI()

# Load the trained Keras model
model_path = "model/cnn_model_2.keras"
model = load_model(model_path)

def preprocess_image(image):
    # Resize the image to (128, 128) using Pillow
    resized = image.resize((128, 128))
    
    # Convert the image to grayscale using Pillow
    gray = resized.convert('L')
    
    # Convert the grayscale image to a NumPy array
    gray_array = np.array(gray)
    
    # Normalize the pixel values to the range [0, 1]
    normalized = gray_array / 255.0
    
    # Reshape the array to (1, 128, 128, 1)
    reshaped = np.reshape(normalized, (1, 128, 128, 1))
    
    return reshaped

class RateLimitingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit=1000, duration=60):
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

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    method = request.method
    path = request.url.path
    client_ip = request.client.host

    with REQUEST_LATENCY.labels(method=method, endpoint=path).time():
        response = await call_next(request)

    status_code = response.status_code
    REQUESTS.labels(method=method, endpoint=path, status_code=status_code, client_ip=client_ip).inc()

    return response

@app.get("/metrics")
def metrics():
    registry = CollectorRegistry()
    registry.register(REQUESTS)
    registry.register(REQUEST_LATENCY)
    return Response(generate_latest(registry), media_type="text/plain")

@app.post("/predict")
async def predict_gender(request: Request, file: UploadFile = File(...)):
    # Read the uploaded image file
    img = Image.open(file.file)

    # Preprocess the image
    features = preprocess_image(img)

    # Make predictions using the loaded model
    prediction = model.predict(features)

    # Map the prediction to gender
    gender = 'female' if prediction[0][0] < 0.5 else 'male'

    # Get the client IP address
    client_ip = request.client.host

    return {"gender": gender}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)