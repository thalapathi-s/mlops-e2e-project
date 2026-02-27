from fastapi import FastAPI
import joblib
import boto3
import os

# ---------------------------
# CHANGE THESE IF NEEDED
# ---------------------------
S3_BUCKET = "loksai-edu-mlproject1"
S3_KEY = "latest/model.pkl"
LOCAL_MODEL_PATH = "models/model.pkl"
# ---------------------------

app = FastAPI()

def download_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, S3_KEY, LOCAL_MODEL_PATH)

# Download model ONCE at startup
download_model()

model, vectorizer = joblib.load(LOCAL_MODEL_PATH)

@app.post("/predict")
def predict(text: str):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return {"sentiment": prediction}
