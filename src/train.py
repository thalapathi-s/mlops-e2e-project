import os
import pandas as pd
import joblib
import boto3
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==============================
# CONFIG
# ==============================
S3_BUCKET = "loksai-edu-mlproject1"
S3_KEY = "latest/model.pkl"

EXPERIMENT_NAME = "sentiment-classifier"

# If running MLflow server externally
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("data/processed/clean.csv")
X, y = df.review_text, df.sentiment

# ==============================
# PREPROCESS
# ==============================
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

Xtr, Xte, ytr, yte = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# ==============================
# TRAIN
# ==============================
model = LogisticRegression(max_iter=300)

with mlflow.start_run() as run:

    model.fit(Xtr, ytr)

    preds = model.predict(Xte)
    acc = accuracy_score(yte, preds)

    # --------------------------
    # Log Parameters
    # --------------------------
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 300)
    mlflow.log_param("test_size", 0.2)

    # --------------------------
    # Log Metrics
    # --------------------------
    mlflow.log_metric("accuracy", acc)

    # --------------------------
    # Log Model Artifact
    # --------------------------
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="SentimentModel"
    )

    # Save combined object locally
    os.makedirs("models", exist_ok=True)
    joblib.dump((model, vectorizer), "models/model.pkl")

    mlflow.log_artifact("models/model.pkl")

    print("Run ID:", run.info.run_id)
    print("Accuracy:", acc)

# ==============================
# Upload to S3 for Serving
# ==============================
s3 = boto3.client("s3")
s3.upload_file("models/model.pkl", S3_BUCKET, S3_KEY)

print("✅ Model trained, logged to MLflow, and uploaded to S3")
