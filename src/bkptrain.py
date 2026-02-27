import pandas as pd
import joblib
import boto3
import os
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

S3_BUCKET = "loksai-edu-mlproject1"
S3_KEY = "latest/model.pkl"


df = pd.read_csv("data/processed/clean.csv")
X, y = df.review_text, df.sentiment

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

Xtr, Xte, ytr, yte = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=300)
model.fit(Xtr, ytr)

preds = model.predict(Xte)
acc = accuracy_score(yte, preds)

with mlflow.start_run():
    mlflow.log_metric("accuracy", acc)

joblib.dump((model, vectorizer), "models/model.pkl")
print("Accuracy:", acc)

# Upload to S3 for serving
s3 = boto3.client("s3")
s3.upload_file("models/model.pkl", S3_BUCKET, S3_KEY)

print("✅ Model trained and uploaded to S3")
