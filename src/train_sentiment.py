import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

df = pd.read_csv("data/processed/cleaned_reviews.csv")
df = df.dropna(subset=["cleaned_review", "review_rating"])

def label_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    return "positive"

df["sentiment"] = df["review_rating"].apply(label_sentiment)

X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_review"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"],
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LinearSVC(class_weight="balanced")),
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion matrix (negative, neutral, positive):")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["negative", "neutral", "positive"],
))

joblib.dump(model, "models/review_sentiment_model.joblib")
print("\nModel saved to models/review_sentiment_model.joblib")