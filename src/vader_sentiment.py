import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

df = pd.read_csv("data/raw/reviews.csv")
df = df.dropna(subset=["review_text", "review_rating"])

def label_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    return "positive"

df["sentiment"] = df["review_rating"].apply(label_sentiment)

analyzer = SentimentIntensityAnalyzer()

def predict_vader(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"

df["vader_sentiment"] = df["review_text"].apply(predict_vader)

labels = ["negative", "neutral", "positive"]

print("VADER accuracy:", round(
    accuracy_score(df["sentiment"], df["vader_sentiment"]), 3
))
print("\nVADER classification report:")
print(classification_report(
    df["sentiment"], df["vader_sentiment"],
    labels=labels, zero_division=0
))

print("VADER confusion matrix (negative, neutral, positive):")
print(confusion_matrix(
    df["sentiment"], df["vader_sentiment"], labels=labels
))

df.to_csv("data/processed/vader_review_predictions.csv", index=False)
print("\nPredictions saved to data/processed/vader_review_predictions.csv")