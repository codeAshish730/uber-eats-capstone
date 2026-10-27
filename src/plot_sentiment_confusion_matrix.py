import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

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

labels = ["negative", "neutral", "positive"]
display = ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=labels,
    display_labels=labels,
    cmap="Blues",
)

display.ax_.set_title("Review Sentiment - Linear SVM")
plt.tight_layout()
plt.savefig(
    "reports/figures/sentiment_confusion_matrix.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

print("Saved reports/figures/sentiment_confusion_matrix.png")