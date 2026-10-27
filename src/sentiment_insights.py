import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

df = pd.read_csv("data/processed/cleaned_reviews.csv")
df = df.dropna(subset=["cleaned_review", "review_rating"])

def label_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    return "positive"

df["sentiment"] = df["review_rating"].apply(label_sentiment)

stop_words = set(ENGLISH_STOP_WORDS)
results = []

for sentiment in ["negative", "neutral", "positive"]:
    texts = df.loc[df["sentiment"] == sentiment, "cleaned_review"]

    words = []
    for text in texts:
        words.extend(
            word for word in str(text).split()
            if word not in stop_words
        )

    top_words = Counter(words).most_common(10)

    print(f"\nTop words for {sentiment} reviews:")
    for word, count in top_words:
        print(f"{word}: {count}")
        results.append({
            "sentiment": sentiment,
            "word": word,
            "count": count,
        })

pd.DataFrame(results).to_csv(
    "reports/sentiment_insights.csv", index=False
)

print("\nInsights saved to reports/sentiment_insights.csv")