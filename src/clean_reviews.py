import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

df = pd.read_csv("data/raw/reviews.csv")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = word_tokenize(text)

    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(cleaned_tokens)

df["cleaned_review"] = df["review_text"].apply(clean_text)

df.to_csv("data/processed/cleaned_reviews.csv", index=False)

print("Cleaned reviews saved!")
print(df[["review_text", "cleaned_review", "review_rating"]].head(10).to_string(index=False))