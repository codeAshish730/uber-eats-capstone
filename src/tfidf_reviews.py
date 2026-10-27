import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

df = pd.read_csv("data/processed/cleaned_reviews.csv")

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
tfidf_matrix = vectorizer.fit_transform(df["cleaned_review"].fillna(""))

save_npz("data/processed/reviews_tfidf.npz", tfidf_matrix)

features = vectorizer.get_feature_names_out()
pd.DataFrame({"feature": features}).to_csv(
    "data/processed/tfidf_features.csv", index=False
)

print("TF-IDF matrix saved!")
print("Matrix shape:", tfidf_matrix.shape)
print("\nTop features:")
print(pd.DataFrame({"feature": features}).head(20).to_string(index=False))