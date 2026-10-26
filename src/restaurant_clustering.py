import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("data/processed/restaurant_features.csv")

feature_columns = [
    "total_orders",
    "revenue",
    "cancellation_rate",
    "avg_review_rating",
    "avg_delivery_time"
]

X = df[feature_columns].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = []

for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    results.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette_score": score
    })

evaluation = pd.DataFrame(results)
evaluation.to_csv(
    "data/processed/restaurant_kmeans_evaluation.csv",
    index=False
)

print("\nRestaurant K-Means evaluation:")
print(evaluation.to_string(index=False))

best_k = int(
    evaluation.loc[evaluation["silhouette_score"].idxmax(), "k"]
)

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["cluster"] = final_model.fit_predict(X_scaled)

df.to_csv(
    "data/processed/restaurant_kmeans_clusters.csv",
    index=False
)

joblib.dump(final_model, "models/restaurant_kmeans.joblib")
joblib.dump(scaler, "models/restaurant_scaler.joblib")

print(f"\nBest number of clusters: {best_k}")
print("Restaurant cluster assignments saved.")