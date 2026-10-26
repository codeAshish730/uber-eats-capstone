import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/figures")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

features = pd.read_csv(DATA_DIR / "customer_features.csv")

features = features.fillna(features.mean(numeric_only=True))

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

results = []
models = {}

for k in range(2, 9):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_features)

    score = silhouette_score(scaled_features, labels)

    results.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette_score": score
    })
    models[k] = model

    print(f"k={k}: inertia={model.inertia_:.2f}, silhouette={score:.3f}")

results_df = pd.DataFrame(results)
results_df.to_csv(DATA_DIR / "kmeans_evaluation.csv", index=False)

best_k = int(results_df.loc[
    results_df["silhouette_score"].idxmax(), "k"
])

best_model = models[best_k]
labels = best_model.predict(scaled_features)

print(f"\nSelected k={best_k} by silhouette score.")

output = features.copy()
output["cluster"] = labels
output.to_csv(DATA_DIR / "customer_kmeans_clusters.csv", index=False)

joblib.dump(best_model, MODEL_DIR / "customer_kmeans.joblib")
joblib.dump(scaler, MODEL_DIR / "customer_scaler.joblib")

plt.figure(figsize=(8, 5))
plt.plot(results_df["k"], results_df["inertia"], marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")
plt.xticks(results_df["k"])
plt.tight_layout()
plt.savefig(REPORT_DIR / "kmeans_elbow.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(
    results_df["k"],
    results_df["silhouette_score"],
    marker="o"
)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("K-Means Silhouette Scores")
plt.xticks(results_df["k"])
plt.tight_layout()
plt.savefig(REPORT_DIR / "kmeans_silhouette.png", dpi=150)
plt.close()

print("\nSaved evaluation results, cluster assignments, model, scaler, and plots.")