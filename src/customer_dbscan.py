
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/figures")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

features = pd.read_csv(DATA_DIR / "customer_features.csv")
features = features.fillna(features.mean(numeric_only=True))

scaler = StandardScaler()
X = scaler.fit_transform(features)

# Start with a reasonable baseline; tune after reviewing the results.
model = DBSCAN(eps=1.5, min_samples=5)
labels = model.fit_predict(X)

output = features.copy()
output["dbscan_cluster"] = labels
output.to_csv(DATA_DIR / "customer_dbscan_clusters.csv", index=False)

joblib.dump(model, MODEL_DIR / "customer_dbscan.joblib")
joblib.dump(scaler, MODEL_DIR / "customer_dbscan_scaler.joblib")

cluster_labels = set(labels)
n_clusters = len(cluster_labels - {-1})
n_noise = int((labels == -1).sum())

print(f"DBSCAN clusters (excluding noise): {n_clusters}")
print(f"Noise observations: {n_noise}")
print("Cluster sizes:")
print(output["dbscan_cluster"].value_counts().sort_index())

# Silhouette score is meaningful only with 2+ non-noise clusters
# and fewer clusters than observations.
non_noise = labels != -1
unique_non_noise = set(labels[non_noise])

if len(unique_non_noise) >= 2 and non_noise.sum() > len(unique_non_noise):
    score = silhouette_score(X[non_noise], labels[non_noise])
    print(f"Silhouette score (non-noise points): {score:.3f}")
else:
    print("Silhouette score not calculated: insufficient non-noise clusters.")

# PCA visualization for DBSCAN results
from sklearn.decomposition import PCA

X_2d = PCA(n_components=2).fit_transform(X)

plt.figure(figsize=(8, 5))
scatter = plt.scatter(
    X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=30
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("DBSCAN Customer Clusters (PCA Projection)")
plt.colorbar(scatter, label="Cluster (-1 = noise)")
plt.tight_layout()
plt.savefig(REPORT_DIR / "dbscan_pca.png", dpi=150)
plt.close()

print("Saved DBSCAN assignments, model, and PCA visualization.")
