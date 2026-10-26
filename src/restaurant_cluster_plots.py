
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load restaurant features and cluster assignments
df = pd.read_csv("data/processed/restaurant_kmeans_clusters.csv")

features = [
    "total_orders",
    "revenue",
    "cancellation_rate",
    "avg_review_rating",
    "avg_delivery_time"
]

X = df[features].fillna(0)
X_scaled = StandardScaler().fit_transform(X)

# Elbow plot
inertias = []
k_values = range(2, 7)

for k in k_values:
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Restaurant K-Means Elbow Plot")
plt.tight_layout()
plt.savefig("reports/figures/restaurant_kmeans_elbow.png", dpi=300)
plt.close()

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["cluster"],
    cmap="viridis",
    alpha=0.8
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Restaurant Segments - K-Means PCA")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("reports/figures/restaurant_kmeans_pca.png", dpi=300)
plt.close()

print("Restaurant elbow and PCA plots saved.")
print("PCA explained variance:", round(pca.explained_variance_ratio_.sum(), 3))
