import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

features = pd.read_csv("data/processed/customer_features.csv")
clusters = pd.read_csv("data/processed/customer_kmeans_clusters.csv")

X = features.select_dtypes(include="number")
X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters["cluster"],
    cmap="viridis",
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segments - K-Means PCA")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()

plt.savefig("reports/figures/customer_kmeans_pca.png", dpi=300)
plt.close()

print("PCA visualization saved to reports/figures/customer_kmeans_pca.png")
print("Explained variance:", pca.explained_variance_ratio_.sum())