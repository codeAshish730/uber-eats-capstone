import pandas as pd

df = pd.read_csv("data/processed/customer_kmeans_clusters.csv")
profiles = df.groupby("cluster").mean().round(2)
print("\nCustomer Cluster Profiles:")
print(profiles.to_string())
profiles.to_csv("reports/customer_cluster_profiles.csv")
print("\nProfiles saved to reports/customer_cluster_profiles.csv")