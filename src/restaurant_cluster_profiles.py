
import pandas as pd

# Load restaurant clusters
df = pd.read_csv("data/processed/restaurant_kmeans_clusters.csv")

# Calculate average metrics for each cluster
profiles = df.groupby("cluster").agg(
    restaurant_count=("restaurant_id", "count"),
    avg_orders=("total_orders", "mean"),
    avg_revenue=("revenue", "mean"),
    avg_rating=("avg_review_rating", "mean"),
    avg_cancellation_rate=("cancellation_rate", "mean"),
    avg_delivery_time=("avg_delivery_time", "mean")
).round(2)

print("\nRestaurant Cluster Profiles:")
print(profiles.to_string())

# Save the profiles
profiles.to_csv("reports/restaurant_cluster_profiles.csv")

print("\nProfiles saved to reports/restaurant_cluster_profiles.csv")
