import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/processed/delivery_features.csv")

Path("reports/figures").mkdir(parents=True, exist_ok=True)

target = "delivery_time_min"

print("DELIVERY TIME SUMMARY")
print(df[target].describe())

q1 = df[target].quantile(0.25)
q3 = df[target].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[
    (df[target] < lower_bound) |
    (df[target] > upper_bound)
]

print("\nIQR outlier boundaries:", lower_bound, "to", upper_bound)
print("Number of potential outliers:", len(outliers))

numeric_columns = [
    "delivery_time_min",
    "distance_km",
    "order_amount",
    "rating",
    "order_hour",
    "day_of_week",
]

correlations = df[numeric_columns].corr()

print("\nCorrelation with delivery time:")
print(correlations[target].sort_values(ascending=False))

correlations.to_csv("reports/delivery_correlations.csv")

plt.figure(figsize=(8, 5))
plt.hist(df[target], bins=15, edgecolor="black")
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery time (minutes)")
plt.ylabel("Number of deliveries")
plt.tight_layout()
plt.savefig("reports/figures/delivery_time_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df["distance_km"], df[target], alpha=0.6)
plt.title("Delivery Time vs Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery time (minutes)")
plt.tight_layout()
plt.savefig("reports/figures/delivery_time_vs_distance.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df["order_amount"], df[target], alpha=0.6)
plt.title("Delivery Time vs Order Amount")
plt.xlabel("Order amount")
plt.ylabel("Delivery time (minutes)")
plt.tight_layout()
plt.savefig("reports/figures/delivery_time_vs_order_amount.png", dpi=150)
plt.close()

print("\nEDA charts and correlation report saved.")