import pandas as pd
from pathlib import Path

deliveries = pd.read_csv("data/raw/deliveries.csv")
orders = pd.read_csv("data/raw/orders.csv")
restaurants = pd.read_csv("data/raw/restaurants.csv")
drivers = pd.read_csv("data/raw/drivers.csv")

df = deliveries.merge(orders, on="order_id", how="inner")

df = df.merge(
    restaurants[["restaurant_id", "rating", "cuisine"]],
    on="restaurant_id",
    how="left",
)

df = df.merge(
    drivers[["driver_id", "vehicle_type"]],
    on="driver_id",
    how="left",
)

df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

df["order_hour"] = df["order_date"].dt.hour
df["day_of_week"] = df["order_date"].dt.dayofweek

Path("data/processed").mkdir(parents=True, exist_ok=True)

output_path = "data/processed/delivery_features.csv"
df.to_csv(output_path, index=False)

print("Feature-engineered dataset saved to:", output_path)
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum().to_string())
print("\nPreview:")
print(df.head().to_string(index=False))