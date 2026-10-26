import pandas as pd

restaurants = pd.read_csv("data/raw/restaurants.csv")
orders = pd.read_csv("data/raw/orders.csv")
deliveries = pd.read_csv("data/raw/deliveries.csv")
reviews = pd.read_csv("data/raw/reviews.csv")

order_features = orders.groupby("restaurant_id").agg(
    total_orders=("order_id", "count"),
    revenue=("order_amount", "sum"),
    cancellation_rate=(
        "order_status",
        lambda x: (x.astype(str).str.lower() == "cancelled").mean()
    )
).reset_index()

# Average customer rating per restaurant
reviews_with_restaurants = reviews.merge(
    orders[["order_id", "restaurant_id"]],
    on="order_id",
    how="left"
)

rating_features = reviews_with_restaurants.groupby(
    "restaurant_id"
).agg(
    avg_review_rating=("review_rating", "mean")
).reset_index()

# Average delivery time per restaurant
delivery_orders = deliveries.merge(
    orders[["order_id", "restaurant_id"]],
    on="order_id",
    how="left"
)

delivery_features = delivery_orders.groupby(
    "restaurant_id"
).agg(
    avg_delivery_time=("delivery_time_min", "mean")
).reset_index()

# Combine features with restaurant details
features = restaurants.merge(
    order_features, on="restaurant_id", how="left"
).merge(
    rating_features, on="restaurant_id", how="left"
).merge(
    delivery_features, on="restaurant_id", how="left"
)

# Fill missing numeric values
numeric_columns = [
    "total_orders", "revenue", "cancellation_rate",
    "avg_review_rating", "avg_delivery_time"
]
features[numeric_columns] = features[numeric_columns].fillna(0)

# Save restaurant features
features.to_csv(
    "data/processed/restaurant_features.csv",
    index=False
)

print("Restaurant features saved successfully!")
print("Number of restaurants:", len(features))
print(features.head().to_string(index=False))
