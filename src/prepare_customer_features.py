import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

orders = pd.read_csv(RAW_DIR / "orders.csv")
reviews = pd.read_csv(RAW_DIR / "reviews.csv")

orders["order_date"] = pd.to_datetime(
    orders["order_date"], errors="coerce"
)

orders["is_weekend"] = (
    orders["order_date"].dt.day_of_week >= 5
).astype(int)

orders["is_late_night"] = (
    (orders["order_date"].dt.hour >= 22)
    | (orders["order_date"].dt.hour < 5)
).astype(int)

customer_features = orders.groupby("customer_id").agg(
    total_orders=("order_id", "count"),
    avg_order_value=("order_amount", "mean"),
    total_spending=("order_amount", "sum"),
    first_order_date=("order_date", "min"),
    last_order_date=("order_date", "max"),
    weekend_orders=("is_weekend", "sum"),
    late_night_orders=("is_late_night", "sum")
).reset_index()

customer_features["active_days"] = (
    customer_features["last_order_date"]
    - customer_features["first_order_date"]
).dt.days + 1

customer_features["ordering_frequency"] = (
    customer_features["total_orders"]
    / customer_features["active_days"].clip(lower=1)
)

customer_reviews = reviews.merge(
    orders[["order_id", "customer_id"]],
    on="order_id",
    how="left"
)

avg_ratings = customer_reviews.groupby("customer_id").agg(
    avg_rating_given=("review_rating", "mean")
).reset_index()

customer_features = customer_features.merge(
    avg_ratings, on="customer_id", how="left"
)

overall_rating = reviews["review_rating"].mean()
customer_features["avg_rating_given"] = (
    customer_features["avg_rating_given"].fillna(overall_rating)
)

customer_ids = customer_features["customer_id"].copy()

customer_features = customer_features.drop(
    columns=[
        "customer_id",
        "first_order_date",
        "last_order_date",
        "active_days"
    ]
)

customer_features.to_csv(
    PROCESSED_DIR / "customer_features.csv",
    index=False
)

pd.DataFrame({"customer_id": customer_ids}).to_csv(
    PROCESSED_DIR / "customer_id_mapping.csv",
    index=False
)

print("Customer feature engineering complete.")
print(f"Customers: {len(customer_features)}")
print("\nFeature columns:")
print(customer_features.columns.tolist())
print("\nPreview:")
print(customer_features.head())
print("\nMissing values per feature:")
print(customer_features.isna().sum())