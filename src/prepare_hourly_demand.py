import pandas as pd
from pathlib import Path

orders = pd.read_csv("data/raw/orders.csv")
orders["order_date"] = pd.to_datetime(
    orders["order_date"], errors="coerce"
)
orders = orders.dropna(subset=["order_date"])

orders["hour"] = orders["order_date"].dt.floor("h")
hourly = (
    orders.groupby("hour")
    .size()
    .rename("order_count")
    .reset_index()
    .sort_values("hour")
)

# Fill any gaps between the first and last observed hour
hourly = hourly.set_index("hour").asfreq("h", fill_value=0)
hourly.index.name = "hour"
hourly = hourly.reset_index()

Path("data/processed").mkdir(parents=True, exist_ok=True)
hourly.to_csv("data/processed/hourly_demand.csv", index=False)

print("Saved: data/processed/hourly_demand.csv")
print("Hourly dataset shape:", hourly.shape)
print(hourly.to_string(index=False))
print("\nTotal orders represented:", hourly["order_count"].sum())
print("\nWARNING: The source data covers only one hour.")
print("This dataset is not sufficient for reliable time-series forecasting.")