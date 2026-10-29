import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/processed/hourly_demand.csv")
df["hour"] = pd.to_datetime(df["hour"])
df = df.sort_values("hour")

Path("reports/figures").mkdir(parents=True, exist_ok=True)

print("HOURLY DEMAND SUMMARY")
print(df["order_count"].describe())

print("\nHourly demand data:")
print(df.to_string(index=False))

plt.figure(figsize=(10, 5))
plt.bar(df["hour"].astype(str), df["order_count"])
plt.title("Hourly Uber Eats Order Demand")
plt.xlabel("Hour")
plt.ylabel("Number of orders")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(
    "reports/figures/hourly_demand.png",
    dpi=150
)
plt.close()

print("\nChart saved to reports/figures/hourly_demand.png")
print("\nLimitation: Only one hourly observation is available.")
print("Daily, weekly, and seasonal patterns cannot be assessed.")