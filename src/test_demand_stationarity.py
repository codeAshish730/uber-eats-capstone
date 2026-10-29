import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/hourly_demand.csv")
df["hour"] = pd.to_datetime(df["hour"])
df = df.sort_values("hour")

demand = df["order_count"].dropna()

Path("reports").mkdir(exist_ok=True)

lines = [
    "HOURLY DEMAND STATIONARITY TEST",
    "=" * 35,
    f"Number of observations: {len(demand)}",
]

if len(demand) < 20 or demand.nunique() < 2:
    lines.append(
        "\nADF test not performed: insufficient observations "
        "or no variation in hourly demand."
    )
    lines.append(
        "Stationarity cannot be determined from the available data."
    )
else:
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(demand)

    lines.append(f"\nADF statistic: {result[0]:.4f}")
    lines.append(f"p-value: {result[1]:.4f}")
    lines.append(f"Critical values: {result[4]}")

    if result[1] < 0.05:
        lines.append("Conclusion: evidence of stationarity.")
    else:
        lines.append(
            "Conclusion: insufficient evidence of stationarity."
        )

report = "\n".join(lines)
print(report)

with open("reports/demand_stationarity.txt", "w") as file:
    file.write(report)

print("\nReport saved to reports/demand_stationarity.txt")
