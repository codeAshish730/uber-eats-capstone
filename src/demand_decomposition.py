import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/hourly_demand.csv")
df["hour"] = pd.to_datetime(df["hour"])
df = df.sort_values("hour")

Path("reports").mkdir(exist_ok=True)

output = []

output.append("HOURLY DEMAND DECOMPOSITION")
output.append("=" * 35)
output.append(f"Number of hourly observations: {len(df)}")

if len(df) < 2:
    output.append(
        "\nDecomposition not performed: "
        "only one hourly observation is available."
    )
    output.append(
        "Trend, seasonality, and residual noise "
        "cannot be separated from one observation."
    )
else:
    output.append(
        "\nMultiple observations exist. "
        "Check the time span and frequency before decomposition."
    )

report = "\n".join(output)

print(report)

with open(
    "reports/demand_decomposition.txt",
    "w"
) as file:
    file.write(report)

print("\nReport saved to reports/demand_decomposition.txt")