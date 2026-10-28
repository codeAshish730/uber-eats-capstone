import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("reports/delivery_test_predictions.csv")

actual = df["actual_delivery_time"]

Path("reports/figures").mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 6))

for model in ["Baseline", "Ridge Regression", "Random Forest"]:
    plt.scatter(
        actual,
        df[model],
        alpha=0.5,
        label=model
    )

lower = min(actual.min(), df[["Baseline", "Ridge Regression", "Random Forest"]].min().min())
upper = max(actual.max(), df[["Baseline", "Ridge Regression", "Random Forest"]].max().max())

plt.plot([lower, upper], [lower, upper], "k--", label="Perfect prediction")
plt.xlabel("Actual delivery time (minutes)")
plt.ylabel("Predicted delivery time (minutes)")
plt.title("Actual vs Predicted Delivery Time")
plt.legend()
plt.tight_layout()

plt.savefig(
    "reports/figures/delivery_actual_vs_predicted.png",
    dpi=150
)
plt.close()

print("Chart saved to reports/figures/delivery_actual_vs_predicted.png")