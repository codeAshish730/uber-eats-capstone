from pathlib import Path
import pandas as pd

Path("reports").mkdir(exist_ok=True)

results = pd.DataFrame([
    {
        "model": "Baseline",
        "status": "Not evaluated",
        "MAE": None,
        "RMSE": None,
        "MAPE": None,
        "reason": "Insufficient hourly history"
    },
    {
        "model": "ARIMA",
        "status": "Not trained",
        "MAE": None,
        "RMSE": None,
        "MAPE": None,
        "reason": "Only one hourly observation"
    },
    {
        "model": "Prophet",
        "status": "Not trained",
        "MAE": None,
        "RMSE": None,
        "MAPE": None,
        "reason": "Only one hourly observation"
    }
])

results.to_csv(
    "reports/demand_model_comparison.csv",
    index=False
)

print("DEMAND MODEL COMPARISON")
print("=" * 30)
print(results.to_string(index=False))
print("\nComparison saved to reports/demand_model_comparison.csv")
print(
    "\nMAE, RMSE, and MAPE are unavailable because "
    "the models could not be evaluated on a chronological test set."
)
