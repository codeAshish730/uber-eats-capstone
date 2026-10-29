import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/hourly_demand.csv")
df["hour"] = pd.to_datetime(df["hour"])
df = df.sort_values("hour").set_index("hour")

demand = df["order_count"].asfreq("h")

Path("reports").mkdir(exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

lines = [
    "ARIMA DEMAND FORECAST",
    "=" * 30,
    f"Hourly observations: {demand.notna().sum()}",
]

if demand.notna().sum() < 30 or demand.dropna().nunique() < 2:
    lines.append(
        "\nARIMA model not trained: insufficient historical "
        "observations or demand variation."
    )
    lines.append(
        "A reliable future demand forecast cannot be produced "
        "from the available data."
    )
else:
    from statsmodels.tsa.arima.model import ARIMA
    import matplotlib.pyplot as plt

    demand = demand.interpolate()

    split = int(len(demand) * 0.8)
    train = demand.iloc[:split]
    test = demand.iloc[split:]

    model = ARIMA(train, order=(1, 1, 1)).fit()
    predictions = model.forecast(steps=len(test))

    comparison = pd.DataFrame({
        "actual_orders": test,
        "predicted_orders": predictions
    })
    comparison.to_csv(
        "reports/arima_test_predictions.csv"
    )

    mae = (test - predictions).abs().mean()
    rmse = ((test - predictions) ** 2).mean() ** 0.5

    lines.append(f"\nTest MAE: {mae:.2f}")
    lines.append(f"Test RMSE: {rmse:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train, label="Training demand")
    plt.plot(test.index, test, label="Actual demand")
    plt.plot(
        predictions.index,
        predictions,
        label="ARIMA forecast"
    )
    plt.legend()
    plt.title("ARIMA Hourly Demand Forecast")
    plt.tight_layout()
    plt.savefig(
        "reports/figures/arima_demand_forecast.png",
        dpi=150
    )
    plt.close()

report = "\n".join(lines)
print(report)

with open("reports/arima_demand_evaluation.txt", "w") as file:
    file.write(report)

print("\nReport saved to reports/arima_demand_evaluation.txt")
