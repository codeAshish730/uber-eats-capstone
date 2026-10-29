import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/hourly_demand.csv")
df["hour"] = pd.to_datetime(df["hour"])
df = df.sort_values("hour")

# Prophet requires columns named ds and y
prophet_data = df.rename(
    columns={"hour": "ds", "order_count": "y"}
)

Path("reports").mkdir(exist_ok=True)

lines = [
    "PROPHET DEMAND FORECAST",
    "=" * 30,
    f"Hourly observations: {len(prophet_data)}",
]

if (
    len(prophet_data) < 30
    or prophet_data["y"].nunique() < 2
):
    lines.append(
        "\nProphet model not trained: insufficient historical "
        "observations or demand variation."
    )
    lines.append(
        "A reliable future demand forecast cannot be produced "
        "from the available data."
    )
else:
    from prophet import Prophet
    import matplotlib.pyplot as plt

    split = int(len(prophet_data) * 0.8)
    train = prophet_data.iloc[:split]
    test = prophet_data.iloc[split:]

    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(
        periods=len(test),
        freq="h",
        include_history=False
    )

    forecast = model.predict(future)

    comparison = pd.DataFrame({
        "hour": test["ds"].to_numpy(),
        "actual_orders": test["y"].to_numpy(),
        "predicted_orders": forecast["yhat"].to_numpy()
    })

    comparison.to_csv(
        "reports/prophet_test_predictions.csv",
        index=False
    )

    mae = (
        comparison["actual_orders"]
        - comparison["predicted_orders"]
    ).abs().mean()

    rmse = (
        (
            comparison["actual_orders"]
            - comparison["predicted_orders"]
        ) ** 2
    ).mean() ** 0.5

    lines.append(f"\nTest MAE: {mae:.2f}")
    lines.append(f"Test RMSE: {rmse:.2f}")

    figure = model.plot(forecast)
    figure.savefig(
        "reports/figures/prophet_demand_forecast.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(figure)

report = "\n".join(lines)
print(report)

with open("reports/prophet_demand_evaluation.txt", "w") as file:
    file.write(report)

print("\nReport saved to reports/prophet_demand_evaluation.txt")
