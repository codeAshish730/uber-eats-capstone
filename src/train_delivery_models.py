import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/processed/delivery_features.csv")

target = "delivery_time_min"

features = [
    "distance_km",
    "order_amount",
    "rating",
    "cuisine",
    "vehicle_type",
]

X = df[features]
y = df[target]

numeric_features = ["distance_km", "order_amount", "rating"]
categorical_features = ["cuisine", "vehicle_type"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Baseline": DummyRegressor(strategy="mean"),
    "Ridge Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=1.0)),
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=3,
        )),
    ]),
}

results = []
predictions = pd.DataFrame({
    "actual_delivery_time": y_test.reset_index(drop=True)
})

for name, model in models.items():
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    mae = mean_absolute_error(y_test, predicted)
    rmse = np.sqrt(mean_squared_error(y_test, predicted))
    r2 = r2_score(y_test, predicted)

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    predictions[name] = predicted

    print(f"\n{name}")
    print(f"MAE:  {mae:.3f} minutes")
    print(f"RMSE: {rmse:.3f} minutes")
    print(f"R2:   {r2:.3f}")

Path("reports/figures").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results).sort_values("MAE")
results_df.to_csv("reports/delivery_model_comparison.csv", index=False)
predictions.to_csv("reports/delivery_test_predictions.csv", index=False)

print("\nModel comparison saved to reports/delivery_model_comparison.csv")
print("Test predictions saved to reports/delivery_test_predictions.csv")

print("\n5-FOLD CROSS-VALIDATION (MAE)")
for name in ["Ridge Regression", "Random Forest"]:
    scores = cross_val_score(
        models[name],
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
    )
    print(f"{name}: mean MAE = {-scores.mean():.3f} minutes")