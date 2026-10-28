import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

df = pd.read_csv("data/processed/delivery_features.csv")

features = [
    "distance_km",
    "order_amount",
    "rating",
    "cuisine",
    "vehicle_type",
]
target = "delivery_time_min"

X = df[features]
y = df[target]

numeric_features = ["distance_km", "order_amount", "rating"]
categorical_features = ["cuisine", "vehicle_type"]

preprocessor = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
        numeric_features,
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]),
        categorical_features,
    ),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=3,
    )),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = model.named_steps[
    "model"
].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance,
}).sort_values("importance", ascending=False)

Path("reports/figures").mkdir(parents=True, exist_ok=True)

importance_df.to_csv(
    "reports/delivery_feature_importance.csv",
    index=False,
)

plot_df = importance_df.head(15).sort_values("importance")

plt.figure(figsize=(9, 6))
plt.barh(plot_df["feature"], plot_df["importance"])
plt.xlabel("Random Forest feature importance")
plt.title("Top Features for Delivery Time Prediction")
plt.tight_layout()
plt.savefig(
    "reports/figures/delivery_feature_importance.png",
    dpi=150,
)
plt.close()

print("Top 10 features:")
print(importance_df.head(10).to_string(index=False))
print("\nSaved reports/delivery_feature_importance.csv")
print("Saved reports/figures/delivery_feature_importance.png")