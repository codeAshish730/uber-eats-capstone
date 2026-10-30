import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

STEPS = [
    ("Customer feature engineering", "src/prepare_customer_features.py"),
    ("Customer segmentation", "src/customer_clustering.py"),
    ("Customer cluster profiles", "src/customer_cluster_profiles.py"),
    ("Restaurant feature engineering", "src/prepare_restaurant_features.py"),
    ("Restaurant segmentation", "src/restaurant_clustering.py"),
    ("Restaurant cluster profiles", "src/restaurant_cluster_profiles.py"),
    ("Review cleaning", "src/clean_reviews.py"),
    ("TF-IDF feature extraction", "src/tfidf_reviews.py"),
    ("Sentiment model training", "src/train_sentiment.py"),
    ("Sentiment insights", "src/sentiment_insights.py"),
    ("Delivery feature engineering", "src/prepare_delivery_features.py"),
    ("Delivery EDA", "src/delivery_eda.py"),
    ("Delivery model evaluation", "src/train_delivery_models.py"),
    ("Delivery feature importance", "src/delivery_feature_importance.py"),
    ("Hourly demand preparation", "src/prepare_hourly_demand.py"),
    ("Demand exploration", "src/explore_hourly_demand.py"),
    ("Demand decomposition check", "src/demand_decomposition.py"),
    ("Demand stationarity check", "src/test_demand_stationarity.py"),
    ("ARIMA feasibility check", "src/train_arima.py"),
    ("Prophet feasibility check", "src/train_prophet.py"),
    ("Demand model comparison", "src/compare_demand_models.py"),
]

results = []

print("=" * 65)
print("UBER EATS END-TO-END ML PIPELINE")
print("=" * 65)
print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

for name, script in STEPS:
    path = ROOT / script

    print(f"\n{'-' * 65}")
    print(f"STEP: {name}")

    if not path.exists():
        status = "MISSING SCRIPT"
        details = f"Script not found: {script}"
        print(details)
    else:
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=300,
            )

            status = "SUCCESS" if result.returncode == 0 else "FAILED"
            details = (result.stdout or "").strip()

            if result.returncode != 0:
                details += "\n" + (result.stderr or "").strip()

            print(details[-2500:] if details else status)

        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
            details = "Script exceeded 300 seconds."
            print(details)

    results.append({
        "step": name,
        "script": script,
        "status": status,
        "details": details[-3000:],
    })

import pandas as pd

summary = pd.DataFrame([
    {
        "step": item["step"],
        "script": item["script"],
        "status": item["status"],
    }
    for item in results
])

summary.to_csv(REPORTS / "pipeline_execution_summary.csv", index=False)

with open(REPORTS / "pipeline_execution_report.txt", "w") as file:
    file.write("UBER EATS END-TO-END ML PIPELINE REPORT\n")
    file.write("=" * 45 + "\n")
    file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for item in results:
        file.write(f"\n{item['step']}: {item['status']}\n")
        file.write(f"Script: {item['script']}\n")
        file.write(item["details"] + "\n")

    file.write("\n\nEXECUTION SUMMARY\n")
    file.write(summary.to_string(index=False))
    file.write("\n\nDATA LIMITATIONS\n")
    file.write(
        "- Demand history currently contains only one hourly observation.\n"
        "- Tip prediction is not included until a suitable tip target is confirmed.\n"
        "- A successful script run does not by itself establish model quality.\n"
    )

print("\n" + "=" * 65)
print("PIPELINE EXECUTION SUMMARY")
print("=" * 65)
print(summary.to_string(index=False))
print("\nSummary saved to reports/pipeline_execution_summary.csv")
print("Full report saved to reports/pipeline_execution_report.txt")