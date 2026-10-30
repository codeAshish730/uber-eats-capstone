# Uber Eats Operational & Customer Intelligence System

## 1. Project Overview

This project uses synthetic food-delivery data to explore customer behavior, restaurant performance, customer feedback, delivery-time prediction, and demand forecasting feasibility.

It combines data preparation, exploratory data analysis, unsupervised learning, natural language processing, regression modeling, and business recommendations.

**Data disclaimer:** All datasets are synthetic and do not represent actual Uber Eats customer or operational records.

## 2. Business Objectives

- Segment customers by ordering behavior and spending.
- Group restaurants by operational performance.
- Explore customer review sentiment.
- Evaluate machine-learning approaches for delivery-time prediction.
- Assess whether the available data supports demand forecasting.
- Translate analytical findings into operational recommendations.

## 3. Datasets

Seven synthetic CSV datasets are stored in `data/raw/`:

- `customers.csv`
- `restaurants.csv`
- `drivers.csv`
- `orders.csv`
- `deliveries.csv`
- `reviews.csv`
- `payments.csv`

The data dictionary is available in `data_dictionary.md`.

## 4. Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Joblib
- Jupyter Notebook
- Git

## 5. Project Structure

```text
uber-eats-capstone/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebook/
│   └── uber_eats_eda.ipynb
├── reports/
│   └── figures/
├── src/
├── data_dictionary.md
├── requirements.txt
└── README.md
```

## 6. Analysis and Modeling

### Customer Segmentation
- Customer feature engineering
- K-Means clustering and cluster evaluation
- DBSCAN clustering
- PCA visualizations and cluster profiles

The best tested K-Means configuration used two clusters, with a silhouette score of approximately 0.332.

### Restaurant Segmentation
Restaurant features include order volume, revenue, cancellation rate, ratings, and delivery time. The best tested two-cluster K-Means solution had a silhouette score of approximately 0.257.

### Review Sentiment Analysis
- Review text cleaning and tokenization
- TF-IDF feature extraction
- SVM-based sentiment classification
- VADER comparison and sentiment insight reports

The SVM sentiment model achieved approximately 37% test accuracy and a macro F1 score of 0.33. These results are weak and should not be used for automated high-impact decisions.

### Delivery-Time Regression
Baseline, Ridge Regression, and Random Forest models were evaluated.

Ridge Regression achieved a test MAE of approximately 14.792 minutes and an R² of approximately -0.013. The results indicate weak predictive performance.

### Demand Forecasting Feasibility
The available order timestamps cover only a very short time window, producing one hourly aggregate. Decomposition, stationarity testing, ARIMA, and Prophet forecasting could not be meaningfully evaluated with the available time series.

## 7. End-to-End Pipeline

Run the project pipeline from the repository root:

```bash
source .venv/bin/activate
python src/run_pipeline.py
```

The pipeline runs the existing preparation, clustering, sentiment, delivery evaluation, and forecasting-feasibility scripts. Review `reports/pipeline_execution_report.txt` and `reports/pipeline_execution_summary.csv` for execution results.

## 8. Key Limitations

- The data is synthetic and may not represent real-world patterns.
- Review text is repetitive, and identical text can have conflicting ratings.
- Delivery-time features lack traffic, weather, preparation time, and pickup waiting time.
- Order timestamps do not provide sufficient historical depth for forecasting.
- Tip prediction has not been implemented because a suitable target was not identified.
- Feature importance and correlation do not establish causation.

## 9. Business Recommendations

- Test retention initiatives for frequent, high-spending customers.
- Investigate cancellation rates, ratings, and delivery delays among restaurants with weaker operational metrics.
- Validate customer complaints against reliable feedback before taking action.
- Collect richer delivery and historical demand data before deploying predictive models.

## 10. Final Deliverables

- Synthetic datasets and data dictionary
- Data preparation and analysis scripts
- EDA notebook
- Customer and restaurant clustering outputs
- Sentiment processing and evaluation reports
- Delivery regression evaluation and feature importance
- Demand forecasting feasibility reports
- Business insights and pipeline execution reports

## 11. Future Improvements

Collect more representative data, expand the historical period, improve validation, add relevant operational features, and monitor model performance against measurable business outcomes.
