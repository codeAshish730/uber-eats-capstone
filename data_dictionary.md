# Uber Eats Operational & Customer Intelligence System

## Data Dictionary

All datasets are synthetically generated for educational and analytical purposes.

### 1. customers.csv

| Column | Description |
|---|---|
| customer_id | Unique customer identifier |
| customer_name | Synthetic customer name |
| city | Customer city |
| signup_date | Date the customer registered |

### 2. restaurants.csv

| Column | Description |
|---|---|
| restaurant_id | Unique restaurant identifier |
| restaurant_name | Synthetic restaurant name |
| city | Restaurant city |
| cuisine | Restaurant cuisine category |
| rating | Restaurant rating from 3.0 to 5.0 |

### 3. drivers.csv

| Column | Description |
|---|---|
| driver_id | Unique driver identifier |
| driver_name | Synthetic driver name |
| vehicle_type | Driver's vehicle type |

### 4. orders.csv

| Column | Description |
|---|---|
| order_id | Unique order identifier |
| customer_id | Customer who placed the order |
| restaurant_id | Restaurant receiving the order |
| order_date | Date and time of the order |
| order_amount | Order amount in unspecified currency units |
| order_status | Order status: Delivered, Cancelled, or Delayed |

### 5. deliveries.csv

| Column | Description |
|---|---|
| delivery_id | Unique delivery identifier |
| order_id | Associated order identifier |
| driver_id | Assigned driver identifier |
| delivery_time_min | Delivery duration in minutes |
| distance_km | Delivery distance in kilometres |

### 6. reviews.csv

| Column | Description |
|---|---|
| review_id | Unique review identifier |
| order_id | Associated order identifier |
| review_text | Synthetic customer review |
| review_rating | Customer rating from 1 to 5 |

### 7. payments.csv

| Column | Description |
|---|---|
| payment_id | Unique payment identifier |
| order_id | Associated order identifier |
| payment_method | Payment method used |
| payment_status | Payment status |

## Notes

- All records are synthetic, not real Uber Eats customer or operational records.
- IDs connect related datasets.
- The order amount has no currency specified.
- The generated data is a starter dataset and may need quality checks before modelling.
