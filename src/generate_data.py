import pandas as pd
from faker import Faker
import random
from pathlib import Path

fake = Faker()
random.seed(42)
Faker.seed(42)

OUTPUT = Path("data/raw")
OUTPUT.mkdir(parents=True, exist_ok=True)

customers = pd.DataFrame([
    {
        "customer_id": i,
        "customer_name": fake.name(),
        "city": fake.city(),
        "signup_date": fake.date_between(
            start_date="-2y", end_date="today"
        ).isoformat()
    }
    for i in range(1, 201)
])

cuisines = ["Indian", "Italian", "Chinese", "Mexican", "American"]
restaurants = pd.DataFrame([
    {
        "restaurant_id": i,
        "restaurant_name": fake.company(),
        "city": fake.city(),
        "cuisine": random.choice(cuisines),
        "rating": round(random.uniform(3.0, 5.0), 1)
    }
    for i in range(1, 31)
])

drivers = pd.DataFrame([
    {
        "driver_id": i,
        "driver_name": fake.name(),
        "vehicle_type": random.choice(
            ["Bike", "Scooter", "Car"]
        )
    }
    for i in range(1, 41)
])

orders = []
for i in range(1, 1001):
    order_date = fake.date_time_between(
        start_date="-6mo", end_date="now"
    )
    orders.append({
        "order_id": i,
        "customer_id": random.randint(1, 200),
        "restaurant_id": random.randint(1, 30),
        "order_date": order_date.isoformat(),
        "order_amount": round(random.uniform(10, 100), 2),
        "order_status": random.choice(
            ["Delivered", "Cancelled", "Delayed"]
        )
    })

orders = pd.DataFrame(orders)

deliveries = pd.DataFrame([
    {
        "delivery_id": i,
        "order_id": i,
        "driver_id": random.randint(1, 40),
        "delivery_time_min": random.randint(15, 70),
        "distance_km": round(random.uniform(1, 15), 1)
    }
    for i in range(1, 1001)
])

reviews = pd.DataFrame([
    {
        "review_id": i,
        "order_id": i,
        "review_text": random.choice([
            "Excellent food and quick delivery",
            "Food was cold and delivery was late",
            "Good quality and friendly driver",
            "Order was okay, nothing special",
            "Very disappointed with this experience",
            "Delicious food, will order again"
        ]),
        "review_rating": random.randint(1, 5)
    }
    for i in range(1, 501)
])

payments = pd.DataFrame([
    {
        "payment_id": i,
        "order_id": i,
        "payment_method": random.choice(
            ["Credit Card", "Debit Card", "UPI", "Cash"]
        ),
        "payment_status": random.choice(
            ["Successful", "Failed", "Refunded"]
        )
    }
    for i in range(1, 1001)
])

datasets = {
    "customers": customers,
    "restaurants": restaurants,
    "drivers": drivers,
    "orders": orders,
    "deliveries": deliveries,
    "reviews": reviews,
    "payments": payments
}

for name, df in datasets.items():
    df.to_csv(OUTPUT / f"{name}.csv", index=False)
    print(f"{name}.csv: {len(df)} rows")

print("\nAll datasets generated successfully!")