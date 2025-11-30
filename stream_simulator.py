import csv
import random
import time
from datetime import datetime

CSV_PATH = "./data/stream/transactions.csv"

USERS = ["u101", "u202", "u303", "u404", "u505"]
MERCHANTS = ["Amazon", "Flipkart", "Myntra", "Uber", "Zomato", "Swiggy"]
LOCATIONS = ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Kolkata"]
STATUSES = ["APPROVED", "DECLINED"]

def append_transaction():
    now = datetime.utcnow().replace(microsecond=0).isoformat()

    row = [
        now,
        random.choice(USERS),
        str(random.randint(50, 8000)),          # amount
        random.choice(MERCHANTS),
        random.choice(LOCATIONS),
        random.choice(STATUSES)
    ]

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print("Generated:", row)

if __name__ == "__main__":
    print("🔥 Real-time simulator started. Writing a transaction every 3 seconds...")
    while True:
        append_transaction()
        time.sleep(30)
