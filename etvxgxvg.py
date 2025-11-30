# app.py (Pathway-only, leaves analysis PENDING for external RAG worker)
import os
from datetime import timedelta
from dotenv import load_dotenv

import pathway as pw
from pathway.internals.join_mode import JoinMode

load_dotenv()
TRANSACTIONS_CSV = "./data/stream/transactions.csv"
WATCHLIST_CSV = "./data/watchlist/watchlist.csv"
OUTPUT_CSV = "suspicious_alerts.csv"

if not os.path.exists(TRANSACTIONS_CSV):
    raise FileNotFoundError(f"Transactions CSV not found at {TRANSACTIONS_CSV}")
if not os.path.exists(WATCHLIST_CSV):
    raise FileNotFoundError(f"Watchlist CSV not found at {WATCHLIST_CSV}")

class WatchSchema(pw.Schema):
    entity_id: str
    risk_level: str
    notes: str

class TxSchema(pw.Schema):
    time: str
    user_id: str
    amount: str
    merchant: str
    location: str
    status: str

@pw.udf(return_type=float)
def safe_float(x):
    try:
        if x is None:
            return 0.0
        return float(str(x).strip())
    except Exception:
        return 0.0

@pw.udf(return_type=str)
def make_meta_str(time_iso_s: str, amount_num: float, location: str, merchant: str, status: str) -> str:
    t = "" if time_iso_s is None else str(time_iso_s)
    try:
        amt = f"{float(amount_num):.6f}"
    except Exception:
        amt = "0.0"
    loc = "" if location is None else str(location)
    merch = "" if merchant is None else str(merchant)
    stat = "" if status is None else str(status)
    return f"{t}|{amt}|{loc}|{merch}|{stat}"

@pw.udf(return_type=str)
def meta_get_time(meta: str) -> str:
    try:
        if not meta:
            return ""
        return meta.split("|", 4)[0]
    except Exception:
        return ""

@pw.udf(return_type=float)
def meta_get_amount(meta: str) -> float:
    try:
        if not meta:
            return 0.0
        parts = meta.split("|", 4)
        return float(parts[1]) if len(parts) > 1 and parts[1] != "" else 0.0
    except Exception:
        return 0.0

@pw.udf(return_type=str)
def meta_get_location(meta: str) -> str:
    try:
        if not meta:
            return ""
        parts = meta.split("|", 4)
        return parts[2] if len(parts) > 2 else ""
    except Exception:
        return ""

@pw.udf(return_type=str)
def meta_get_merchant(meta: str) -> str:
    try:
        if not meta:
            return ""
        parts = meta.split("|", 4)
        return parts[3] if len(parts) > 3 else ""
    except Exception:
        return ""

@pw.udf(return_type=str)
def meta_get_status(meta: str) -> str:
    try:
        if not meta:
            return ""
        parts = meta.split("|", 4)
        return parts[4] if len(parts) > 4 else ""
    except Exception:
        return ""

# read sources
raw_tx = pw.io.csv.read(TRANSACTIONS_CSV, schema=TxSchema, mode="streaming")
watchlist = pw.io.csv.read(WATCHLIST_CSV, schema=WatchSchema, mode="static")

# typed stream
transactions = raw_tx.with_columns(
    amount_f=safe_float(pw.this.amount),
    time_str=pw.this.time,
    parsed_time=pw.this.time.dt.strptime("%Y-%m-%dT%H:%M:%S"),
)

t_stream = transactions.with_columns(
    tx_meta=make_meta_str(pw.this.time_str, pw.this.amount_f, pw.this.location, pw.this.merchant, pw.this.status)
)

window_stats = (
    t_stream
    .windowby(
        pw.this.parsed_time,
        window=pw.temporal.sliding(duration=timedelta(minutes=60), hop=timedelta(minutes=1)),
    )
    .groupby(pw.this.user_id)
    .reduce(
        pw.this.user_id,
        velocity_sum_1h=pw.reducers.sum(pw.this.amount_f),
        velocity_count_1h=pw.reducers.count(pw.this.amount_f),
        latest_meta=pw.reducers.max(pw.this.tx_meta),
    )
)

unpacked = window_stats.with_columns(
    latest_time=meta_get_time(pw.this.latest_meta),
    latest_amount=meta_get_amount(pw.this.latest_meta),
    latest_location=meta_get_location(pw.this.latest_meta),
    latest_merchant=meta_get_merchant(pw.this.latest_meta),
    latest_status=meta_get_status(pw.this.latest_meta),
)

full_context = unpacked.join(
    watchlist,
    pw.left.user_id == pw.right.entity_id,
    how=JoinMode.LEFT,
).select(
    user_id=pw.left.user_id,
    latest_time=pw.left.latest_time,
    amount=pw.left.latest_amount,
    location=pw.left.latest_location,
    merchant=pw.left.latest_merchant,
    status=pw.left.latest_status,
    velocity_sum_1h=pw.left.velocity_sum_1h,
    velocity_count_1h=pw.left.velocity_count_1h,
    watchlist_risk=pw.right.risk_level,
    watchlist_notes=pw.right.notes,
).with_columns(
    amount_f=safe_float(pw.this.amount),
    velocity_sum_f=safe_float(pw.this.velocity_sum_1h),
)

alerts_trigger = full_context.filter(
    (pw.this.amount_f > 2000.0)
    | (pw.this.velocity_sum_f > 5000.0)
    | (~pw.this.watchlist_risk.is_none())
)

# provide a PENDING placeholder for external enricher
@pw.udf(return_type=str)
def pending_marker():
    return "PENDING"

analysis_col = pending_marker()

final_results = alerts_trigger.select(
    time=pw.this.latest_time,
    user_id=pw.this.user_id,
    amount=pw.this.amount_f,
    velocity_sum_1h=pw.this.velocity_sum_f,
    watchlist_risk=pw.this.watchlist_risk,
    analysis=analysis_col,
)

# write CSV for the external enricher to pick up
pw.io.csv.write(final_results, OUTPUT_CSV)

if __name__ == "__main__":
    print("🚀 Compliance Sentinel (Pathway pipeline) running...")
    print(f"Watching: {TRANSACTIONS_CSV}")
    pw.run()
