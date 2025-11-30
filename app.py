# app.py (patched, Pathway 0.16.4 compatible)
import os
from datetime import timedelta
from dotenv import load_dotenv

import pathway as pw
from pathway.internals.join_mode import JoinMode

# Try to import optional RAG/LLM pieces — degrade gracefully if missing
try:
    from pathway.xpacks.llm.document_store import DocumentStore
    from pathway.xpacks.llm.question_answering import BaseRAGQuestionAnswerer
    from pathway.xpacks.llm.llms import LiteLLMChat
    from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
    _HAS_RAG = True
except Exception:
    DocumentStore = None
    BaseRAGQuestionAnswerer = None
    LiteLLMChat = None
    SentenceTransformerEmbedder = None
    _HAS_RAG = False

# -----------------------
# Config
# -----------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DOCS_FOLDER = "./data/documents/PS_WEEK_1.pdf"
TRANSACTIONS_CSV = "./data/stream/transactions.csv"
WATCHLIST_CSV = "./data/watchlist/watchlist.csv"
OUTPUT_CSV = "suspicious_alerts.csv"

if not os.path.exists(TRANSACTIONS_CSV):
    raise FileNotFoundError(f"Transactions CSV not found at {TRANSACTIONS_CSV}")
if not os.path.exists(WATCHLIST_CSV):
    raise FileNotFoundError(f"Watchlist CSV not found at {WATCHLIST_CSV}")

# -----------------------
# Schemas
# -----------------------
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

# -----------------------
# Optional RAG setup
# -----------------------
if _HAS_RAG:
    try:
        embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
        docs = pw.io.fs.read(DOCS_FOLDER, format="binary", mode="static", with_metadata=True)
        vector_store = DocumentStore(
            docs=docs,
            retriever_factory=pw.stdlib.indexing.BruteForceKnnFactory(embedder=embedder),
        )
        if not GEMINI_API_KEY:
            _HAS_RAG = False
        else:
            rag_app = BaseRAGQuestionAnswerer(
                llm=LiteLLMChat(
                    model="gemini/gemini-1.5-flash",
                    api_key=GEMINI_API_KEY,
                    temperature=0.0,
                ),
                indexer=vector_store,
            )
    except Exception:
        _HAS_RAG = False
        rag_app = None
else:
    rag_app = None

# -----------------------
# Read CSVs
# -----------------------
raw_tx = pw.io.csv.read(TRANSACTIONS_CSV, schema=TxSchema, mode="streaming")
watchlist = pw.io.csv.read(WATCHLIST_CSV, schema=WatchSchema, mode="static")

# -----------------------
# UDFs
# -----------------------
@pw.udf(return_type=float)
def safe_float(x):
    try:
        if x is None:
            return 0.0
        return float(str(x).strip())
    except Exception:
        return 0.0

@pw.udf(return_type=str)
def make_meta_str(t, amt, loc, merch, stat):
    try:
        amt2 = f"{float(amt):.6f}"
    except:
        amt2 = "0.0"
    return f"{t or ''}|{amt2}|{loc or ''}|{merch or ''}|{stat or ''}"

@pw.udf(return_type=str)
def meta_get_time(m):
    return m.split("|", 4)[0] if m else ""

@pw.udf(return_type=float)
def meta_get_amount(m):
    try:
        return float(m.split("|", 4)[1])
    except:
        return 0.0

@pw.udf(return_type=str)
def meta_get_location(m):
    try:
        return m.split("|", 4)[2]
    except:
        return ""

@pw.udf(return_type=str)
def meta_get_merchant(m):
    try:
        return m.split("|", 4)[3]
    except:
        return ""

@pw.udf(return_type=str)
def meta_get_status(m):
    try:
        return m.split("|", 4)[4]
    except:
        return ""

# -----------------------
# Typed TX stream
# -----------------------
transactions = raw_tx.with_columns(
    amount_f=safe_float(pw.this.amount),
    time_str=pw.this.time,
    parsed_time=pw.this.time.dt.strptime("%Y-%m-%dT%H:%M:%S"),
)

# -----------------------
# Build meta
# -----------------------
t_stream = transactions.with_columns(
    tx_meta=make_meta_str(
        pw.this.time_str,
        pw.this.amount_f,
        pw.this.location,
        pw.this.merchant,
        pw.this.status,
    )
)

# -----------------------
# Window (60 min slide, 1 min hop)
# -----------------------
window_stats = (
    t_stream
    .windowby(
        pw.this.parsed_time,
        window=pw.temporal.sliding(duration=timedelta(minutes=60), hop=timedelta(minutes=1)),
    )
    .groupby(pw.this.user_id)
    .reduce(
        pw.this.user_id,
        velocity_sum_1h=pw.reducers.sum(pw.this.amount_f),     # ✅ sum
        velocity_count_1h=pw.reducers.count(pw.this.amount_f), # ✅ count
        latest_meta=pw.reducers.max(pw.this.tx_meta),
    )
)

# -----------------------
# Compute average manually
# -----------------------
window_stats = window_stats.with_columns(
    velocity_avg_1h = pw.this.velocity_sum_1h / (pw.this.velocity_count_1h + 1e-9)   # ✅ FIX
)

# -----------------------
# Unpack meta
# -----------------------
unpacked = window_stats.with_columns(
    latest_time=meta_get_time(pw.this.latest_meta),
    latest_amount=meta_get_amount(pw.this.latest_meta),
    latest_location=meta_get_location(pw.this.latest_meta),
    latest_merchant=meta_get_merchant(pw.this.latest_meta),
    latest_status=meta_get_status(pw.this.latest_meta),
)

# -----------------------
# Join with watchlist
# -----------------------
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

    velocity_avg_1h=pw.left.velocity_avg_1h,          # ✅ FIXED
    velocity_count_1h=pw.left.velocity_count_1h,

    watchlist_risk=pw.right.risk_level,
    watchlist_notes=pw.right.notes,
)

# -----------------------
# Re-type numeric
# -----------------------
full_context = full_context.with_columns(
    amount_f=safe_float(pw.this.amount),
    velocity_avg_f=safe_float(pw.this.velocity_avg_1h),  # ✅ FIX
)

# -----------------------
# Alerts
# -----------------------
alerts_trigger = full_context.filter(
    (pw.this.amount_f > 2000.0)
    | (pw.this.velocity_avg_f > 5000.0)       # ✅ FIX
    | (~pw.this.watchlist_risk.is_none())
)

# -----------------------
# Rule-based explanation
# -----------------------
@pw.udf(return_type=str)
def rule_based_explanation(amount, vel, cnt, risk, notes, merchant, location):
    short = "SUSPICIOUS" if (amount > 2000) or (vel > 5000) or (risk not in (None, "", "None")) else "OK"
    reasons = []
    if amount > 2000:
        reasons.append(f"Single transaction ${amount:.2f} exceeded threshold.")
    if vel > 5000:
        reasons.append(f"1h velocity ${vel:.2f} exceeded threshold across {int(cnt)} txns.")
    if risk not in (None, "", "None"):
        reasons.append(f"Watchlist flag: {risk}. Notes: {notes or 'N/A'}")
    if merchant:
        reasons.append(f"Merchant: {merchant}")
    if location:
        reasons.append(f"Location: {location}")
    if not reasons:
        reasons.append("No suspicious indicators.")
    return "verdict=" + short + " || " + " ".join(reasons)

analysis_col = rule_based_explanation(
    pw.this.amount_f,
    pw.this.velocity_avg_f,         # ✅ FIX
    pw.this.velocity_count_1h,
    pw.this.watchlist_risk,
    pw.this.watchlist_notes,
    pw.this.merchant,
    pw.this.location,
)

# -----------------------
# Final output
# -----------------------
final_results = alerts_trigger.select(
    time=pw.this.latest_time,
    user_id=pw.this.user_id,
    amount=pw.this.amount_f,
    velocity_avg_1h=pw.this.velocity_avg_f,   # ✅ FIX
    watchlist_risk=pw.this.watchlist_risk,
    analysis=analysis_col,
)

pw.io.csv.write(final_results, OUTPUT_CSV)

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    print("🚀 Compliance Sentinel running...")
    pw.run()
