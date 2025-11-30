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

# Use the uploaded problem statement as a document (developer-provided file)
# Path from conversation: /mnt/data/PS WEEK 1.pdf
DOCS_FOLDER = "./data/documents/PS_WEEK_1.pdf" # kept as file path; you may replace with a folder "./data/documents"
TRANSACTIONS_CSV = "./data/stream/transactions.csv"
WATCHLIST_CSV = "./data/watchlist/watchlist.csv"
OUTPUT_CSV = "suspicious_alerts.csv"

if not os.path.exists(TRANSACTIONS_CSV):
    raise FileNotFoundError(f"Transactions CSV not found at {TRANSACTIONS_CSV}")
if not os.path.exists(WATCHLIST_CSV):
    raise FileNotFoundError(f"Watchlist CSV not found at {WATCHLIST_CSV}")

# -----------------------
# Schemas (plain Python types)
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
# Optional RAG / embedding setup
# -----------------------
if _HAS_RAG:
    try:
        print("RAG: loading embedder and document store (may take a moment)...")
        embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

        # DocumentStore can accept a file path (binary) or a folder; keep flexible.
        docs = pw.io.fs.read(
            DOCS_FOLDER,
            format="binary",
            mode="static",
            with_metadata=True,
        )

        vector_store = DocumentStore(
            docs=docs,
            retriever_factory=pw.stdlib.indexing.BruteForceKnnFactory(embedder=embedder),
        )

        # Note: creating BaseRAGQuestionAnswerer object is fine, but xpack method signatures
        # vary by version. We will not call rag_app directly in the pipeline to avoid version mismatches.
        if not GEMINI_API_KEY:
            print("Warning: GEMINI_API_KEY not set. RAG LLM calls disabled.")
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
    except Exception as e:
        print("Warning: failed to initialize RAG components; continuing without RAG.", e)
        _HAS_RAG = False
        rag_app = None
else:
    print("RAG xpack not available; using rule-based explanations instead.")
    rag_app = None

# -----------------------
# Read CSV sources
# -----------------------
raw_tx = pw.io.csv.read(
    TRANSACTIONS_CSV,
    schema=TxSchema,
    mode="streaming",
)

watchlist = pw.io.csv.read(
    WATCHLIST_CSV,
    schema=WatchSchema,
    mode="static",
)

# -----------------------
# Typed UDFs (avoid ANY)
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
def safe_str(x):
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""

# Create a lexicographic meta string for reducers
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

# -----------------------
# Create typed transaction stream (amount as float)
# -----------------------
transactions = raw_tx.with_columns(
    amount_f=safe_float(pw.this.amount),
    time_str=pw.this.time,
    parsed_time=pw.this.time.dt.strptime("%Y-%m-%dT%H:%M:%S"),
)

# -----------------------
# Build meta string for lexicographic reducer
# -----------------------
t_stream = transactions.with_columns(
    tx_meta=make_meta_str(pw.this.time_str, pw.this.amount_f, pw.this.location, pw.this.merchant, pw.this.status)
)

# -----------------------
# Windowing (60 min sliding, 1 min hop) and reducers
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
        velocity_sum_1h=pw.reducers.sum(pw.this.amount_f),
        velocity_count_1h=pw.reducers.count(pw.this.amount_f),
        latest_meta=pw.reducers.max(pw.this.tx_meta),
    )
)

# -----------------------
# Unpack latest_meta into typed columns
# -----------------------
unpacked = window_stats.with_columns(
    latest_time=meta_get_time(pw.this.latest_meta),
    latest_amount=meta_get_amount(pw.this.latest_meta),
    latest_location=meta_get_location(pw.this.latest_meta),
    latest_merchant=meta_get_merchant(pw.this.latest_meta),
    latest_status=meta_get_status(pw.this.latest_meta),
)

# -----------------------
# Join with watchlist (left join)
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
    velocity_sum_1h=pw.left.velocity_sum_1h,
    velocity_count_1h=pw.left.velocity_count_1h,
    watchlist_risk=pw.right.risk_level,
    watchlist_notes=pw.right.notes,
)

# -----------------------
# Re-enforce numeric typed columns after join
# -----------------------
full_context = full_context.with_columns(
    amount_f=safe_float(pw.this.amount),
    velocity_sum_f=safe_float(pw.this.velocity_sum_1h),
)

# -----------------------
# Alerts: numeric comparisons on typed floats and watchlist presence
# -----------------------
alerts_trigger = full_context.filter(
    (pw.this.amount_f > 2000.0)
    | (pw.this.velocity_sum_f > 5000.0)
    | (~pw.this.watchlist_risk.is_none())
)

# -----------------------
# Explainers: short verdict + detailed reason (UDF used inside pipeline)
# This guarantees a deterministic, explainable output even if RAG is unavailable.
# -----------------------
@pw.udf(return_type=str)
def rule_based_explanation(amount: float, vel_sum: float, vel_count: float, risk: str, notes: str, merchant: str, location: str) -> str:
    # Short verdict
    short = "SUSPICIOUS" if (amount is not None and amount > 2000.0) or (vel_sum is not None and vel_sum > 5000.0) or (risk not in (None, "", "None")) else "OK"
    # Reasons (assemble multiple signals)
    reasons = []
    if amount is not None and amount > 2000.0:
        reasons.append(f"Single-transaction amount ${amount:.2f} exceeds 2000 threshold.")
    if vel_sum is not None and vel_sum > 5000.0:
        reasons.append(f"1h velocity ${vel_sum:.2f} exceeds 5000 threshold across {int(vel_count or 0)} txns.")
    if risk not in (None, "", "None"):
        reasons.append(f"Watchlist flag: {risk}. Notes: {notes or 'N/A'}.")
    if merchant:
        reasons.append(f"Merchant: {merchant}.")
    if location:
        reasons.append(f"Location: {location}.")
    if not reasons:
        reasons.append("No strong rule-based signals, transaction within thresholds.")
    # Detailed paragraph
    detailed = " ".join(reasons) + " Recommend: investigate if multiple signals present; escalate for manual review if watchlist flagged."
    # Combine short + detailed
    return f"verdict={short} || explanation={detailed}"

# If RAG is available and you want to integrate later, you can call rag_app outside the pipeline on selected alerts.
# For now we always include the rule-based explanation so pipeline is robust.
analysis_col = rule_based_explanation(
    pw.this.amount_f,
    pw.this.velocity_sum_f,
    pw.this.velocity_count_1h,
    pw.this.watchlist_risk,
    pw.this.watchlist_notes,
    pw.this.merchant,
    pw.this.location,
)

final_results = alerts_trigger.select(
    time=pw.this.latest_time,      
    user_id=pw.this.user_id,
    amount=pw.this.amount_f,
    velocity_sum_1h=pw.this.velocity_sum_f,
    watchlist_risk=pw.this.watchlist_risk,
    analysis=analysis_col,
)

# -----------------------
# Output to CSV
# -----------------------
pw.io.csv.write(final_results, OUTPUT_CSV)

# -----------------------
# Run pipeline
# -----------------------
if __name__ == "__main__":
    print("🚀 Compliance Sentinel (patched) running...")
    print(f"Watching: {TRANSACTIONS_CSV}")
    if _HAS_RAG:
        print("RAG components initialized (LLM disabled unless GEMINI_API_KEY set).")
    else:
        print("RAG not used — rule-based/explainable analysis active.")
    pw.run() 