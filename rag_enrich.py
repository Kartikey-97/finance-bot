# rag_enrich.py
import os
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# === Config ===
ALERTS_CSV = "suspicious_alerts.csv"
ENRICHED_CSV = "suspicious_alerts_enriched.csv"
DOC_PATH = "/mnt/data/PS WEEK 1.pdf"   # <-- the uploaded problem-statement PDF (developer-supplied)

# Try to import RAG/embedding pieces (optional)
_HAS_RAG = False
try:
    import pathway as pw
    from pathway.xpacks.llm.document_store import DocumentStore
    from pathway.xpacks.llm.question_answering import BaseRAGQuestionAnswerer
    from pathway.xpacks.llm.llms import LiteLLMChat
    from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
    _HAS_RAG = True
except Exception as e:
    # Not fatal — we'll fall back to deterministic explanation or direct LLM call
    _HAS_RAG = False

def rule_based_explanation(row):
    # deterministic short fallback
    amount = float(row.get("amount") or 0)
    vel = float(row.get("velocity_sum_1h") or 0)
    risk = row.get("watchlist_risk") or ""
    reasons = []
    if amount > 2000:
        reasons.append(f"High-value transaction ${amount:.2f}.")
    if vel > 5000:
        reasons.append(f"Velocity ${vel:.2f} in 1h across {int(row.get('velocity_count_1h',0))} txns.")
    if risk not in ("", "None", None):
        reasons.append(f"Watchlist flag: {risk}.")
    if row.get("merchant"):
        reasons.append(f"Merchant: {row.get('merchant')}.")
    if row.get("location"):
        reasons.append(f"Location: {row.get('location')}.")
    if not reasons:
        reasons = ["No major risk indicators detected."]
    return "verdict=SUSPICIOUS || explanation=" + " ".join(reasons)

def init_rag():
    # Initialize Pathway DocumentStore + RAG (if possible)
    try:
        embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
        docs = pw.io.fs.read(DOC_PATH, format="binary", mode="static", with_metadata=True)
        vector_store = DocumentStore(
            docs=docs,
            retriever_factory=pw.stdlib.indexing.BruteForceKnnFactory(embedder=embedder),
        )
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")
        rag_app = BaseRAGQuestionAnswerer(
            llm=LiteLLMChat(
                model="gemini/gemini-1.5-flash",
                api_key=GEMINI_API_KEY,
                temperature=0.0,
            ),
            indexer=vector_store,
        )
        return rag_app
    except Exception as e:
        print("RAG initialization failed:", e)
        return None

def ask_rag(rag_app, row):
    # Build a short prompt using the alert and request reasoning referencing the PDF.
    prompt = (
        f"Using the uploaded compliance policy and AML docs, analyze this transaction.\n"
        f"Amount: {row.get('amount')}\n"
        f"Velocity (1h): {row.get('velocity_sum_1h')} across {row.get('velocity_count_1h')}\n"
        f"Watchlist: {row.get('watchlist_risk')}\n"
        f"Merchant: {row.get('merchant')}\n"
        f"Location: {row.get('location')}\n\n"
        "Answer with a short verdict (SUSPICIOUS/OK) and a concise reason (<= 40 words)."
    )
    try:
        # rag_app.run_sync returns a string in this environment (based on your previous code)
        resp = rag_app.run_sync(prompt)
        if not resp or not isinstance(resp, str):
            raise RuntimeError("Empty or invalid LLM reply")
        # normalize output a bit
        reply = resp.strip()
        if not reply.lower().startswith("verdict"):
            return "verdict=SUSPICIOUS || explanation=" + reply
        return reply
    except Exception as e:
        print("RAG call failed:", e)
        return None

def enrich_all():
    # load alerts
    if not os.path.exists(ALERTS_CSV):
        print("No alerts CSV yet:", ALERTS_CSV)
        return
    df = pd.read_csv(ALERTS_CSV)
    # Ensure we don't overwrite merchant/location columns if they exist in the raw CSV:
    if "merchant" not in df.columns:
        df["merchant"] = ""
    if "location" not in df.columns:
        df["location"] = ""
    # init rag once
    rag_app = init_rag() if _HAS_RAG else None

    # iterate rows and enrich where analysis is missing or default
    out_rows = []
    for _, row in df.iterrows():
        current_analysis = (str(row.get("analysis") or "")).strip()
        # if analysis already exists and isn't empty, skip enriching (but you can override if you want)
        if current_analysis not in ("", "None"):
            out_rows.append(row.to_dict())
            continue
        # Try RAG if available
        enriched = None
        if rag_app is not None:
            enriched = ask_rag(rag_app, row)
        if not enriched:
            enriched = rule_based_explanation(row)
        d = row.to_dict()
        d["analysis"] = enriched
        out_rows.append(d)

    out_df = pd.DataFrame(out_rows)
    # Atomic write
    tmp = ENRICHED_CSV + ".tmp"
    out_df.to_csv(tmp, index=False)
    os.replace(tmp, ENRICHED_CSV)
    print(f"Wrote enriched alerts -> {ENRICHED_CSV}")

if __name__ == "__main__":
    # single run (call in a loop or cron as you prefer)
    enrich_all()
