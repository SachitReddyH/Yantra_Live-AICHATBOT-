import os
import time
import re
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import pandas as pd
import chromadb
import cohere
from groq import Groq

# ==========================
# ENVIRONMENT
# ==========================
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPPORT_PHONE = os.getenv("YANTRALIVE_SUPPORT_PHONE", "+91-9876543210")
SUPPORT_EMAIL = os.getenv("YANTRALIVE_SUPPORT_EMAIL", "support@yantralive.com")

if not COHERE_API_KEY:
    raise RuntimeError("Missing COHERE_API_KEY in .env")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in .env")

co = cohere.Client(COHERE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# ==========================
# FASTAPI
# ==========================
app = FastAPI(
    title="YantraLive RAG Chatbot (Cohere + Groq)",
    version="1.2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Brochure storage
# ==========================
# Place PDFs in data/brochures (e.g. data/brochures/vj20.pdf)
BROCHURE_DIR = os.path.join("data", "brochures")
os.makedirs(BROCHURE_DIR, exist_ok=True)

# Mount static just for raw direct access fallback (not used for preview iframe)
app.mount("/static_brochures", StaticFiles(directory=BROCHURE_DIR), name="static_brochures")


def _build_brochure_map():
    m = {}
    if not os.path.isdir(BROCHURE_DIR):
        return m
    for fname in os.listdir(BROCHURE_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        name_no_ext = os.path.splitext(fname)[0]
        key = re.sub(r"[^a-z0-9]", "", name_no_ext.lower())
        m[key] = fname
    return m


BROCHURE_MAP = _build_brochure_map()


@app.get("/api/brochures/list")
def list_brochures():
    """
    Returns list of available brochures (normalized key -> filename).
    Useful for frontend diagnostics.
    """
    return JSONResponse(BROCHURE_MAP)


@app.get("/brochures/view/{filename}", response_class=HTMLResponse)
def brochure_view(filename: str):
    """
    Serve a small HTML wrapper page that embeds the PDF via an <object>.
    Embedding the wrapper (same-origin) avoids Chrome blocking.
    """
    safe = os.path.basename(filename)
    path = os.path.join(BROCHURE_DIR, safe)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Brochure not found")

    # raw URL (same origin) - this will be served by /brochures/raw/{filename}
    raw_url = f"/brochures/raw/{safe}"

    # Minimal HTML wrapper. User can open raw URL in new tab; wrapper displays PDF inline.
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{safe} — Brochure</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    html,body {{ height:100%; margin:0; background:#f7f7f7; }}
    .topbar {{ padding:10px; background:#fff; border-bottom:1px solid #eee; display:flex; gap:8px; align-items:center; }}
    .open-btn {{ padding:6px 10px; border-radius:6px; border:1px solid #ccc; background:#fff; text-decoration:none; color:#111; font-size:13px; }}
    .iframe-wrap {{ height: calc(100% - 52px); }}
    object {{ width:100%; height:100%; border:none; }}
  </style>
</head>
<body>
  <div class="topbar">
    <strong>{safe}</strong>
    <a class="open-btn" href="{raw_url}" target="_blank" rel="noopener noreferrer">Open in new tab</a>
    <a class="open-btn" href="{raw_url}" download>Download</a>
  </div>
  <div class="iframe-wrap" role="document">
    <!-- object tag lets browser use built-in PDF viewer; same-origin wrapper avoids framing blocks -->
    <object data="{raw_url}" type="application/pdf" aria-label="brochure">
      <p>Your browser does not support inline PDF viewing. <a href="{raw_url}" target="_blank">Open brochure</a></p>
    </object>
  </div>
</body>
</html>"""
    return HTMLResponse(content=html, status_code=200)


@app.get("/brochures/raw/{filename}")
def brochure_raw(filename: str):
    """
    Serve raw PDF bytes with Content-Disposition:inline to encourage in-browser preview.
    FileResponse supports Range requests so partial content (206) will work.
    """
    safe = os.path.basename(filename)
    path = os.path.join(BROCHURE_DIR, safe)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Brochure not found")

    # Content-Disposition inline to encourage rendering (not download)
    headers = {
        "Content-Disposition": f'inline; filename="{safe}"'
    }
    return FileResponse(path, media_type="application/pdf", headers=headers)


# ==========================
# CHROMA (we pass embeddings manually)
# ==========================
chroma_client = chromadb.Client()


def create_or_get_collection(name: str):
    try:
        return chroma_client.create_collection(name=name)
    except Exception:
        return chroma_client.get_collection(name=name)


end_customer_collection = create_or_get_collection("yantra_end_customer")
spare_parts_collection = create_or_get_collection("yantra_spare_parts")
dealer_collection = create_or_get_collection("yantra_dealers")

# ==========================
# Pydantic MODELS
# ==========================
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    answer: str
    used_context: List[str]
    from_fallback: bool = False
    brochure_url: Optional[str] = None

# ==========================
# COHERE EMBEDDINGS + RETRY
# ==========================
EMBED_MODEL = "embed-english-v3.0"
EMBED_BATCH_SIZE = 64
EMBED_BATCH_SLEEP_SECONDS = 1.0


def _cohere_embed_with_retry(
    texts: List[str],
    input_type: str,
    label: str = "",
    max_retries: int = 5,
) -> List[List[float]]:
    for attempt in range(max_retries):
        try:
            resp = co.embed(
                texts=texts,
                model=EMBED_MODEL,
                input_type=input_type,
            )
            return resp.embeddings
        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg:
                wait = 5 * (attempt + 1)
                print(
                    f"[COHERE] Rate limited while embedding {label} "
                    f"(attempt {attempt + 1}/{max_retries}). Sleeping {wait}s."
                )
                time.sleep(wait)
                continue

            print(f"[COHERE] Non-rate-limit error while embedding {label}: {e}")
            raise

    raise RuntimeError(f"Cohere embed retries exceeded for {label}")


def embed_documents(texts: List[str]) -> List[List[float]]:
    return _cohere_embed_with_retry(
        texts=texts,
        input_type="search_document",
        label=f"documents batch (size={len(texts)})",
    )


def embed_query(text: str) -> List[float]:
    embeddings = _cohere_embed_with_retry(
        texts=[text],
        input_type="search_query",
        label="user query",
    )
    return embeddings[0]


# ==========================
# LOAD CSV + INDEX (Cohere -> Chroma)
# ==========================
DATA_DIR = "data"
END_CUSTOMER_FILE = os.path.join(DATA_DIR, "end_customer.csv")
SPARE_PARTS_FILE = os.path.join(DATA_DIR, "spare_parts.csv")
DEALERS_FILE = os.path.join(DATA_DIR, "dealers.csv")


def load_and_index_one(path: str, collection, tag: str):
    if not os.path.exists(path):
        print(f"[INFO] Dataset not found for {tag}: {path} (skipping)")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"[WARN] Dataset {tag} is empty: {path}")
        return

    documents: List[str] = []
    ids: List[str] = []

    for i, row in df.iterrows():
        row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        documents.append(f"[{tag}] {row_text}")
        ids.append(f"{tag.lower()}_row_{i}")

    total_docs = len(documents)
    print(f"[INDEX] Starting indexing for {tag}: {total_docs} rows")

    try:
        for start in range(0, total_docs, EMBED_BATCH_SIZE):
            end = min(start + EMBED_BATCH_SIZE, total_docs)
            batch_docs = documents[start:end]
            batch_ids = ids[start:end]

            try:
                batch_vectors = embed_documents(batch_docs)
            except Exception as batch_err:
                print(
                    f"[WARN] Failed to embed batch {start}:{end} for {tag}: {batch_err}"
                )
                continue

            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_vectors,
            )
            print(f"[INDEX] Indexed rows {start} to {end - 1} for {tag}")
            time.sleep(EMBED_BATCH_SLEEP_SECONDS)

        print(
            f"[INDEX] Finished indexing {total_docs} rows for {tag} using Cohere embeddings."
        )
    except Exception as e:
        print(f"[WARN] Failed to embed/index dataset {tag} with Cohere: {e}")
        print("[WARN] Starting server without this vector index; chat may fallback.")


def load_all_datasets():
    load_and_index_one(END_CUSTOMER_FILE, end_customer_collection, "END_CUSTOMER")
    time.sleep(2)
    load_and_index_one(SPARE_PARTS_FILE, spare_parts_collection, "SPARE_PARTS")
    time.sleep(2)
    load_and_index_one(DEALERS_FILE, dealer_collection, "DEALERS")


try:
    load_all_datasets()
except Exception as e:
    print(f"[WARN] Dataset indexing failed on startup: {e}")


# ==========================
# FALLBACK MESSAGE
# ==========================
def fallback() -> str:
    return (
        "I couldn't find this information in the latest YantraLive dataset.\n\n"
        f"Please contact human support:\n"
        f"📞 {SUPPORT_PHONE}\n"
        f"📧 {SUPPORT_EMAIL}"
    )


# ==========================
# SB -> VJ normalizer (internal only)
# ==========================
def auto_normalize_sb(text: str) -> str:
    if not text:
        return text

    pattern = re.compile(r"\b(SB)(?:[-\s]*)(\d*)\b", re.IGNORECASE)

    def repl(m: re.Match) -> str:
        digits = m.group(2) or ""
        return "VJ" + digits

    return pattern.sub(repl, text)


# ==========================
# GROQ GENERATION (Llama 3.3) with prefix rule
# ==========================
GROQ_MODEL_ID = "llama-3.3-70b-versatile"


def generate_with_groq(context: str, user_question: str) -> Optional[str]:
    # keep your system prompt exactly as before (omitted here for brevity)
    prefix_rule = (
        "IMPORTANT – ANSWERING GUIDELINES (apply these before any other instruction):\n"
        "- Treat SB-* mentions internally as VJ-* (do this silently). Never mention or explain this mapping to the user.\n"
        "- Do NOT include any provenance or extraction notes (e.g., 'extracted from ...') in the reply; show only the answer.\n"
        "- If the user mentions ONLY a machine model (e.g., 'Hyundai R30') return ALL details present in the CONTEXT for that machine model.\n"
        "  Provide full rows / all dataset columns and keep language natural and helpful.\n"
        "- If the user mentions ONLY a breaker model (e.g., 'VJ20 HD') return ALL details present in the CONTEXT for that breaker model,\n"
        "  EXCLUDING the 'compatible machines' section initially. After listing breaker details, then list compatible machines as a BULLET LIST.\n"
        "- If there are multiple compatible breakers, list them all. Do not add a 'that's all in dataset' line or similar closing text.\n"
        "- Keep responses concise, human-friendly, and start direct answers with a short lead like: 'Here is the price for Hyundai R30' when answering price queries.\n"
        "- Do not reveal internal normalizations or synonyms. If user typed SB*, simply answer referencing VJ* (without explaining the mapping).\n"
        "- Avoid extra filler lines. Answer to the point.\n\n"
    )

    system_prompt = (
        "You are a strict RAG assistant for YantraLive END-CUSTOMER, SPARE_PARTS, "
        "and DEALER rock breaker data.\n"
        "\n"
        "GENERAL RULES:\n"
        "- You MUST use ONLY the facts from the CONTEXT.\n"
        "- If the answer is not clearly present in the CONTEXT, reply EXACTLY: UNSURE_FROM_DATA.\n"
        "- Do NOT guess. Do NOT use outside knowledge.\n"
        "- Keep answers concise, factual, and formatted cleanly.\n"
        "- Respect the dataset tags [END_CUSTOMER], [SPARE_PARTS], [DEALERS] when reasoning.\n"
        "\n"
        "... (rest of original system prompt kept unchanged) ..."
    )

    full_system_prompt = prefix_rule + system_prompt

    user_content = f"""
CONTEXT (rows from YantraLive datasets):
{context}

USER QUESTION:
{user_question}
"""

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL_ID,
            messages=[
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )
        answer = resp.choices[0].message.content
        return answer
    except Exception as e:
        print(f"[GROQ ERROR] {e}")
        return None


# ==========================
# ROUTES: /api/health and /api/chat
# ==========================
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_msg = req.messages[-1].content
    normalized_user_msg = auto_normalize_sb(user_msg)

    # embed query, query chroma, call groq, same as before...
    try:
        query_vec = embed_query(normalized_user_msg)
    except Exception as e:
        print(f"[ERROR] Failed to embed user query with Cohere: {e}")
        return ChatResponse(
            answer=fallback(),
            used_context=[],
            from_fallback=True,
        )

    docs: List[str] = []

    def _query_collection(coll, label: str):
        try:
            result = coll.query(
                query_embeddings=[query_vec],
                n_results=10,
            )
            return result["documents"][0] if result["documents"] else []
        except Exception as e:
            print(f"[ERROR] Failed to query {label} collection: {e}")
            return []

    docs.extend(_query_collection(end_customer_collection, "END_CUSTOMER"))
    docs.extend(_query_collection(spare_parts_collection, "SPARE_PARTS"))
    docs.extend(_query_collection(dealer_collection, "DEALERS"))

    if not docs:
        return ChatResponse(
            answer=fallback(),
            used_context=[],
            from_fallback=True,
        )

    unique_docs = list(dict.fromkeys(docs))
    context = "\n\n---\n\n".join(unique_docs)

    raw = generate_with_groq(context=context, user_question=normalized_user_msg)
    if raw is None:
        return ChatResponse(
            answer=fallback(),
            used_context=unique_docs,
            from_fallback=True,
        )

    raw = raw.strip()
    if "UNSURE_FROM_DATA" in raw:
        return ChatResponse(
            answer=fallback(),
            used_context=unique_docs,
            from_fallback=True,
        )

    # Try to attach brochure view URL if model found
    brochure_url = None
    m = re.search(r"\b(vj)[\s\-]*?(\d{1,3})(?:\s*(hd))?\b", normalized_user_msg, re.IGNORECASE)
    if m:
        parts = [m.group(1) or "", m.group(2) or ""]
        if m.group(3):
            parts.append(m.group(3))
        key_raw = "".join(parts)
        key_norm = re.sub(r"[^a-z0-9]", "", key_raw.lower())
        fname = BROCHURE_MAP.get(key_norm)
        if not fname:
            # try without HD suffix
            alt_key = re.sub(r"hd$", "", key_norm)
            fname = BROCHURE_MAP.get(alt_key)
        if fname:
            base = str(request.base_url).rstrip("/")
            brochure_url = f"{base}/brochures/view/{fname}"

    # Also check raw Groq answer for VJ tokens
    if not brochure_url:
        m2 = re.search(r"\b(vj)[\s\-]*?(\d{1,3})(?:\s*(hd))?\b", raw, re.IGNORECASE)
        if m2:
            key_raw = "".join([m2.group(1) or "", m2.group(2) or ""] + ([m2.group(3)] if m2.group(3) else []))
            key_norm = re.sub(r"[^a-z0-9]", "", key_raw.lower())
            fname = BROCHURE_MAP.get(key_norm)
            if fname:
                base = str(request.base_url).rstrip("/")
                brochure_url = f"{base}/brochures/view/{fname}"

    return ChatResponse(
        answer=raw,
        used_context=unique_docs,
        from_fallback=False,
        brochure_url=brochure_url,
    )
