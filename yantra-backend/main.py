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
from openai import OpenAI

# ==========================
# ENVIRONMENT
# ==========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPPORT_PHONE = os.getenv("YANTRALIVE_SUPPORT_PHONE", "+91-9876543210")
SUPPORT_EMAIL = os.getenv("YANTRALIVE_SUPPORT_EMAIL", "support@yantralive.com")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

# create OpenAI client (modern 1.0+ interface)
client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================
# FASTAPI
# ==========================
app = FastAPI(
    title="YantraLive RAG Chatbot (OpenAI + Chroma)",
    version="1.4",
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
    brochure_urls: Optional[List[str]] = None  # changed to list


# ==========================
# OpenAI EMBEDDINGS + RETRY (modern client)
# ==========================
EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 64
EMBED_BATCH_SLEEP_SECONDS = 1.0


def _openai_embed_with_retry(
    texts: List[str],
    max_retries: int = 5,
    label: str = "",
) -> List[List[float]]:
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            # resp.data is a list of items with .embedding
            return [item.embedding for item in resp.data]
        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg:
                wait = 5 * (attempt + 1)
                print(
                    f"[OPENAI] Rate limited while embedding {label} "
                    f"(attempt {attempt + 1}/{max_retries}). Sleeping {wait}s."
                )
                time.sleep(wait)
                continue

            print(f"[OPENAI] Non-rate-limit error while embedding {label}: {e}")
            raise

    raise RuntimeError(f"OpenAI embed retries exceeded for {label}")


def embed_documents(texts: List[str]) -> List[List[float]]:
    return _openai_embed_with_retry(
        texts=texts,
        label=f"documents batch (size={len(texts)})",
    )


def embed_query(text: str) -> List[float]:
    embeddings = _openai_embed_with_retry(
        texts=[text],
        label="user query",
    )
    return embeddings[0]


# ==========================
# LOAD CSV + INDEX (OpenAI -> Chroma)
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
            f"[INDEX] Finished indexing {total_docs} rows for {tag} using OpenAI embeddings."
        )
    except Exception as e:
        print(f"[WARN] Failed to embed/index dataset {tag} with OpenAI: {e}")
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
# OpenAI GENERATION (chat) with prefix rule (modern client)
# ==========================
GROQ_MODEL_ID = "llama-3.3-70b-versatile"  # kept for reference if needed
CHAT_MODEL_ID = "gpt-3.5-turbo-16k"


def generate_with_openai(context: str, user_question: str) -> Optional[str]:
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
        # NEW RULES ADDED PER REQUEST:
        "- If the user asks ONLY for a BROCHURE (for example: 'give me brochure for vj30' or 'brochure vj30'), keep the assistant reply minimal (a single-line acknowledgement\n"
        "  referencing the model is acceptable). Do NOT attempt to embed, preview, or describe the PDF. The backend will attach brochure_urls when available so the frontend\n"
        "  can provide buttons that open the brochures in new tabs.\n\n"
    )

    system_prompt = (
        "You are a strict RAG assistant for YantraLive END-CUSTOMER, SPARE_PARTS, "
        "and DEALER rock breaker data.\n"
        "\n"
        "GENERAL RULES:\n"
        "- You MUST use ONLY the facts from the CONTEXT.\n"
        "- If the answer is not clearly present in the CONTEXT, reply EXACTLY: UNSURE_FROM_DATA.\n"
        "- Do NOT guess. Do NOT use outside knowledge.\n        "
        "- Answer clearly, professionally, and in a structured way for the end customer.\n"
        "- Respect the dataset tags [END_CUSTOMER], [SPARE_PARTS], [DEALERS] when reasoning.\n"
        "\n"
        "FOR COMPATIBILITY QUESTIONS (e.g., 'which breaker is compatible with SANY SY20', "
        "'which all breakers work with Hyundai R30', 'show options for X machine'):\n"
        "- Scan ALL lines in the CONTEXT.\n"
        "- Identify EVERY row where the machine brand and/or machine model match the user question.\n"
        "- Collect all DISTINCT compatible breaker models / SKUs from those rows.\n"
        "- Return them as a BULLET LIST, one breaker per line.\n"
        "- For each breaker, include key details if present: breaker model name, SKU, chisel diameter, "
        "impact energy, and any important notes (like price or stock).\n"
        "- Do NOT arbitrarily pick only one breaker if multiple are present; always show all relevant options.\n"
        "\n"
        "FOR COMPARISON QUESTIONS (e.g., 'Compare JCB and CAT', 'Compare breaker A vs breaker B'):\n"
        "- Start your answer with: 'Sure, here is a comparison between ... and ...:'\n"
        "- Build a clean Markdown table using this format:\n"
        "  | Feature | Option 1 | Option 2 |\n"
        "  |--------|----------|----------|\n"
        "  | ...    | ...      | ...      |\n"
        "- Choose meaningful features from the CONTEXT such as machine model, breaker model, "
        "impact energy, chisel diameter, price, stock, etc.\n"
        "- Only include facts that are clearly present in the CONTEXT.\n"
        "\n"
        "FOR SPECIFIC PARAMETER QUESTIONS (e.g., 'What is the chisel diameter for Hyundai R30?', "
        "'What is the impact energy in joules for model VJ20 HD?'):\n"
        "- Find the row(s) that match the machine/breaker mentioned.\n"
        "- Extract the exact requested numeric or textual value from those row(s).\n"
        "- If multiple rows give different values, mention each distinct value clearly.\n"
        "\n"
        "FOR SUBJECTIVE/BEST-OPTION QUESTIONS (e.g., 'Which is the best model for X?', "
        "'Which breaker is more suitable for Y machine?'):\n"
        "- First, list all relevant options as a bullet list with their key specs.\n"
        "- Then, based ONLY on the CONTEXT (features such as impact energy, recommended tonnage, "
        "application type, or any hints in the data), choose ONE option as the best.\n"
        "- Clearly say: 'According to the available data, the best option is <NAME> because ...'\n"
        "- Do NOT invent reasons that are not supported by the CONTEXT.\n"
        "\n"
        "REMEMBER:\n"
        "- Never invent breakers, dealers, or spare parts that are not present in the CONTEXT.\n"
        "- If the machine, breaker, part, or dealer mentioned is not present at all, reply UNSURE_FROM_DATA.\n"
    )

    full_system_prompt = prefix_rule + system_prompt

    user_content = f"""
CONTEXT (rows from YantraLive datasets):
{context}

USER QUESTION:
{user_question}
"""

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL_ID,
            messages=[
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=2000,
        )
        answer = resp.choices[0].message.content
        return answer
    except Exception as e:
        print(f"[OPENAI ERROR] {e}")
        return None


# ==========================
# ROUTES: /api/health and /api/chat
# ==========================
@app.get("/api/health")
def health():
    return {"status": "ok"}


def _extract_vj_tokens(text: str) -> List[str]:
    """
    Find all VJ tokens like VJ20, VJ30, VJ43HD etc.
    Returns normalized keys e.g. vj20, vj43hd (lowercase, alphanumeric only)
    """
    tokens = []
    if not text:
        return tokens
    for m in re.finditer(r"\b(vj)[\s\-]*?(\d{1,3})(?:\s*(hd))?\b", text, re.IGNORECASE):
        parts = [m.group(1) or "", m.group(2) or ""]
        if m.group(3):
            parts.append(m.group(3))
        key_raw = "".join(parts)
        key_norm = re.sub(r"[^a-z0-9]", "", key_raw.lower())
        tokens.append(key_norm)
    return tokens


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_msg = req.messages[-1].content
    normalized_user_msg = auto_normalize_sb(user_msg)

    # embed query, query chroma, call openai chat, same as before...
    try:
        query_vec = embed_query(normalized_user_msg)
    except Exception as e:
        print(f"[ERROR] Failed to embed user query with OpenAI: {e}")
        return ChatResponse(
            answer=fallback(),
            used_context=[],
            from_fallback=True,
            brochure_urls=None,
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
            brochure_urls=None,
        )

    unique_docs = list(dict.fromkeys(docs))
    context = "\n\n---\n\n".join(unique_docs)

    raw = generate_with_openai(context=context, user_question=normalized_user_msg)
    if raw is None:
        return ChatResponse(
            answer=fallback(),
            used_context=unique_docs,
            from_fallback=True,
            brochure_urls=None,
        )

    raw = raw.strip()
    if "UNSURE_FROM_DATA" in raw:
        return ChatResponse(
            answer=fallback(),
            used_context=unique_docs,
            from_fallback=True,
            brochure_urls=None,
        )

    # Collect all VJ tokens found in user message AND in the model's raw output
    found_keys = set()
    user_tokens = _extract_vj_tokens(normalized_user_msg)
    for k in user_tokens:
        found_keys.add(k)

    answer_tokens = _extract_vj_tokens(raw)
    for k in answer_tokens:
        found_keys.add(k)

    brochure_urls: List[str] = []
    base = str(request.base_url).rstrip("/")

    for key in found_keys:
        fname = BROCHURE_MAP.get(key)
        if not fname:
            # try removing hd suffix if present in key
            alt = re.sub(r"hd$", "", key)
            fname = BROCHURE_MAP.get(alt)
        if fname:
            brochure_urls.append(f"{base}/brochures/view/{fname}")

    # dedupe while preserving order
    brochure_urls = list(dict.fromkeys(brochure_urls))

    return ChatResponse(
        answer=raw,
        used_context=unique_docs,
        from_fallback=False,
        brochure_urls=brochure_urls if brochure_urls else None,
    )
