from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from io import BytesIO
from PyPDF2 import PdfReader

from datetime import datetime
import redis
import os
import json
import io
import uuid

app = FastAPI(title="Fabclaw AI", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

try:
    r = redis.from_url(redis_url, decode_responses=True)
    r.ping()
    print("Redis connected")
except Exception as e:
    print("Redis failed, using memory fallback:", e)
    r = None

memory_store = {}

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_HISTORY = 20

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    mode: str = "chat"

def build_system_prompt():
    today = datetime.now().strftime("%B %d, %Y")

    return f"""
You are Fabclaw AI, a helpful AI assistant for productivity, coding, research support, business intelligence, and creative problem-solving.

Identity rules:
- Your name is Fabclaw AI.
- Do not claim to be GPT-3.
- Do not guess your exact model identity.
- If asked what model you are, say: "I am Fabclaw AI, powered by OpenAI through this app."

Date and knowledge rules:
- Today's date is {today}.
- You do not currently have live web browsing.
- If the user asks for current events, recent news, live prices, current brand/product availability, or anything after your available knowledge, clearly say that live web search is needed.
- Do not pretend to browse the internet.

Response style:
- Be direct, useful, and accurate.
- Use Markdown when helpful.
- Use code blocks for code.
- If unsure, say so clearly.
""".strip()


def get_history(key):
    if r:
        data = r.get(key)
        return json.loads(data) if data else []
    return memory_store.get(key, [])


def save_history(key, history):
    if r:
        r.set(key, json.dumps(history), ex=60 * 60 * 24 * 7)
    else:
        memory_store[key] = history


def session_key(user_id, session_id):
    return f"chat:{user_id}:{session_id}"


@app.get("/")
def root():
    return {
        "status": "Fabclaw AI running",
        "model": MODEL,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "redis": "connected" if r else "fallback mode",
        "model": MODEL,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        key = session_key(req.user_id, req.session_id)

        history = get_history(key)
        history.append({"role": "user", "content": req.message})
        history = history[-MAX_HISTORY:]

        system_prompt = build_system_prompt()

        if req.mode == "research":
            system_prompt += """

Research Mode Instructions:
- Respond like a professional research analyst.
- Structure answers using:
  Summary
  Key Findings
  Analysis
  Recommendations
- Be detailed and organized.
- Use bullet points when useful.
"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                *history,
            ],
            temperature=0.7,
        )

        reply = response.choices[0].message.content or ""

        history.append({"role": "assistant", "content": reply})
        history = history[-MAX_HISTORY:]

        save_history(key, history)

        return {
            "reply": reply,
            "session_id": req.session_id,
            "model": MODEL,
            "mode": req.mode,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{user_id}")
def create_session(user_id: str):
    return {
        "session_id": str(uuid.uuid4())[:8],
    }


@app.get("/history/{user_id}/{session_id}")
def history(user_id: str, session_id: str):
    key = session_key(user_id, session_id)
    return {
        "history": get_history(key),
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        filename = file.filename or "uploaded-file"
        content_type = file.content_type or "unknown"

        raw = await file.read()
        size = len(raw)

        preview = ""

        if content_type.startswith("text/") or filename.endswith(".txt"):
            preview = raw.decode("utf-8", errors="ignore")[:3000]

        elif content_type == "application/pdf":
            try:
                pdf_reader = PdfReader(io.BytesIO(raw))
                text = ""

                max_pages = min(3, len(pdf_reader.pages))

                for i in range(max_pages):
                    extracted = pdf_reader.pages[i].extract_text()
                    if extracted:
                        text += extracted + "\n"
                clean_text = " ".join(text.split())
                preview = clean_text[:3000] if clean_text else "No text found in the first 3 pages of this PDF."
                
            except Exception as pdf_error:
                preview = f"PDF received, but text extraction failed: {str(pdf_error)}"

        else:
            preview = "File received. Text extraction for this file type will be added next."

        return {
            "filename": filename,
            "content_type": content_type,
            "size": size,
            "preview": preview,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/test-redis")
def test_redis():
    try:
        if r:
            r.set("ping", "pong")
            return {"redis": r.get("ping")}
        return {"redis": "using memory fallback"}
    except Exception as e:
        return {"error": str(e)}
