from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

import redis
import os
import json
import uuid

# ======================================
# APP
# ======================================

app = FastAPI()

# ✅ CORS (fixes frontend + fetch issues)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# OPENAI
# ======================================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ======================================
# REDIS (SAFE INIT)
# ======================================

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

try:
    r = redis.from_url(redis_url, decode_responses=True)
    r.ping()
    print("✅ Redis connected")
except Exception as e:
    print("❌ Redis failed, using in-memory fallback:", e)
    r = None

# fallback memory (if Redis fails)
memory_store = {}

# ======================================
# CONFIG
# ======================================

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are Fabclaw AI, a helpful assistant."
MAX_HISTORY = 20

# ======================================
# REQUEST MODEL
# ======================================

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str

# ======================================
# HELPERS
# ======================================

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

# ======================================
# ROOT
# ======================================

@app.get("/")
def root():
    return {"status": "Fabclaw AI running 🚀"}

# ======================================
# CHAT
# ======================================

@app.post("/chat")
def chat(req: ChatRequest):

    try:
        key = session_key(req.user_id, req.session_id)

        history = get_history(key)

        history.append({"role": "user", "content": req.message})
        history = history[-MAX_HISTORY:]

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history
            ]
        )

        reply = response.choices[0].message.content

        history.append({"role": "assistant", "content": reply})

        save_history(key, history)

        return {
            "reply": reply,
            "session_id": req.session_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================
# SESSIONS
# ======================================

@app.post("/sessions/{user_id}")
def create_session(user_id: str):
    return {
        "session_id": str(uuid.uuid4())[:8]
    }

# ======================================
# HISTORY
# ======================================

@app.get("/history/{user_id}/{session_id}")
def history(user_id: str, session_id: str):
    key = session_key(user_id, session_id)
    return {"history": get_history(key)}

# ======================================
# DEBUG
# ======================================

@app.get("/test-redis")
def test_redis():
    try:
        if r:
            r.set("ping", "pong")
            return {"redis": r.get("ping")}
        else:
            return {"redis": "using memory fallback"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "redis": "connected" if r else "fallback mode"
    }
