from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from openai import OpenAI
import redis
import os
import json
import uuid

# =========================================
# APP SETUP
# =========================================

app = FastAPI()

# ✅ FIX: CORS (THIS WAS YOUR MAIN ISSUE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:8000",
        "*",  # keep for dev only
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# OPENAI
# =========================================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "gpt-4o-mini"
SYSTEM_PROMPT = "You are Fabclaw AI, a helpful assistant."
MAX_HISTORY = 20

# =========================================
# REDIS
# =========================================

redis_url = os.getenv("REDIS_URL")
r = redis.from_url(redis_url, decode_responses=True)

SESSION_PREFIX = "chat"
SESSION_INDEX_PREFIX = "sessions"
MEMORY_TTL = 60 * 60 * 24 * 7  # 7 days

# =========================================
# REQUEST MODEL
# =========================================

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str

# =========================================
# HELPERS
# =========================================

def chat_key(user_id, session_id):
    return f"{SESSION_PREFIX}:{user_id}:{session_id}"

def session_key(user_id):
    return f"{SESSION_INDEX_PREFIX}:{user_id}"

def load_history(user_id, session_id):
    data = r.get(chat_key(user_id, session_id))
    if not data:
        return []
    try:
        return json.loads(data)
    except:
        return []

def save_history(user_id, session_id, history):
    r.set(chat_key(user_id, session_id), json.dumps(history), ex=MEMORY_TTL)

def clean_history(history):
    return [
        m for m in history
        if isinstance(m, dict)
        and m.get("role") in ["user", "assistant"]
        and isinstance(m.get("content"), str)
    ][-MAX_HISTORY:]

def get_sessions(user_id):
    data = r.get(session_key(user_id))
    if not data:
        return []
    try:
        return json.loads(data)
    except:
        return []

def save_sessions(user_id, sessions):
    r.set(session_key(user_id), json.dumps(sessions))

def add_session(user_id, session_id):
    sessions = get_sessions(user_id)
    if session_id not in sessions:
        sessions.append(session_id)
    save_sessions(user_id, sessions)
    return sessions

# =========================================
# ROOT
# =========================================

@app.get("/")
def root():
    return {"status": "Fabclaw AI running 🚀"}

# =========================================
# CHAT
# =========================================

@app.post("/chat")
def chat(request: ChatRequest):

    try:
        history = clean_history(load_history(request.user_id, request.session_id))

        add_session(request.user_id, request.session_id)

        history.append({"role": "user", "content": request.message})

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history]
        )

        reply = response.choices[0].message.content

        history.append({"role": "assistant", "content": reply})

        save_history(request.user_id, request.session_id, history)

        return {
            "reply": reply,
            "session_id": request.session_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================================
# STREAMING CHAT
# =========================================

@app.post("/chat-stream")
def chat_stream(request: ChatRequest):

    def generator():
        try:
            history = clean_history(load_history(request.user_id, request.session_id))
            add_session(request.user_id, request.session_id)

            history.append({"role": "user", "content": request.message})

            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
                stream=True
            )

            full = ""

            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    full += token
                    yield f"data: {token}\n\n"

            history.append({"role": "assistant", "content": full})
            save_history(request.user_id, request.session_id, history)

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")

# =========================================
# HISTORY
# =========================================

@app.get("/history/{user_id}/{session_id}")
def history(user_id: str, session_id: str):
    return load_history(user_id, session_id)

@app.delete("/history/{user_id}/{session_id}")
def clear_history(user_id: str, session_id: str):
    return {"deleted": r.delete(chat_key(user_id, session_id)) == 1}

# =========================================
# SESSIONS
# =========================================

@app.get("/sessions/{user_id}")
def list_sessions(user_id: str):
    return {"sessions": get_sessions(user_id)}

@app.post("/sessions/{user_id}")
def create_session(user_id: str):
    new_id = str(uuid.uuid4())[:8]
    add_session(user_id, new_id)
    return {"session_id": new_id}

# =========================================
# DEBUG
# =========================================

@app.get("/debug/keys")
def debug_keys():
    return {
        "chat_keys": r.keys("chat:*"),
        "session_keys": r.keys("sessions:*")
    }
