from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import redis
import os
import json

# ============================================
# App Setup
# ============================================

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ============================================
# Redis Setup
# ============================================

redis_url = os.getenv("REDIS_URL")

r = redis.from_url(
    redis_url,
    decode_responses=True
)

# ============================================
# Config
# ============================================

MODEL_NAME = "gpt-4o-mini"
MAX_HISTORY = 20
MEMORY_TTL = 604800  # 7 days
SYSTEM_PROMPT = "You are Fabclaw AI, a helpful assistant."

# ============================================
# Request Model
# ============================================

class ChatRequest(BaseModel):
    user_id: str
    message: str

# ============================================
# Helpers
# ============================================

def get_key(user_id: str) -> str:
    return f"chat:{user_id}"

def load_history(user_id: str):
    key = get_key(user_id)
    data = r.get(key)

    if not data:
        return []

    try:
        history = json.loads(data)
        if isinstance(history, list):
            return history
        return []
    except Exception:
        return []

def clean_history(history):
    cleaned = []

    for msg in history:
        if (
            isinstance(msg, dict)
            and msg.get("role") in ["user", "assistant"]
            and isinstance(msg.get("content"), str)
        ):
            cleaned.append(msg)

    return cleaned

def save_history(user_id: str, history):
    r.set(
        get_key(user_id),
        json.dumps(history),
        ex=MEMORY_TTL
    )

# ============================================
# ROUTES
# ============================================

@app.get("/")
def root():
    return {"status": "Fabclaw AI running 🚀"}

# --------------------------------------------
# Redis sanity test
# --------------------------------------------

@app.get("/redis-test")
def redis_test():
    r.set("test_key", "hello")
    return {"value": r.get("test_key")}

# ============================================
# HISTORY (USER-FACING)
# ============================================

@app.get("/history/{user_id}")
def history(user_id: str):
    return {
        "user_id": user_id,
        "history": load_history(user_id)
    }

@app.delete("/history/{user_id}")
def clear_history(user_id: str):
    deleted = r.delete(get_key(user_id))

    return {
        "user_id": user_id,
        "status": "cleared",
        "deleted": deleted == 1
    }

# ============================================
# CHAT ENDPOINT
# ============================================

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        history = load_history(request.user_id)
        history = clean_history(history)

        # Add user message
        history.append({
            "role": "user",
            "content": request.message
        })

        history = history[-MAX_HISTORY:]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history
            ]
        )

        ai_reply = response.choices[0].message.content

        history.append({
            "role": "assistant",
            "content": ai_reply
        })

        history = history[-MAX_HISTORY:]

        save_history(request.user_id, history)

        return {
            "user_id": request.user_id,
            "response": ai_reply
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# 🔥 DEBUG DASHBOARD ENDPOINTS
# ============================================

@app.get("/debug/memory/{user_id}")
def debug_memory(user_id: str):
    """
    Shows raw stored memory for a user (decoded JSON)
    """
    data = r.get(get_key(user_id))

    if not data:
        return {
            "user_id": user_id,
            "key": get_key(user_id),
            "exists": False,
            "data": []
        }

    try:
        return {
            "user_id": user_id,
            "key": get_key(user_id),
            "exists": True,
            "data": json.loads(data)
        }
    except Exception:
        return {
            "user_id": user_id,
            "key": get_key(user_id),
            "exists": True,
            "raw": data,
            "error": "failed_to_parse_json"
        }

@app.get("/debug/keys")
def debug_keys():
    """
    WARNING: dev-only endpoint
    Shows all Redis keys matching chat sessions
    """
    keys = r.keys("chat:*")

    return {
        "count": len(keys),
        "keys": keys
    }

@app.get("/debug/sanity/{user_id}")
def debug_sanity(user_id: str):
    """
    Full system sanity check:
    - Redis state
    - Parsed memory
    - Clean history view
    """
    raw = r.get(get_key(user_id))
    parsed = load_history(user_id)

    return {
        "user_id": user_id,
        "redis_key": get_key(user_id),
        "exists_in_redis": raw is not None,
        "raw": raw,
        "parsed_history": parsed,
        "message_count": len(parsed)
    }
