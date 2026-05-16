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
# Request Models
# ============================================

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str

# ============================================
# Helpers
# ============================================

def get_key(user_id: str, session_id: str) -> str:
    return f"chat:{user_id}:{session_id}"

def load_history(user_id: str, session_id: str):
    key = get_key(user_id, session_id)

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

def save_history(user_id: str, session_id: str, history):
    key = get_key(user_id, session_id)

    r.set(
        key,
        json.dumps(history),
        ex=MEMORY_TTL
    )

# ============================================
# Root
# ============================================

@app.get("/")
def root():
    return {"status": "Fabclaw AI Session System Running 🚀"}

# ============================================
# Redis Sanity Test
# ============================================

@app.get("/redis-test")
def redis_test():
    r.set("test_key", "hello")

    return {
        "value": r.get("test_key")
    }

# ============================================
# Chat Endpoint
# ============================================

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        # Load history
        history = load_history(
            request.user_id,
            request.session_id
        )

        history = clean_history(history)

        # Add user message
        history.append({
            "role": "user",
            "content": request.message
        })

        # Trim history
        history = history[-MAX_HISTORY:]

        # OpenAI call
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                *history
            ]
        )

        ai_reply = response.choices[0].message.content

        # Add assistant response
        history.append({
            "role": "assistant",
            "content": ai_reply
        })

        # Final trim
        history = history[-MAX_HISTORY:]

        # Save memory
        save_history(
            request.user_id,
            request.session_id,
            history
        )

        return {
            "user_id": request.user_id,
            "session_id": request.session_id,
            "response": ai_reply
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ============================================
# History Endpoints
# ============================================

@app.get("/history/{user_id}/{session_id}")
def history(user_id: str, session_id: str):
    return {
        "user_id": user_id,
        "session_id": session_id,
        "history": load_history(user_id, session_id)
    }

@app.delete("/history/{user_id}/{session_id}")
def clear_history(user_id: str, session_id: str):
    key = get_key(user_id, session_id)

    deleted = r.delete(key)

    return {
        "user_id": user_id,
        "session_id": session_id,
        "status": "cleared",
        "deleted": deleted == 1
    }

# ============================================
# Debug Endpoints
# ============================================

@app.get("/debug/memory/{user_id}/{session_id}")
def debug_memory(user_id: str, session_id: str):
    key = get_key(user_id, session_id)

    data = r.get(key)

    if not data:
        return {
            "user_id": user_id,
            "session_id": session_id,
            "key": key,
            "exists": False,
            "data": []
        }

    try:
        return {
            "user_id": user_id,
            "session_id": session_id,
            "key": key,
            "exists": True,
            "data": json.loads(data)
        }

    except Exception:
        return {
            "user_id": user_id,
            "session_id": session_id,
            "key": key,
            "exists": True,
            "raw": data,
            "error": "failed_to_parse_json"
        }

@app.get("/debug/sanity/{user_id}/{session_id}")
def debug_sanity(user_id: str, session_id: str):
    key = get_key(user_id, session_id)

    raw = r.get(key)

    parsed = load_history(
        user_id,
        session_id
    )

    return {
        "user_id": user_id,
        "session_id": session_id,
        "redis_key": key,
        "exists_in_redis": raw is not None,
        "raw": raw,
        "parsed_history": parsed,
        "message_count": len(parsed)
    }

@app.get("/debug/keys")
def debug_keys():
    keys = r.keys("chat:*")

    return {
        "count": len(keys),
        "keys": keys
    }
