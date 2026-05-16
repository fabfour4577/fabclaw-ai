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

        if not isinstance(history, list):
            return []

        return history

    except Exception:
        return []

def clean_history(history):
    cleaned = []

    for msg in history:
        if (
            isinstance(msg, dict)
            and "role" in msg
            and "content" in msg
            and msg["role"] in ["user", "assistant"]
        ):
            cleaned.append(msg)

    return cleaned

def save_history(user_id: str, history):
    key = get_key(user_id)

    r.set(
        key,
        json.dumps(history),
        ex=MEMORY_TTL
    )

# ============================================
# Routes
# ============================================

@app.get("/")
def root():
    return {"status": "Fabclaw AI running 🚀"}

@app.get("/redis-test")
def redis_test():
    r.set("test_key", "hello")
    return {"value": r.get("test_key")}

@app.get("/history/{user_id}")
def history(user_id: str):
    return {
        "user_id": user_id,
        "history": load_history(user_id)
    }

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        # Load + clean history
        history = load_history(request.user_id)
        history = clean_history(history)

        # Add user message
        history.append({
            "role": "user",
            "content": request.message
        })

        # Trim history (prevents growth)
        history = history[-MAX_HISTORY:]

        # Call OpenAI
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history
            ]
        )

        ai_reply = response.choices[0].message.content

        # Add assistant reply
        history.append({
            "role": "assistant",
            "content": ai_reply
        })

        # Final trim before saving
        history = history[-MAX_HISTORY:]

        # Save to Redis
        save_history(request.user_id, history)

        return {
            "user_id": request.user_id,
            "response": ai_reply
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
