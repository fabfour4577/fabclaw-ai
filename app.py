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
# Constants
# ============================================

MODEL_NAME = "gpt-4o-mini"

MAX_HISTORY = 20

MEMORY_TTL = 604800  # 7 days

SYSTEM_PROMPT = (
    "You are Fabclaw AI, a helpful assistant."
)

# ============================================
# Request Models
# ============================================

class ChatRequest(BaseModel):
    user_id: str
    message: str

# ============================================
# Helpers
# ============================================

def get_user_key(user_id: str) -> str:
    return f"chat:{user_id}"

def load_history(user_id: str):
    key = get_user_key(user_id)

    history_json = r.get(key)

    if not history_json:
        return []

    try:
        history = json.loads(history_json)

        if not isinstance(history, list):
            return []

        return history

    except Exception:
        return []

def save_history(user_id: str, history):
    key = get_user_key(user_id)

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
    return {
        "status": "Fabclaw AI running 🚀"
    }

@app.get("/redis-test")
def redis_test():
    r.set("test_key", "hello")

    value = r.get("test_key")

    return {
        "value": value
    }

@app.get("/history/{user_id}")
def get_history(user_id: str):
    history = load_history(user_id)

    return {
        "user_id": user_id,
        "history": history
    }

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        # ====================================
        # Load Existing History
        # ====================================

        history = load_history(request.user_id)

        # ====================================
        # Add User Message
        # ====================================

        history.append({
            "role": "user",
            "content": request.message
        })

        # ====================================
        # Prevent Infinite Growth
        # ====================================

        history = history[-MAX_HISTORY:]

        # ====================================
        # OpenAI Request
        # ====================================

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

        # ====================================
        # Save Assistant Reply
        # ====================================

        history.append({
            "role": "assistant",
            "content": ai_reply
        })

        # ====================================
        # Save Back To Redis
        # ====================================

        save_history(
            request.user_id,
            history
        )

        # ====================================
        # Response
        # ====================================

        return {
            "user_id": request.user_id,
            "response": ai_reply
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
