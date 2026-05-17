from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

import redis
import os
import json
import uuid

# ============================================
# APP SETUP
# ============================================

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ============================================
# REDIS SETUP
# ============================================

redis_url = os.getenv("REDIS_URL")

r = redis.from_url(
    redis_url,
    decode_responses=True
)

# ============================================
# CONFIG
# ============================================

MODEL_NAME = "gpt-4o-mini"
MAX_HISTORY = 20
MEMORY_TTL = 60 * 60 * 24 * 7  # 7 days

SYSTEM_PROMPT = "You are Fabclaw AI, a helpful assistant."

SESSION_PREFIX = "chat"
SESSION_INDEX_PREFIX = "sessions"

# ============================================
# REQUEST MODEL
# ============================================

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str

# ============================================
# HELPERS - MEMORY
# ============================================

def chat_key(user_id: str, session_id: str):
    return f"{SESSION_PREFIX}:{user_id}:{session_id}"


def load_history(user_id: str, session_id: str):
    key = chat_key(user_id, session_id)

    data = r.get(key)

    if not data:
        return []

    try:
        history = json.loads(data)

        if isinstance(history, list):
            return history

        return []

    except:
        return []


def save_history(user_id: str, session_id: str, history):
    key = chat_key(user_id, session_id)

    r.set(
        key,
        json.dumps(history),
        ex=MEMORY_TTL
    )


def clean_history(history):
    cleaned = []

    for msg in history:
        if (
            isinstance(msg, dict)
            and msg.get("role") in ["user", "assistant"]
            and isinstance(msg.get("content"), str)
        ):
            cleaned.append(msg)

    return cleaned[-MAX_HISTORY:]

# ============================================
# SESSION REGISTRY
# ============================================

def session_index_key(user_id: str):
    return f"{SESSION_INDEX_PREFIX}:{user_id}"


def get_sessions(user_id: str):
    data = r.get(session_index_key(user_id))

    if not data:
        return []

    try:
        sessions = json.loads(data)

        return sessions if isinstance(sessions, list) else []

    except:
        return []


def save_sessions(user_id: str, sessions):
    r.set(
        session_index_key(user_id),
        json.dumps(sessions)
    )


def add_session(user_id: str, session_id: str):
    sessions = get_sessions(user_id)

    if session_id not in sessions:
        sessions.append(session_id)

    save_sessions(user_id, sessions)

    return sessions

# ============================================
# ROOT
# ============================================

@app.get("/")
def root():
    return {
        "status": "Fabclaw AI session system running 🚀"
    }

# ============================================
# REDIS TEST
# ============================================

@app.get("/redis-test")
def redis_test():
    r.set("test_key", "hello")

    return {
        "value": r.get("test_key")
    }

# ============================================
# NORMAL CHAT ENDPOINT
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

        # Ensure session exists
        sessions = get_sessions(request.user_id)

        if request.session_id not in sessions:
            add_session(
                request.user_id,
                request.session_id
            )

        print(
            f"[CHAT] "
            f"user={request.user_id} "
            f"session={request.session_id}"
        )

        # Add user message
        history.append({
            "role": "user",
            "content": request.message
        })

        history = history[-MAX_HISTORY:]

        # OpenAI request
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

        # Save assistant response
        history.append({
            "role": "assistant",
            "content": ai_reply
        })

        history = history[-MAX_HISTORY:]

        save_history(
            request.user_id,
            request.session_id,
            history
        )

        return {
            "user_id": request.user_id,
            "session_id": request.session_id,
            "reply": ai_reply
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ============================================
# STREAMING CHAT ENDPOINT
# ============================================

@app.post("/chat-stream")
def chat_stream(request: ChatRequest):

    async def event_generator():

        try:
            # Load history
            history = load_history(
                request.user_id,
                request.session_id
            )

            history = clean_history(history)

            # Ensure session exists
            add_session(
                request.user_id,
                request.session_id
            )

            # Add user message
            history.append({
                "role": "user",
                "content": request.message
            })

            history = history[-MAX_HISTORY:]

            # Streaming OpenAI request
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    *history
                ],
                stream=True
            )

            full_reply = ""

            for chunk in response:

                delta = chunk.choices[0].delta

                if delta and delta.content:

                    token = delta.content

                    full_reply += token

                    yield f"data: {token}\n\n"

            # Save assistant response
            history.append({
                "role": "assistant",
                "content": full_reply
            })

            history = history[-MAX_HISTORY:]

            save_history(
                request.user_id,
                request.session_id,
                history
            )

            yield "data: [DONE]\n\n"

        except Exception as e:

            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# ============================================
# HISTORY
# ============================================

@app.get("/history/{user_id}/{session_id}")
def history(user_id: str, session_id: str):

    return {
        "user_id": user_id,
        "session_id": session_id,
        "history": load_history(
            user_id,
            session_id
        )
    }


@app.delete("/history/{user_id}/{session_id}")
def clear_history(user_id: str, session_id: str):

    deleted = r.delete(
        chat_key(user_id, session_id)
    )

    return {
        "user_id": user_id,
        "session_id": session_id,
        "deleted": deleted == 1
    }

# ============================================
# SESSIONS
# ============================================

@app.get("/sessions/{user_id}")
def list_sessions(user_id: str):

    return {
        "user_id": user_id,
        "sessions": get_sessions(user_id)
    }


@app.post("/sessions/{user_id}")
def create_session(user_id: str):

    new_session = str(uuid.uuid4())[:8]

    sessions = add_session(
        user_id,
        new_session
    )

    return {
        "user_id": user_id,
        "session_id": new_session,
        "sessions": sessions
    }

# ============================================
# DEBUG
# ============================================

@app.get("/debug/memory/{user_id}/{session_id}")
def debug_memory(user_id: str, session_id: str):

    return {
        "exists": r.get(
            chat_key(user_id, session_id)
        ) is not None,

        "history": load_history(
            user_id,
            session_id
        )
    }


@app.get("/debug/sessions/{user_id}")
def debug_sessions(user_id: str):

    return {
        "user_id": user_id,
        "sessions": get_sessions(user_id)
    }


@app.get("/debug/keys")
def debug_keys():

    return {
        "chat_keys": r.keys("chat:*"),
        "session_keys": r.keys("sessions:*")
    }
