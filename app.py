from fastapi import FastAPI
from pydantic import BaseModel
import os
import redis
import json
from openai import OpenAI

app = FastAPI()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Redis connection
redis_url = os.getenv("REDIS_URL")

r = redis.from_url(redis_url, decode_responses=True)

# Request model
class ChatRequest(BaseModel):
    user_id: str
    message: str


@app.get("/")
def root():
    return {"status": "Fabclaw AI running 🚀"}


# Redis test endpoint
@app.get("/redis-test")
def redis_test():
    r.set("test_key", "hello")

    value = r.get("test_key")

    return {"value": value}


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        # Load user history from Redis
        history_json = r.get(request.user_id)

        if history_json:
            history = json.loads(history_json)
        else:
            history = []

        # Add user message
        history.append({
            "role": "user",
            "content": request.message
        })

        # Call OpenAI with conversation history
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are Fabclaw AI, a helpful assistant."
                },
                *history
            ]
        )

        ai_reply = response.choices[0].message.content

        # Save assistant reply
        history.append({
            "role": "assistant",
            "content": ai_reply
        })

        # Save updated conversation back to Redis
        r.set(request.user_id, json.dumps(history))

        return {
            "user_id": request.user_id,
            "response": ai_reply
        }

    except Exception as e:
        return {
            "error": str(e)
        }
