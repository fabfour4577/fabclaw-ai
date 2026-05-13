from fastapi import FastAPI
from pydantic import BaseModel
from collections import defaultdict
import os
from openai import OpenAI

app = FastAPI()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simple in-memory memory store
memory = defaultdict(list)

# Request model
class ChatRequest(BaseModel):
    user_id: str
    message: str


@app.get("/")
def root():
    return {"status": "Fabclaw AI running 🚀"}


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        # Get user history
        history = memory[request.user_id]

        # Add user message
        history.append({"role": "user", "content": request.message})

        # Call OpenAI with conversation history
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are Fabclaw AI, a helpful assistant."},
                *history
            ]
        )

        ai_reply = response.choices[0].message.content

        # Save assistant reply
        history.append({"role": "assistant", "content": ai_reply})

        return {
            "user_id": request.user_id,
            "response": ai_reply
        }

    except Exception as e:
        return {
            "error": str(e)
        }
