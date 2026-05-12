from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# simple memory (temporary, in RAM)
memory = []

class Message(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "Fabclaw AI Agent v1 running"}

@app.get("/mcp")
def mcp():
    return {
        "service": "fabclaw-ai",
        "status": "active",
        "version": "v1"
    }

@app.get("/tools")
def tools():
    return {
        "tools": ["calculator (future)", "chat", "memory"]
    }

@app.post("/chat")
def chat(msg: Message):

    # store user message
    memory.append({"role": "user", "content": msg.text})

    # keep last 6 messages only (cost control)
    recent = memory[-6:]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Fabclaw AI, a helpful assistant with memory."}
        ] + recent
    )

    reply = response.choices[0].message.content

    # store AI response
    memory.append({"role": "assistant", "content": reply})

    return {
        "reply": reply,
        "memory_size": len(memory)
    }

@app.get("/memory")
def get_memory():
    return memory[-10:]
