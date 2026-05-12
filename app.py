from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Message(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "Fabclaw AI running"}

@app.get("/mcp")
def mcp():
    return {
        "service": "fabclaw-ai",
        "status": "active",
        "tools": ["chat"]
    }

@app.post("/chat")
def chat(msg: Message):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Fabclaw AI, a helpful assistant."},
            {"role": "user", "content": msg.text}
        ]
    )

    return {
        "reply": response.choices[0].message.content
    }
