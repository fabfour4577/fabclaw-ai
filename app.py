from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Message(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "Fabclaw AI server running"}

@app.get("/mcp")
def mcp():
    return {"service": "fabclaw-ai", "status": "active"}

@app.post("/chat")
def chat(msg: Message):
    return {
        "reply": f"You said: {msg.text}"
    }
