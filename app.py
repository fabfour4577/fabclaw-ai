from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "Fabclaw AI Agent v1 running"}

@app.post("/chat")
def chat(request: ChatRequest):
    return {
        "user_message": request.message,
        "response": f"You said: {request.message}"
    }
