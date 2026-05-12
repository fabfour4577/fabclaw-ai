from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

# Initialize OpenAI client (uses OPENAI_API_KEY from environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "Fabclaw AI Agent v1 running"}

@app.post("/chat")
def chat(request: ChatRequest):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Fabclaw AI, a helpful AI assistant."},
            {"role": "user", "content": request.message}
        ]
    )

    return {
        "user_message": request.message,
        "response": response.choices[0].message.content
    }
