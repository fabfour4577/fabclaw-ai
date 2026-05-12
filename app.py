from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are Fabclaw AI, a helpful assistant."},
                {"role": "user", "content": request.message}
            ]
        )

        return {
            "user_message": request.message,
            "response": response.choices[0].message.content
        }

    except Exception as e:
        print("🔥 ERROR:", str(e))   # <-- THIS will show in Render logs
        return {
            "error": str(e)
        }
