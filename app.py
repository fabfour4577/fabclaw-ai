from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Fabclaw AI server running"}

@app.get("/mcp")
def mcp():
    return {
        "service": "fabclaw-ai",
        "status": "active"
    }
