from fastapi import FastAPI
from app.models import ChatRequest
from app.chatbot import process_chat

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    return process_chat(request.messages)