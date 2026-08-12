"""
app.py
FastAPI backend for the International Grad Student Assistant.

Run locally:
    uvicorn app:app --reload

Then open frontend/index.html in a browser (it calls http://localhost:8000/chat).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import answer_question

app = FastAPI(
    title="IIT International Grad Student Assistant",
    description="RAG-based assistant answering OPT/CPT/registration/financial "
                "aid questions for Illinois Tech international grad students. "
                "General information only — not official immigration or legal advice.",
    version="1.0.0",
)

# Wide-open CORS for local dev / demo purposes. Tighten this to your actual
# frontend's domain before putting this in front of real users.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        result = answer_question(req.question)
    except RuntimeError as e:
        # Most likely a missing API key — surface a clear error instead of a 500.
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(answer=result["answer"], sources=result["sources"])
