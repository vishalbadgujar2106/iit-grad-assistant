"""
app.py
FastAPI backend for GradGuide, a graduate student assistant for Illinois Tech.

Run locally:
    uvicorn app:app --reload

Then open http://localhost:8000 in a browser. The chat UI is served from the
same server, so it calls /chat on whatever host it was loaded from.
"""

import logging
import os
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag import answer_question
from ratelimit import SlidingWindowLimiter

log = logging.getLogger("uvicorn.error")

FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "index.html")
MAX_QUESTION_CHARS = 1000
MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_MESSAGE_CHARS = 4000

# Rate limits for /chat. Override with environment variables if needed.
per_client_limiter = SlidingWindowLimiter([
    (int(os.environ.get("RATE_LIMIT_PER_MINUTE", "8")), 60),
    (int(os.environ.get("RATE_LIMIT_PER_DAY", "60")), 24 * 3600),
])
global_limiter = SlidingWindowLimiter([
    (int(os.environ.get("RATE_LIMIT_GLOBAL_PER_HOUR", "300")), 3600),
])
# Only trust X-Forwarded-For behind a proxy that sets it; otherwise anyone
# could fake an IP to dodge the limit. On by default on Vercel (VERCEL=1),
# which overwrites X-Forwarded-For with the real client IP. Without it, every
# visitor would share the proxy's address and one rate-limit bucket.
TRUST_PROXY_HEADERS = os.environ.get(
    "TRUST_PROXY_HEADERS", "1" if os.environ.get("VERCEL") == "1" else ""
).lower() in ("1", "true", "yes")

app = FastAPI(
    title="GradGuide",
    description="RAG-based assistant answering OPT/CPT/registration/financial "
                "aid questions for Illinois Tech international grad students. "
                "General information only, not official immigration or legal advice.",
    version="1.1.0",
)

# The UI is served from this same app, so browsers don't need CORS at all.
# Set ALLOWED_ORIGINS (comma-separated) only if the frontend is hosted on a
# different domain from the backend.
allowed_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_HISTORY_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_CHARS)
    history: list[Message] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND)


@app.get("/health")
def health():
    return {"status": "ok"}


def client_ip(request: Request) -> str:
    if TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request):
    retry_after = per_client_limiter.check(client_ip(request))
    message = ("You're sending questions faster than the assistant allows. "
               "Please wait a bit and try again.")
    if not retry_after:
        retry_after = global_limiter.check("all")
        message = ("The assistant is getting a lot of questions right now. "
                   "Please try again later.")
    if retry_after:
        raise HTTPException(
            status_code=429, detail=message,
            headers={"Retry-After": str(retry_after)},
        )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Please type a question first.")

    enforce_rate_limit(request)

    try:
        result = answer_question(
            req.question.strip(),
            history=[m.model_dump() for m in req.history],
        )
    except anthropic.RateLimitError as e:
        log.warning("Anthropic rate limit: %s", e)
        raise HTTPException(
            status_code=429,
            detail="The assistant is getting a lot of questions right now. "
                   "Please try again in a minute.",
        ) from e
    except (anthropic.APIConnectionError, anthropic.InternalServerError,
            anthropic.OverloadedError) as e:
        log.warning("Anthropic unavailable: %s", e)
        raise HTTPException(
            status_code=503,
            detail="The assistant couldn't reach its language model. Please try again shortly.",
        ) from e
    except (anthropic.APIError, RuntimeError) as e:
        # Bad/missing API key, bad model name, missing knowledge base, etc.
        # These are server misconfigurations; log the detail, don't show it.
        log.exception("Chat request failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="The assistant isn't configured correctly right now. "
                   "Please try again later.",
        ) from e

    return ChatResponse(answer=result["answer"], sources=result["sources"])
