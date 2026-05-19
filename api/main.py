import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import agent as agent_module
from agent import retriever

# ---------- Pydantic models ----------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ---------- App ----------

app = FastAPI(title="SHL Assessment Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    # Warm up: loads the FAISS index, chunks, and embedding model into memory
    # once at startup so the first real request is not slow.
    retriever.retrieve("test", top_k=1)


# ---------- Endpoints ----------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages list must not be empty")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, agent_module.run_agent, messages
            ),
            timeout=25.0,
        )
    except asyncio.TimeoutError:
        return ChatResponse(
            reply="Request timed out. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )
    except Exception:
        return ChatResponse(
            reply="I encountered an error. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )

    recommendations = [
        Recommendation(
            name=r.get("name", ""),
            url=r.get("url", ""),
            test_type=r.get("test_type", ""),
        )
        for r in result.get("recommendations", [])
    ]

    return ChatResponse(
        reply=result.get("reply", ""),
        recommendations=recommendations,
        end_of_conversation=result.get("end_of_conversation", False),
    )
