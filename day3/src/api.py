"""
DAY 3 — HTTP API.

READ FIRST:  ../03-fastapi-openresponses.md
             ../09-a2a.md   (for the agent card endpoint)

Do not continue to 04-docker.md until:
    curl http://localhost:8000/healthz            -> {"status":"ok"}
    curl -X POST http://localhost:8000/v1/responses \
         -H 'Content-Type: application/json' -d '{"input":"hi"}'
returns an OpenResponses-shaped JSON object.

TODO:
  1. app = FastAPI(...); agent = build_agent()   <- built ONCE, at startup
  2. GET  /healthz
  3. POST /v1/responses  — accept {"input": "...", "model": optional},
     invoke the agent, return:
       {id, object:"response", created_at, status:"completed", model,
        output:[{type:"message", role:"assistant",
                 content:[{type:"output_text", text: ...}]}]}
     (a deliberate SUBSET of OpenResponses — the shape, not the whole spec)
  4. GET /.well-known/agent-card.json — your A2A Agent Card. Use
     STUDENT_NAME and PUBLIC_URL from the environment; the card's "url"
     field must point at YOUR /v1/responses.
"""

# TODO

import os
import time
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

try:
    from .agent import build_agent
except ImportError:
    from agent import build_agent


STUDENT_NAME = os.getenv("STUDENT_NAME", "student")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8000")

app = FastAPI(
    title=f"{STUDENT_NAME} AI Agent",
    description="Day 3 FastAPI service for exposing an AI Agent over HTTP",
)

agent = build_agent()


class AgentRequest(BaseModel):
    input: str
    model: str | None = None


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}


@app.post("/v1/responses")
async def create_response(request: AgentRequest):

    start_time = time.time()

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": request.input,
                }
            ]
        }
    )

    response_text = result["messages"][-1].content

    return {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(start_time),
        "status": "completed",
        "model": request.model or "deep-agent",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": response_text,
                    }
                ],
            }
        ],
    }


@app.get("/.well-known/agent-card.json")
async def agent_card():
    return {
        "protocolVersion": "1.0",
        "name": f"{STUDENT_NAME}-agent",
        "description": (
            "Student AI Agent capable of generating research briefs, "
            "performing calculations, and providing time-related information."
        ),
        "url": f"{PUBLIC_URL}/v1/responses",
        "version": "0.1.0",
        "capabilities": {
            "streaming": False
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": "research-brief",
                "name": "Technical Research Brief",
                "description": "Structured research brief generation.",
                "tags": ["research", "analysis"],
            },
            {
                "id": "calculate",
                "name": "Math Assistant",
                "description": "Performs basic arithmetic calculations.",
                "tags": ["math"],
            },
        ],
    }