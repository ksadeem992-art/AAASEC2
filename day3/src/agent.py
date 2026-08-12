"""
DAY 3 — Agent implementation.

READ FIRST:  ../01-deep-agents.md

Do not continue to api.py until:
    USE_FAKE=1 uv run python src/agent.py
prints a reply, AND (with real keys) the agent answers using its tools.

The contract this file must satisfy — the ONLY thing api.py will rely on:

    def build_agent() -> object with .ainvoke({"messages": [...]})

TODO:

  1. Two boring tools: calculate(expression) and current_time().
     (Boring is the point. Day 3 is about everything AROUND the agent.)
  2. build_agent():
       - if USE_FAKE: return a FakeAgent with the same .ainvoke shape
       - else: create_deep_agent(model=<ChatOpenAI via OpenRouter>,
                                 tools=[...], system_prompt=...,
                                 backend=FilesystemBackend(root_dir=<day3/>,
                                                           virtual_mode=True),
                                 skills=["/skills/"])
  3. A __main__ smoke test that invokes the agent once and prints the reply.

NOTE: default backends give the agent FILESYSTEM tools but NO shell.
An execute tool requires a sandbox backend — that is Day 4, on purpose.
"""

# TODO

import ast
import operator as op
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

USE_FAKE = os.getenv("USE_FAKE", "0") == "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # day3/



_ALLOWED = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod,
}


def _safe_eval(node):
    """Evaluate an arithmetic AST. No names, no calls, no attributes —
    this is NOT a shell and must never become one (that's Day 4)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only basic arithmetic is allowed")


def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4) ** 2'."""
    return _safe_eval(ast.parse(expression, mode="eval").body)


def current_time() -> str:
    """Return the current UTC date and time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


SYSTEM_PROMPT = (
    "You are a helpful research and analysis agent for the AAASEC2 course. "
    "Use your tools when arithmetic or the current time is needed. "
    "When a skill matches the user's request, follow it exactly."
)



class FakeAgent:
    """Deterministic stand-in so the WHOLE pipeline (FastAPI, Docker,
    compose, curl) can be exercised with zero API keys. Same interface
    as the real agent: .ainvoke({'messages': [...]}) -> {'messages': [...]}"""

    class _Msg:
        def __init__(self, content):
            self.content = content

    async def ainvoke(self, payload, config=None):
        user = payload["messages"][-1]
        text = user["content"] if isinstance(user, dict) else user.content
        reply = (
            f"[FAKE AGENT] I received: '{text[:120]}'. "
            f"With real keys I would plan, use tools, and consult my skills. "
            f"calculate('6*7') would give {calculate('6*7')}."
        )
        return {"messages": payload["messages"] + [self._Msg(reply)]}



def build_agent():
    """Return SOMETHING with .ainvoke({'messages': [...]}).

    api.py depends on this signature — never on what's behind it.
    """
    if USE_FAKE:
        return FakeAgent()

    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
    )

    
    backend = FilesystemBackend(root_dir=str(PROJECT_ROOT), virtual_mode=True)

    return create_deep_agent(
        model=llm,
        tools=[calculate, current_time],
        system_prompt=SYSTEM_PROMPT,
        backend=backend,
        skills=["/skills/"],  
    )


if __name__ == "__main__":
    import asyncio

    agent = build_agent()
    result = asyncio.run(
        agent.ainvoke({"messages": [{"role": "user", "content": "What is 17 * 23? And what time is it?"}]})
    )
    print(result["messages"][-1].content)
