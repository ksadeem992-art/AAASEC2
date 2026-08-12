"""
DAY 4 — The mission: protected MCP data + shell computation + trace.

READ FIRST:  ../03-putting-it-together.md
(requires src/secure_mcp.py running in another terminal, and 00 + 02 done)

The ONLY async code in this file is GIVEN below, fully commented —
you do not write any async today. (New to async? The 60-second
explainer at the top of src/check_auth.py covers everything used here.)

Your TODOs are all synchronous and small:
  1. import build pieces from your shell_agent (llm, SYSTEM_PROMPT,
     make_backend)
  2. under __main__: backend, cleanup = make_backend()
  3. agent = create_deep_agent(model=..., system_prompt=...,
                               tools=[fetch_internal_report],   # <- the given one
                               backend=backend)
  4. invoke MISSION, print the last message, cleanup in finally.
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

# reuse yesterday's lesson: build on the pieces you already have
from shell_agent import SYSTEM_PROMPT, llm, make_backend
from deepagents import create_deep_agent

load_dotenv()

MCP_URL = os.getenv("MCP_URL", "http://localhost:8002/mcp")
TOKEN = os.getenv("MCP_ADMIN_TOKEN", "admin-secret-token")


def fetch_internal_report() -> str:
    """Fetch the protected quarterly report from the secure MCP server.

    (A plain tool that speaks MCP inside — the agent doesn't know or
    care that a network protocol and a bearer token live in here.)
    """
    async def _call():
        async with Client(MCP_URL, auth=BearerAuth(token=TOKEN)) as c:
            result = await c.call_tool("get_internal_report", {})
            return json.dumps(result.data)

    return asyncio.run(_call())


MISSION = (
    "1. Call fetch_internal_report to get the quarterly data. "
    "2. Write analyze.py in your sandbox that computes total revenue, "
    "total costs, and profit margin per month from that data. "
    "3. Execute it. 4. Report the numbers and one sentence of insight."
)


if __name__ == "__main__":
    backend, cleanup = make_backend()
    try:
        agent = create_deep_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            tools=[fetch_internal_report],
            backend=backend,
        )
        result = agent.invoke({"messages": [{"role": "user", "content": MISSION}]})
        print(result["messages"][-1].content)
    finally:
        cleanup()
