"""
DAY 4 — CHALLENGE template (heavily scaffolded — read top to bottom).

READ FIRST:  ../04-challenge.md

This file is a fill-in-the-blanks walkthrough. Every ▢ marks a spot
where you write a few lines; everything else is done and commented.
If you completed 00-03, each blank is something you have already
written once today. Total new code: roughly 25 lines.

The shape of what you're building:

    your prompt
        │
        ▼
    Deep Agent ──(tool)──► your authenticated MCP server   [information]
        │
        └──(backend)─────► execute on your machine          [computation]
                                    │
                              LangSmith trace               [visibility]

Run order:
    terminal 1:  uv run python src/secure_mcp.py
    terminal 2:  uv run python src/challenge.py
    browser   :  smith.langchain.com -> project aaasec2-day4 -> your run
"""
import asyncio
import json
import os

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

from deepagents import create_deep_agent
from shell_agent import SYSTEM_PROMPT, llm, make_backend

load_dotenv()

MCP_URL = os.getenv("MCP_URL", "http://localhost:8002/mcp")
ADMIN_TOKEN = os.getenv("MCP_ADMIN_TOKEN", "admin-secret-token")

MY_TOOL_NAME = "get_lab_inventory"


def fetch_my_data() -> str:
    """Fetch my protected lab inventory from my secure MCP server."""

    async def _call() -> str:
        async with Client(MCP_URL, auth=BearerAuth(token=ADMIN_TOKEN)) as c:
            result = await c.call_tool(MY_TOOL_NAME, {})
            return json.dumps(result.data)

    return asyncio.run(_call())


MISSION = (
    "1. Call fetch_my_data to get the inventory. "
    "2. Write a Python program that computes the total value per item "
    "and the grand total in SAR, and flags any item with qty < 5. "
    "3. Execute it with python. "
    "4. Report exactly what the program printed, plus one insight."
)


if __name__ == "__main__":
    backend, cleanup = make_backend()
    try:
        agent = create_deep_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            tools=[fetch_my_data],
            backend=backend,
        )
        result = agent.invoke({"messages": [{"role": "user", "content": MISSION}]})
        print(result["messages"][-1].content)
    finally:
        cleanup()

