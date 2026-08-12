"""
DAY 3 — A2A discovery + delegation client.

READ FIRST:  ../09-a2a.md
USED IN:     ../10-challenge.md

Usage:
    uv run python src/a2a_client.py http://<peer> "task for their agent"

TODO:
  1. discover(peer_base_url) -> GET {peer}/.well-known/agent-card.json,
     print the card's name + skills, return the card dict.
  2. delegate(card, task) -> POST to card["url"] (NEVER hardcode the
     endpoint — read it from the card; that indirection IS the protocol)
     and extract the output_text from the OpenResponses reply.
  3. __main__ wiring the two together from sys.argv.
"""

# TODO

import json
import sys

import httpx


def discover(peer_base_url: str) -> dict:
    url = peer_base_url.rstrip("/") + "/.well-known/agent-card.json"
    card = httpx.get(url, timeout=10).raise_for_status().json()
    print(f"── discovered: {card['name']} (v{card.get('version', '?')})")
    print(f"   {card.get('description', '')}")
    for skill in card.get("skills", []):
        print(f"   • {skill['name']}: {skill['description']}")
    return card


def delegate(card: dict, task: str) -> str:
    endpoint = card["url"]  # from the card — never hardcoded
    print(f"── delegating to {endpoint} ...")
    resp = httpx.post(endpoint, json={"input": task}, timeout=120).raise_for_status().json()
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    return part["text"]
    raise ValueError(f"no output_text in response: {json.dumps(resp)[:300]}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    peer, task = sys.argv[1], sys.argv[2]
    card = discover(peer)
    answer = delegate(card, task)
    print("\n── their agent replied:\n")
    print(answer)

