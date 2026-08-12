"""
DAY 4 — Deep Agent with shell access.

READ FIRST:  ../00-deep-agent-shell.md

TODO:
  1. llm — same OpenRouter ChatOpenAI as every day.
  2. make_backend() -> (backend, cleanup_fn). Default "local":
       LocalShellBackend(root_dir=<day4/work/>, virtual_mode=True,
                         env={"PATH": os.environ["PATH"]})
       cleanup = nothing (it's your host)
     EXTRA (05-extra-sandbox.md): "daytona" -> Daytona().create()
       + DaytonaSandbox, cleanup = sandbox.stop
  3. __main__: create backend, create_deep_agent(model, system_prompt,
     backend=backend), invoke the calculator task, print the reply,
     cleanup in a finally block.
"""

# TODO

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from deepagents import create_deep_agent

load_dotenv()

PROVIDER = os.getenv("SANDBOX_PROVIDER", "local")

WORK_DIR = Path(__file__).resolve().parent.parent / "work"

llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
)

SYSTEM_PROMPT = (
    "You are a Python coding assistant with shell access. "
    "Write files with your filesystem tools and run them with execute. "
    "If a command fails, read the error and fix your code."
)


def make_backend():
    """Return (backend, cleanup_fn). Default runs on THIS machine."""
    if PROVIDER == "local":
        from deepagents.backends import LocalShellBackend

        WORK_DIR.mkdir(exist_ok=True)
        backend = LocalShellBackend(
            root_dir=str(WORK_DIR),
            virtual_mode=True,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )
        return backend, (lambda: None)  # nothing to destroy — it's your host

    # ---- EXTRA: real sandboxes (see 05-extra-sandbox.md) ----
    if PROVIDER == "daytona":
        from daytona import Daytona
        from langchain_daytona import DaytonaSandbox

        sandbox = Daytona().create()
        return DaytonaSandbox(sandbox=sandbox), sandbox.stop

    if PROVIDER == "langsmith":
        from deepagents.backends import LangSmithSandbox
        from langsmith.sandbox import SandboxClient

        client = SandboxClient()
        sb = client.create_sandbox()
        return LangSmithSandbox(sandbox=sb), lambda: client.delete_sandbox(sb.name)

    raise ValueError(f"unknown SANDBOX_PROVIDER: {PROVIDER}")


TASK = (
    "1. Create calculator.py with add/sub/mul/div functions (div raises on zero). "
    "2. Write test_calculator.py with pytest tests, including the zero case. "
    "3. Run the tests with execute (use 'python -m pytest'; pip install pytest "
    "first if it's missing). "
    "4. If anything fails, fix it and re-run until green. "
    "5. Report the final pytest output."
)

if __name__ == "__main__":
    print("Starting...")

    backend, cleanup = make_backend()
    print("Backend ready")

    try:
        agent = create_deep_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            backend=backend,
        )

        print("Agent created")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": TASK}]}
        )

        print("Finished")
        print(result)

    finally:
        cleanup()
