# ============================================================
# DAY 2 LAB — SKELETON: Build a Multi-Agent Research Team
# ============================================================
# Fill in every TODO. Don't open the solution (day2_lab_solution.py)
# until you pass the self-check at the bottom.
#
# WHAT CHANGES FROM DAY 1 — read this table twice:
#
#   Day 1 (single agent)              Day 2 (multi-agent)
#   ─────────────────────             ─────────────────────────────
#   nodes = Python functions          nodes = LLM agents w/ personas
#   routing = your if/else            routing = supervisor LLM decides
#   one prompt for everything         one system prompt PER agent
#   tools available everywhere        tools SCOPED (only researcher
#                                       can search the web)
#   loop = quality-score retry        loop = critic sends draft back
#                                       to writer for revision
#
# What does NOT change: State + Nodes + Edges. A multi-agent system
# is STILL just a StateGraph. If you can build Day 1, you can build
# this — the new ideas are personas, the supervisor, and guardrails.
#
# The system you're building (the SUPERVISOR pattern):
#
#              ┌──────────── supervisor ─────────────┐
#              │       (LLM decides who's next)      │
#     ┌────────┼───────────┬───────────┬─────────────┤
#     ↓        ↓           ↓           ↓             ↓
#  researcher  analyst    writer     critic       FINISH
#     │        │           │           │             ↓
#     └────────┴───────────┴───────────┘            END
#          (every worker reports back to the supervisor)
#
# Recommended reading BEFORE you start (~25 min):
#   1. Multi-agent concepts (architectures, supervisor pattern):
#      https://docs.langchain.com/oss/python/langgraph/multi-agent
#   2. Refresh: conditional branching + loops (you need both again):
#      https://docs.langchain.com/oss/python/langgraph/use-graph-api#conditional-branching
#   3. Structured output (the supervisor's decision is structured!):
#      https://docs.langchain.com/oss/python/langchain/structured-output
#
# Setup: same as Day 1 — `uv sync`, keys in .env, or USE_FAKE=1.
# ============================================================

import os
import operator
from datetime import datetime
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage

# TODO STEP 0 — same imports as Day 1:
# StateGraph, START, END from langgraph.graph
# InMemorySaver from langgraph.checkpoint.memory
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

MAX_REVISIONS = 2     
MAX_TURNS = 12        


# ============================================================
# STEP 1 — SHARED STATE: the team's "blackboard"
# ============================================================
# Day 1's state was a data PIPELINE (each field filled once, in
# order). Day 2's state is a BLACKBOARD: every agent reads all of
# it and writes only its own section; the supervisor reads it to
# decide who goes next.
#
# Define a TypedDict with:
#   task (str)
#   research_notes  <- List[str], APPEND-ONLY (which reducer? Day 1!)
#   analysis (str), draft (str), critique (str)
#   revision_count (int), turn_count (int)
#   next_agent (str)   <- the supervisor writes its decision HERE
#   execution_logs     <- append-only, same as Day 1
#
# ASK YOURSELF: why must research_notes append but draft overwrite?
# What would happen to the revision loop if draft used operator.add?

class TeamState(TypedDict):
    task: str
    # TODO: add the remaining 8 keys (two use Annotated + operator.add)
    research_notes: Annotated[List[str], operator.add]
    analysis: str
    draft: str
    critique: str
    revision_count: int
    turn_count: int
    next_agent: str
    execution_logs: Annotated[List[str], operator.add]



# ============================================================
# STEP 2 — STRUCTURED ROUTING DECISION
# ============================================================
# Day 1: structured output produced a quality SCORE.
# Day 2: structured output produces a ROUTING DECISION — this is
# the trick that turns an LLM into a supervisor. Literal[...] means
# the model CANNOT invent an agent that doesn't exist.
#
# WHERE TO LOOK: structured-output docs (same page as Day 1).

class RouterDecision(BaseModel):
    """The supervisor's choice of who acts next."""
    next_agent: Literal["researcher", "analyst", "writer", "critic", "FINISH"]
    reason: str = Field(description="One sentence explaining the choice")


# ============================================================
# STEP 3 — ONE LLM, FOUR PERSONAS (+ tools scoped per agent)
# ============================================================
# A multi-agent "team" doesn't need four models — it needs four
# SYSTEM PROMPTS. (In production you might also vary the model per
# agent: cheap model for the critic, big one for the writer.)
#
# TODO:
from langchain.chat_models import ChatOpenAI


from tavily_python import TavilySearch
# 1. Write a PERSONAS dict: role -> system prompt, for
#    "researcher", "analyst", "writer", "critic".
#    Each persona must say what the agent DOES and what it MUST NOT
#    do (e.g. the researcher never analyzes). Boundaries between
#    agents live in the prompts — write them sharp.
# 2. Create llm (ChatOpenAI + OpenRouter, exactly like Day 1) and
#    search_tool (TavilySearch(max_results=4)).
# 3. supervisor_llm = llm.with_structured_output(RouterDecision)
# 4. Helper: run_persona(role, user_content) → invoke llm with
#    [SystemMessage(PERSONAS[role]), HumanMessage(user_content)]
#    and return response.content.
#
# TOOL SCOPING: only the researcher node may call search_tool.
# That's a deliberate design decision, not a limitation — ask
# yourself what could go wrong if the critic could search.

PERSONAS = {
    # TODO: four personas
      "researcher": SystemMessage(content="""
    You are the Researcher. Your job is to search and collect factual, sourceable notes
    relevant to the team's task. Append short findings to state.research_notes.
    Do NOT analyze, draft, or critique. Always include sources when available.
    Be concise and factual.
    """.strip()),

        "analyst": SystemMessage(content="""
    You are the Analyst. Read state.research_notes and produce a short analysis summary
    in state.analysis. Highlight relevance, contradictions, and gaps.
    Do NOT perform web searches or write the final draft.
    """.strip()),

        "writer": SystemMessage(content="""
    You are the Writer. Using state.research_notes and state.analysis, produce a full
    draft in state.draft. Overwrite state.draft on each revision. Do NOT append to
    research_notes or perform searches. Cite key points from research_notes.
    """.strip()),

        "critic": SystemMessage(content="""
    You are the Critic. Evaluate state.draft and write a short critique in state.critique.
    If the draft needs revision, the supervisor should send control back to 'writer'.
    If acceptable, the supervisor should choose 'FINISH'. Do NOT search the web or
    write the final draft yourself.
    """.strip()),
}

# TODO: llm, search_tool, supervisor_llm, run_persona


llm = ChatOpenAI(temperature=0)
search_tool = TavilySearch(max_results=4)


supervisor_llm = llm.with_structured_output(RouterDecision)


def run_persona(role: str, user_content: str) -> str:
    messages = [PERSONAS[role], HumanMessage(content=user_content)]
    resp = llm(messages=messages)
    
    return getattr(resp, "content", None) or resp.generations[0][0].message.content


# ============================================================
# STEP 4 — THE SUPERVISOR NODE (the piece Day 1 didn't have)
# ============================================================
# The supervisor node must:
# 1. Increment turn_count.
# 2. Build a STATUS SUMMARY of the blackboard (which sections are
#    filled? what does the critique say? how many revisions?).
#    Don't dump the full text of everything — the supervisor needs
#    STATUS, not content. (Why? Think tokens and attention.)
# 3. Ask supervisor_llm for a RouterDecision.
# 4. GUARDRAILS — never trust an LLM to terminate a loop:
#      a) if turn_count > MAX_TURNS → force FINISH
#      b) if the LLM picks writer/critic but revision_count >=
#         MAX_REVISIONS and a draft exists → force FINISH
#    This is Day 1's iteration cap wearing a new hat. Same lesson:
#    the LLM proposes, YOUR CODE disposes.
# 5. Return {"next_agent": ..., "turn_count": ..., "execution_logs": [...]}
#
# WHERE TO LOOK: multi-agent docs → "Supervisor" section.

def supervisor_node(state: TeamState):
    # TODO
   
    state["turn_count"] = state.get("turn_count", 0) + 1

    
    status_parts = []
    status_parts.append(f"turn_count={state['turn_count']}")
    status_parts.append(f"revision_count={state.get('revision_count', 0)}")
    status_parts.append(f"research_notes_count={len(state.get('research_notes', []))}")
    status_parts.append(f"has_analysis={'yes' if state.get('analysis') else 'no'}")
    status_parts.append(f"has_draft={'yes' if state.get('draft') else 'no'}")
    status_parts.append(f"has_critique={'yes' if state.get('critique') else 'no'}")

    status_summary = " | ".join(status_parts)

    
    human_text = (
        f"TeamState status summary: {status_summary}\n"
        f"Task: {state.get('task','(no task)')}\n"
        "Decide which agent should act next. Return JSON matching RouterDecision."
    )

    
    try:
        resp = supervisor_llm(messages=[SUPERVISOR_SYSTEM, HumanMessage(content=human_text)])
      
        decision = resp if isinstance(resp, RouterDecision) else getattr(resp, "parsed", None) or getattr(resp, "output", None) or resp
    except Exception as e:
     
        decision = RouterDecision(next_agent="FINISH", reason=f"supervisor error: {e}")

    
    if isinstance(decision, dict):
        decision = RouterDecision(**decision)

   
    forced_finish = False
    if state["turn_count"] > MAX_TURNS:
        forced_finish = True
        forced_reason = "turn_count exceeded MAX_TURNS"
    elif decision.next_agent in ("writer", "critic"):
        if state.get("revision_count", 0) >= MAX_REVISIONS and state.get("draft"):
            forced_finish = True
            forced_reason = "max revisions reached and draft exists"

    if forced_finish:
        final_next = "FINISH"
        final_reason = forced_reason
    else:
        final_next = decision.next_agent
        final_reason = decision.reason

   
    state["next_agent"] = final_next
    log_entry = f"{datetime.utcnow().isoformat()}Z - Supervisor -> {final_next}: {final_reason}"
    
    if "execution_logs" not in state or state["execution_logs"] is None:
        state["execution_logs"] = []
    state["execution_logs"] = state["execution_logs"] + [log_entry]

    
    return {"next_agent": final_next, "turn_count": state["turn_count"], "execution_logs": [log_entry]}



def researcher_node(state: TeamState):
    """Search the web (ONLY this agent may), condense to notes."""
    query = state.get("task", "")
    
    try:
        resp = search_tool.invoke({"query": query})
        results = resp.get("results", []) if isinstance(resp, dict) else resp
    except Exception:
        results = []

    
    parts = []
    for r in results:
        title = r.get("title") if isinstance(r, dict) else getattr(r, "title", None)
        snippet = r.get("content") or r.get("snippet") if isinstance(r, dict) else getattr(r, "content", None)
        url = r.get("url") if isinstance(r, dict) else getattr(r, "url", None)
        parts.append(f"Title: {title or '(no title)'}\nSnippet: {snippet or '(no snippet)'}\nURL: {url or '(no url)'}")
    raw = "\n\n---\n\n".join(parts) if parts else "(no search results)"

    
    prompt = f"Task: {query}\n\nSearch results:\n{raw}\n\nPlease append concise, sourceable findings as a short note."
    notes_text = run_persona("researcher", prompt)

   
    log_entry = f"{datetime.utcnow().isoformat()}Z - Researcher ran search and added notes."
    return {"research_notes": [notes_text], "execution_logs": [log_entry]}


def analyst_node(state: TeamState):
    """Turn raw notes into analysis."""
    notes = state.get("research_notes", [])
    recent = "\n\n".join(notes[-6:]) if notes else "(no notes)"
    prompt = (
        f"Task: {state.get('task','(no task)')}\n\n"
        f"Recent research notes:\n{recent}\n\n"
        "Produce a short analysis summary focusing on relevance, contradictions, and gaps."
    )
    analysis_text = run_persona("analyst", prompt)
    log_entry = f"{datetime.utcnow().isoformat()}Z - Analyst produced analysis."
    return {"analysis": analysis_text, "execution_logs": [log_entry]}


def writer_node(state: TeamState):
    """Write the draft — or REVISE it if a critique is present."""
    task = state.get("task", "(no task)")
    research_notes = state.get("research_notes", [])
    analysis = state.get("analysis", "")
    prev_draft = state.get("draft", "")
    critique = state.get("critique", "") or ""
    revising = critique.strip().upper().startswith("REVISE")

    recent_notes = "\n\n".join(research_notes[-6:]) if research_notes else "(no notes)"

    if revising:
        prompt = (
            f"Task: {task}\n\n"
            f"Analysis:\n{analysis}\n\n"
            f"Research notes (recent):\n{recent_notes}\n\n"
            f"Previous draft:\n{prev_draft}\n\n"
            f"Critique (instructions):\n{critique}\n\n"
            "Revise the draft accordingly and produce a new full draft. Overwrite the draft."
        )
    else:
        prompt = (
            f"Task: {task}\n\n"
            f"Analysis:\n{analysis}\n\n"
            f"Research notes (recent):\n{recent_notes}\n\n"
            "Produce a full structured draft. Overwrite any existing draft."
        )

    new_draft = run_persona("writer", prompt)
    revision_inc = 1 if revising else 0
    new_revision_count = state.get("revision_count", 0) + revision_inc

    log_entry = f"{datetime.utcnow().isoformat()}Z - Writer produced draft (revising={revising})."
    return {
        "draft": new_draft,
        "critique": "",
        "revision_count": new_revision_count,
        "execution_logs": [log_entry],
    }


def critic_node(state: TeamState):
    """Review the draft against the research notes."""
    draft = state.get("draft", "")
    research_notes = state.get("research_notes", [])
    recent = "\n\n".join(research_notes[-6:]) if research_notes else "(no notes)"
    prompt = (
        f"Task: {state.get('task','(no task)')}\n\n"
        f"Draft:\n{draft}\n\n"
        f"Research notes (recent):\n{recent}\n\n"
        "Evaluate the draft. If acceptable, reply with 'APPROVED'. "
        "If it needs changes, reply with 'REVISE: <short list of fixes>'."
    )
    critique_text = run_persona("critic", prompt).strip()

    if critique_text.upper().startswith("APPROVED"):
        final_critique = "APPROVED"
    elif critique_text.upper().startswith("REVISE"):
        final_critique = critique_text
    else:
        final_critique = f"REVISE: {critique_text}"

    log_entry = f"{datetime.utcnow().isoformat()}Z - Critic reviewed draft: {final_critique.splitlines()[0][:120]}"
    return {"critique": final_critique, "execution_logs": [log_entry]}

  

# ============================================================
# STEP 6 — ROUTING FUNCTION + WIRE THE GRAPH
# ============================================================
# The conditional-edge function is now TRIVIAL — it just reads the
# supervisor's decision:
#
#     def route_from_supervisor(state) -> str:
#         return state["next_agent"]
#
# Compare with Day 1, where all decision logic lived inside
# quality_router. The intelligence MOVED from the edge into a node.
#
# Wiring checklist:
# 1. add all five nodes
# 2. START → supervisor
# 3. add_conditional_edges("supervisor", route_from_supervisor,
#        {"researcher": "researcher", "analyst": "analyst",
#         "writer": "writer", "critic": "critic", "FINISH": END})
# 4. EVERY worker gets an edge BACK to supervisor — the
#    hub-and-spoke shape that defines the supervisor pattern.
#    (A for-loop over the four worker names is idiomatic.)

# TODO: route_from_supervisor + graph wiring


def route_from_supervisor(state: TeamState) -> str:
    return state.get("next_agent", "FINISH")


graph = StateGraph(checkpointer=InMemorySaver())


graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("analyst", analyst_node)
graph.add_node("writer", writer_node)
graph.add_node("critic", critic_node)


graph.add_edge(START, "supervisor")

graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "critic": "critic",
        "FINISH": END,
    },
)


for worker in ("researcher", "analyst", "writer", "critic"):
    graph.add_edge(worker, "supervisor")

# graph is now wired; you can run it with an initial TeamState when ready
# example (do not run unless you prepared initial state and env):
# initial_state = TeamState(
#     task="Your research question",
#     research_notes=[],
#     analysis="",
#     draft="",
#     critique="",
#     revision_count=0,
#     turn_count=0,
#     next_agent="researcher",
#     execution_logs=[]
# )
# result = graph.run(initial_state)


# ============================================================
# STEP 7 — COMPILE, VISUALIZE, RUN
# ============================================================
# Same as Day 1: compile with InMemorySaver, print the Mermaid
# diagram (it should look like a STAR, not Day 1's chain), stream
# with stream_mode="values" and a thread_id, print the final draft.
#
# EXPERIMENT 1: set MAX_REVISIONS = 0. What happens to quality?
# EXPERIMENT 2: delete guardrail (a) and make the critic always
#   say REVISE. Watch the turn cap save you — then delete guardrail
#   (b) too and meet your old friend GraphRecursionError.
# EXPERIMENT 3: swap the analyst's persona for a terrible one
#   ("you are vague and generic"). How far does the damage spread
#   through the team? This is why persona boundaries matter.

if __name__ == "__main__":
    initial_state = {
        "task": "Should our company adopt multi-agent AI systems in 2026?",
        "research_notes": [],
        "analysis": "",
        "draft": "",
        "critique": "",
        "revision_count": 0,
        "turn_count": 0,
        "next_agent": "",
        "execution_logs": [],
    }
    # TODO: compile, visualize, stream, print final draft + stats

    
    try:
       
        if hasattr(graph, "compile"):
            graph.compile()
    except Exception:
        pass

    
    try:
        mermaid = graph.to_mermaid()
        print("=== Graph (Mermaid) ===")
        print(mermaid)
    except Exception:
        print("=== Graph diagram unavailable (to_mermaid not supported) ===")

    
    print("\n=== Running graph (stream_mode='values') ===")
    try:
        run_result = graph.run(
            initial_state,
            stream_mode="values",
            thread_id="day2_run_1",
        )
        
        final_state = run_result if isinstance(run_result, dict) else getattr(run_result, "final_state", None) or getattr(run_result, "state", None) or run_result
    except Exception as e:
        print("Graph run failed:", e)
        final_state = None


    print("\n=== Final Outputs ===")
    if final_state and isinstance(final_state, dict):
        print("Final draft:\n")
        print(final_state.get("draft", "(no draft)"))
        print("\n--- Stats ---")
        print("revision_count:", final_state.get("revision_count"))
        print("turn_count:", final_state.get("turn_count"))
        logs = final_state.get("execution_logs", [])
        print("execution_logs_count:", len(logs))
       
        print("\nLast execution logs:")
        for line in logs[-8:]:
            print(line)
    else:
        print("No final state available. Check graph.run return shape or errors above.")

# ============================================================
# SELF-CHECK before you look at the solution
# ============================================================
# [ ] I can explain the supervisor pattern in one sentence
# [ ] My routing function reads state — the DECISION was made in a node
# [ ] research_notes appends; draft overwrites; I know why each
# [ ] The writer RESETS critique — I can explain what breaks if not
#     (hint: what does the supervisor see on the turn after a revision?)
# [ ] Only researcher_node touches search_tool
# [ ] My supervisor has BOTH guardrails, and I triggered EXPERIMENT 2
# [ ] My Mermaid diagram is a star: supervisor in the middle
# [ ] I can name one task where Day 1's single agent is the BETTER
#     design (multi-agent is not free: more calls, more latency,
#     more places to break — coordination must earn its cost)
#
# Stuck? Debugging order that works:
#   1. stream_mode="updates" — watch each supervisor decision + reason
#   2. print the status summary your supervisor_node builds — is the
#      LLM seeing an accurate picture of the blackboard?
#   3. check your conditional-edge dict covers ALL five decisions
#   4. only THEN open day2_lab_solution.py
# ============================================================
