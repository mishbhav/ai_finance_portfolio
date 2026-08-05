import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "decisions.jsonl"


def log_decision(query: str, result: dict) -> None:
    """Appends one decision to the log. Called by the orchestrator after
    every full_debate run — not quick_lookup, since there's no decision
    to evaluate in that case."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "plan": result.get("plan"),
        "decision": result.get("decision"),
        "scored_arguments": result.get("scored_arguments"),
        "simulation_summary": result.get("simulation_summary"),
        "outcome": None,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_recent_decisions(limit: int = 5) -> list[dict]:
    """Returns the most recent N logged decisions, newest first.
    Used by the orchestrator to give itself memory of past runs."""
    if not LOG_PATH.exists():
        return []

    with open(LOG_PATH, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    recent_lines = lines[-limit:]
    entries = [json.loads(line) for line in recent_lines]
    entries.reverse()  # newest first
    return entries


def update_outcome(timestamp: str, outcome: str) -> bool:
    """Finds the decision matching `timestamp` and sets its outcome field.
    Rewrites the whole file — fine at this log's expected scale (a CV demo
    logging dozens/hundreds of entries), not intended to scale to a
    production-sized log without moving to a real database."""
    if not LOG_PATH.exists():
        return False

    with open(LOG_PATH, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    found = False
    for entry in entries:
        if entry["timestamp"] == timestamp:
            entry["outcome"] = outcome
            found = True
            break

    if found:
        with open(LOG_PATH, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    return found