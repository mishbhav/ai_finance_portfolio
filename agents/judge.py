from agents.llm_client import call_llm
import json
import re
from datetime import datetime, timezone

SCORING_PROMPT = """You are an evidence-grounding auditor. You will be given
an investment argument and the evidence it was supposed to be based on.
Your only job is to judge how well the argument's claims are actually
supported by the evidence — not whether you agree with the conclusion.

Respond with ONLY valid JSON, no other text, in this exact shape:
{"score": <int 1-10>, "justification": "<one sentence>"}

A score of 1-3 means the argument makes claims the evidence doesn't support.
A score of 8-10 means every claim traces clearly back to the evidence given."""

FALLBACK_SCORE = {"score": 0, "justification": "Error scoring this argument — treated as unsupported."}


def _citation_freshness_penalty(evidence_metadata: list[dict]) -> int:
    """Returns a penalty (0, 1, or 2) to subtract from a groundedness
    score based on how stale the most recent cited evidence is. No
    penalty if metadata is missing/unparseable — this is an enhancement,
    not a requirement, and should never cause a scoring failure on its
    own. Corpus documents with no 'published' field (e.g. the original
    .txt corpus, as opposed to ingested news) are treated as evergreen
    and never penalized."""
    published_dates = []
    for meta in evidence_metadata or []:
        published = (meta or {}).get("published")
        if not published:
            continue
        try:
            published_dates.append(datetime.fromisoformat(published.replace("Z", "+00:00")))
        except (ValueError, AttributeError):
            continue

    if not published_dates:
        return 0

    most_recent = max(published_dates)
    age_days = (datetime.now(timezone.utc) - most_recent).days

    if age_days > 90:
        return 2
    if age_days > 30:
        return 1
    return 0


def score_argument(argument: str, evidence: list[str], evidence_metadata: list[dict] = None) -> dict:
    evidence_block = "\n\n".join(f"- {chunk}" for chunk in evidence)
    user_message = f"Argument:\n{argument}\n\nEvidence it was based on:\n{evidence_block}"

    try:
        raw_response = call_llm(system_prompt=SCORING_PROMPT, user_message=user_message)
        cleaned_response = raw_response.strip()

        match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned_response, re.DOTALL)
        stripped_response = match.group(1) if match else cleaned_response

        dict_response = json.loads(stripped_response)

        if not isinstance(dict_response, dict):
            raise ValueError("Parsed JSON is not an object")
        if "score" not in dict_response or "justification" not in dict_response:
            raise ValueError("Missing required keys")

        score = int(dict_response["score"])
        score = max(1, min(10, score))
        justification = str(dict_response["justification"])

        penalty = _citation_freshness_penalty(evidence_metadata)
        if penalty > 0:
            score = max(1, score - penalty)
            justification += f" (score reduced by {penalty} for stale evidence)"

        return {"score": score, "justification": justification}

    except Exception as e:
        print(f"Failed to score argument: {e}")
        return FALLBACK_SCORE.copy()


SYNTHESIS_PROMPT = """You are a portfolio decision judge. You will be given
several investment arguments along with a groundedness score for each
(higher score = better supported by evidence). Weigh your synthesis toward
higher-scored arguments — a confident but poorly-grounded argument (low
score) should carry less weight than a well-grounded one, even if it's
less assertively written.

Produce a brief final recommendation and explain which argument(s)
most influenced your decision and why."""


def synthesize_decision(query: str, scored_arguments: list[dict]) -> str:
    user_message_parts = [
        f"User Query: {query}\n",
        "Please synthesize a decision based on the following evaluated investment arguments:\n"
    ]

    for i, arg in enumerate(scored_arguments, start=1):
        argument_block = (
            f"--- Argument {i} ---\n"
            f"Stance: {arg.get('stance', 'N/A')}\n"
            f"Groundedness Score: {arg.get('score', 0)}/10\n"
            f"Argument: {arg.get('argument', '')}\n"
            f"Score Justification: {arg.get('justification', 'No justification provided.')}\n"
        )
        user_message_parts.append(argument_block)

    user_message = "\n".join(user_message_parts)

    final_synthesis = call_llm(
        system_prompt=SYNTHESIS_PROMPT,
        user_message=user_message
    )

    return final_synthesis