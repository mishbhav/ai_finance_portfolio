import anthropic
from pathlib import Path
import sys

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from config.settings import MODEL_API_KEY

_client = anthropic.Anthropic(api_key=MODEL_API_KEY)

_MODEL = "claude-sonnet-5"


def call_llm(system_prompt: str, user_message: str, max_tokens: int = 512) -> str:
    """Single-purpose wrapper: send a system prompt + user message,
    get back the model's text response as a plain string.
    Currently backed by Anthropic — swap this file's internals only
    to change provider; callers never need to change."""
    response = _client.messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    # response.content is a list of content blocks; join any text blocks
    return "".join(block.text for block in response.content if block.type == "text")