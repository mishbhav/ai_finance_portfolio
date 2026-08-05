import google.generativeai as genai
from pathlib import Path
import sys

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from config.settings import MODEL_API_KEY

genai.configure(api_key=MODEL_API_KEY)

_model = genai.GenerativeModel(model_name="gemini-2.0-flash")

def call_llm(system_prompt: str, user_message: str, max_tokens: int = 512) -> str:
    """Single-purpose wrapper: send a system prompt + user message,
    get back the model's text response as a plain string.
    Currently backed by Gemini — swap this file's internals only
    to change provider; callers never need to change."""
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system_prompt,
    )
    response = model.generate_content(
        user_message,
        generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens),
    )
    return response.text