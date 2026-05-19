import os
import re
import json
import sys
from pathlib import Path

# When run directly as `python agent/agent.py`, Python inserts agent/ into
# sys.path[0], which makes `from agent.retriever` find agent.py itself
# (not the package). Remove the script's own directory first, then add
# the project root so package-style imports resolve correctly.
_root = Path(__file__).parent.parent
_agent_dir = Path(__file__).parent
for _p in (str(_agent_dir), str(_agent_dir.resolve())):
    while _p in sys.path:
        sys.path.remove(_p)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from groq import Groq
from dotenv import load_dotenv

from agent.retriever import retrieve, VALID_URLS
from agent.prompts import SYSTEM_PROMPT, format_catalog_context

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def _build_search_query(messages: list[dict]) -> str:
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    recent = user_msgs[-2:] if len(user_msgs) >= 2 else user_msgs
    return " ".join(recent)


def _parse_recommendations(text: str) -> list[dict]:
    match = re.search(r"<recommendations>(.*?)</recommendations>", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(1).strip())
    except (json.JSONDecodeError, ValueError):
        return []


def _validate_urls(recommendations: list[dict]) -> list[dict]:
    return [r for r in recommendations if r.get("url", "") in VALID_URLS]


def _clean_reply(text: str) -> str:
    text = re.sub(r"<recommendations>.*?</recommendations>", "", text, flags=re.DOTALL)
    text = re.sub(r"<end_conversation\s*/>", "", text)
    return text.strip()


def run_agent(messages: list[dict]) -> dict:
    try:
        query = _build_search_query(messages)
        items = retrieve(query, top_k=15)
        catalog_context = format_catalog_context(items)
        system_prompt = SYSTEM_PROMPT.format(catalog_context=catalog_context)

        response = _client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "system", "content": system_prompt}] + messages,
        )

        raw_text = response.choices[0].message.content
        recommendations = _validate_urls(_parse_recommendations(raw_text))
        end_of_conversation = bool(re.search(r"<end_conversation\s*/>", raw_text))
        reply = _clean_reply(raw_text)

        return {
            "reply": reply,
            "recommendations": recommendations,
            "end_of_conversation": end_of_conversation,
        }

    except Exception:
        return {
            "reply": "I encountered an error. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        }


if __name__ == "__main__":
    messages = [{"role": "user", "content": "I need to hire a Java developer"}]
    result = run_agent(messages)
    print(result)
