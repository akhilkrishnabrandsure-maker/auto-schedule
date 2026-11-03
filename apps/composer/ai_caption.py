"""AI caption and description writing for the composer.

Uses Google Gemini when GEMINI_API_KEY is set, otherwise Claude when
ANTHROPIC_API_KEY is set.
"""

import json
import logging

import httpx
from django.conf import settings

logger = logging.getLogger(__name__)

TONES = {"friendly", "professional", "playful", "bold", "informative"}

SYSTEM_PROMPT = """You write social media posts for a brand's scheduling tool.
Write in the requested tone, in the same language as the brief.
Fit every target platform's character limit; when platforms differ, write for the strictest one.
For YouTube, write a description of a few short paragraphs; for other platforms, a caption.
Add a few relevant hashtags at the end unless the brief says not to.
Only include a title when one of the platforms needs it (YouTube, Pinterest); otherwise return an empty title.
Never invent facts, prices, dates or links that aren't in the brief or the current text.
Reply with JSON only: {"title": "...", "caption": "..."}"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "caption": {"type": "string"},
    },
    "required": ["title", "caption"],
    "additionalProperties": False,
}

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class AICaptionError(Exception):
    """Raised with a message that is safe to show the user."""


def provider() -> str:
    if settings.GEMINI_API_KEY:
        return "gemini"
    if settings.ANTHROPIC_API_KEY:
        return "claude"
    return ""


def is_enabled() -> bool:
    return bool(provider())


def generate_caption(*, brief: str, current_caption: str, title: str, platforms: list[dict], tone: str) -> dict:
    """Return {"title": str, "caption": str} written by the configured AI."""
    platform_lines = [
        f"- {p['platform']}" + (f" (limit {p['limit']} characters)" if p.get("limit") else "") for p in platforms
    ] or ["- not chosen yet"]
    parts = [
        f"Tone: {tone if tone in TONES else 'friendly'}",
        "Target platforms:\n" + "\n".join(platform_lines),
    ]
    if brief:
        parts.append(f"Brief:\n{brief}")
    if title:
        parts.append(f"Current title:\n{title}")
    if current_caption:
        parts.append(f"Current text (improve or rewrite it):\n{current_caption}")
    prompt = "\n\n".join(parts)

    which = provider()
    if which == "gemini":
        text = _gemini(prompt)
    elif which == "claude":
        text = _claude(prompt)
    else:
        raise AICaptionError("AI writing isn't set up. Add GEMINI_API_KEY to the server settings.")

    try:
        data = json.loads(_strip_code_fence(text))
    except json.JSONDecodeError as e:
        raise AICaptionError("The AI's reply couldn't be read. Try again.") from e
    if not isinstance(data, dict):
        raise AICaptionError("The AI's reply couldn't be read. Try again.")
    return {"title": str(data.get("title", "")).strip(), "caption": str(data.get("caption", "")).strip()}


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _gemini(prompt: str) -> str:
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    try:
        resp = httpx.post(
            GEMINI_URL.format(model=settings.GEMINI_MODEL),
            headers={"x-goog-api-key": settings.GEMINI_API_KEY},
            json=body,
            timeout=60.0,
        )
    except httpx.HTTPError as e:
        raise AICaptionError("Couldn't reach Gemini. Check the server's internet connection.") from e

    if resp.status_code in (400, 401, 403) and "API key" in resp.text:
        raise AICaptionError("The Gemini API key was rejected. Check GEMINI_API_KEY.")
    if resp.status_code == 404:
        raise AICaptionError(f"Gemini model '{settings.GEMINI_MODEL}' wasn't found. Check GEMINI_MODEL.")
    if resp.status_code == 429:
        raise AICaptionError("Gemini is busy or the quota is used up. Try again in a minute.")
    if resp.status_code >= 400:
        logger.warning("Gemini caption request failed: %s %s", resp.status_code, resp.text[:300])
        raise AICaptionError("Gemini couldn't write a caption right now. Try again.")

    data = resp.json()
    if data.get("promptFeedback", {}).get("blockReason"):
        raise AICaptionError("Gemini declined to write this post. Try rewording the brief.")
    try:
        candidate = data["candidates"][0]
        return "".join(p.get("text", "") for p in candidate["content"]["parts"])
    except (KeyError, IndexError, TypeError) as e:
        raise AICaptionError("Gemini declined to write this post. Try rewording the brief.") from e


def _claude(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY, timeout=60.0)
    try:
        response = client.beta.messages.create(
            model=settings.AI_CAPTION_MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            # If Claude declines, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError as e:
        raise AICaptionError("The Claude API key was rejected. Check ANTHROPIC_API_KEY.") from e
    except anthropic.RateLimitError as e:
        raise AICaptionError("Claude is busy right now. Try again in a minute.") from e
    except anthropic.APIStatusError as e:
        logger.warning("AI caption request failed: %s (request %s)", e.status_code, e.request_id)
        raise AICaptionError("Claude couldn't write a caption right now. Try again.") from e
    except anthropic.APIConnectionError as e:
        raise AICaptionError("Couldn't reach Claude. Check the server's internet connection.") from e

    if response.stop_reason == "refusal":
        raise AICaptionError("Claude declined to write this post. Try rewording the brief.")
    return next((b.text for b in response.content if b.type == "text"), "")
