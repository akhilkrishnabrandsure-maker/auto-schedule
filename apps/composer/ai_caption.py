"""AI caption and description writing for the composer, powered by Claude."""

import json
import logging

import anthropic
from django.conf import settings

logger = logging.getLogger(__name__)

TONES = {"friendly", "professional", "playful", "bold", "informative"}

SYSTEM_PROMPT = """You write social media posts for a brand's scheduling tool.
Write in the requested tone, in the same language as the brief.
Fit every target platform's character limit; when platforms differ, write for the strictest one.
For YouTube, write a description of a few short paragraphs; for other platforms, a caption.
Add a few relevant hashtags at the end unless the brief says not to.
Only include a title when one of the platforms needs it (YouTube, Pinterest); otherwise return an empty title.
Never invent facts, prices, dates or links that aren't in the brief or the current text."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "caption": {"type": "string"},
    },
    "required": ["title", "caption"],
    "additionalProperties": False,
}


class AICaptionError(Exception):
    """Raised with a message that is safe to show the user."""


def is_enabled() -> bool:
    return bool(settings.ANTHROPIC_API_KEY)


def generate_caption(*, brief: str, current_caption: str, title: str, platforms: list[dict], tone: str) -> dict:
    """Return {"title": str, "caption": str} written by Claude."""
    if not is_enabled():
        raise AICaptionError("AI writing isn't set up. Add ANTHROPIC_API_KEY to the server settings.")

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

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY, timeout=60.0)
    try:
        response = client.beta.messages.create(
            model=settings.AI_CAPTION_MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "\n\n".join(parts)}],
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

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise AICaptionError("Claude's reply couldn't be read. Try again.") from e
    return {"title": data.get("title", "").strip(), "caption": data.get("caption", "").strip()}
