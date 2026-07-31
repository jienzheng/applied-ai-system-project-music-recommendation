"""Generation layer: asks Claude to pick the best songs from retrieved candidates."""

import json
import logging
import os
from typing import List, Dict

import anthropic

from .validator import validate_recommendations, fallback_recommendations

logger = logging.getLogger(__name__)

MODEL_NAME = "claude-sonnet-4-6"
MAX_TOKENS = 2048
NUM_RECOMMENDATIONS = 3

SYSTEM_PROMPT = """You are a music recommendation assistant. You will be given a user's \
natural-language request and a list of candidate songs retrieved from a catalog.

Rules:
- You may ONLY recommend songs that appear in the provided candidate list.
- Never invent, modify, or recommend a song that is not in the candidate list.
- Choose exactly {n} songs that best match the user's request.
- For each choice, give a short 1-2 sentence explanation of why it fits the request.

Respond with ONLY valid JSON in this exact shape, no other text:
{{"recommendations": [{{"title": "...", "artist": "...", "reason": "..."}}, ...]}}
""".format(n=NUM_RECOMMENDATIONS)


def _format_candidates(candidates: List[Dict]) -> str:
    lines = []
    for song, score in candidates:
        mood_tags = ", ".join(song.get("mood_tags", []))
        instrumental = "instrumental" if song.get("instrumental") else "has vocals"
        lines.append(
            f"- \"{song['title']}\" by {song['artist']} | genre: {song['genre']} | "
            f"mood: {mood_tags} | tempo: {song['tempo']} | {instrumental} | "
            f"{song['description']} (similarity: {score:.3f})"
        )
    return "\n".join(lines)


def _build_user_message(query: str, candidates: List[Dict], correction: str = "") -> str:
    message = f'User request: "{query}"\n\nCandidate songs:\n{_format_candidates(candidates)}'
    if correction:
        message += f"\n\n{correction}"
    return message


def _parse_response_json(text: str) -> List[Dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text)
    return data.get("recommendations", [])


class MusicRecommenderError(Exception):
    """Raised for user-facing recommendation failures (API key, network, rate limit)."""


class Recommender:
    """Combines retrieval with Claude-generated, guardrail-validated recommendations."""

    def __init__(self, retriever, client: anthropic.Anthropic = None):
        self.retriever = retriever
        if client is None and not os.environ.get("ANTHROPIC_API_KEY"):
            raise MusicRecommenderError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        try:
            self.client = client or anthropic.Anthropic()
        except Exception as exc:
            logger.error("Failed to initialize Anthropic client: %s", exc)
            raise MusicRecommenderError(
                "Could not initialize the Anthropic client. Check that ANTHROPIC_API_KEY "
                "is set in your environment or .env file."
            ) from exc

    def _call_model(self, user_message: str) -> str:
        try:
            response = self.client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.AuthenticationError as exc:
            logger.error("Authentication error calling Anthropic API: %s", exc)
            raise MusicRecommenderError(
                "Authentication failed. Check that ANTHROPIC_API_KEY is set correctly."
            ) from exc
        except anthropic.RateLimitError as exc:
            logger.error("Rate limit hit calling Anthropic API: %s", exc)
            raise MusicRecommenderError(
                "Rate limit reached. Please wait a moment and try again."
            ) from exc
        except anthropic.APIConnectionError as exc:
            logger.error("Network error calling Anthropic API: %s", exc)
            raise MusicRecommenderError(
                "Network error reaching the Anthropic API. Check your internet connection."
            ) from exc
        except anthropic.APIStatusError as exc:
            logger.error("Anthropic API error: %s", exc)
            raise MusicRecommenderError(f"The Anthropic API returned an error: {exc}") from exc

        text = next((b.text for b in response.content if b.type == "text"), "")
        logger.debug("Raw model response: %s", text)
        return text

    def recommend(self, query: str) -> List[Dict]:
        """Retrieve candidates, ask Claude to pick the best k, and validate the result."""
        logger.info("Received query: %r", query)
        candidates = self.retriever.retrieve(query)
        logger.info(
            "Retrieved %d candidates with scores: %s",
            len(candidates),
            [(s["title"], round(sc, 4)) for s, sc in candidates],
        )
        candidate_songs = [song for song, _ in candidates]

        user_message = _build_user_message(query, candidates)
        raw_text = self._call_model(user_message)

        try:
            recommendations = _parse_response_json(raw_text)
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("Failed to parse model response as JSON: %s", exc)
            recommendations = []

        valid, invalid = validate_recommendations(recommendations, candidate_songs)
        logger.info("Validation result: %d valid, %d invalid", len(valid), len(invalid))

        if len(valid) >= NUM_RECOMMENDATIONS:
            return valid[:NUM_RECOMMENDATIONS]

        # Re-prompt once with a correction message.
        logger.warning("Validation failed or incomplete; re-prompting model once")
        correction = (
            "Your previous response included songs not present in the candidate list above. "
            "You must choose ONLY from the candidate list provided. Please try again."
        )
        retry_message = _build_user_message(query, candidates, correction=correction)
        retry_raw = self._call_model(retry_message)

        try:
            retry_recommendations = _parse_response_json(retry_raw)
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("Failed to parse retry response as JSON: %s", exc)
            retry_recommendations = []

        retry_valid, retry_invalid = validate_recommendations(retry_recommendations, candidate_songs)
        logger.info(
            "Retry validation result: %d valid, %d invalid", len(retry_valid), len(retry_invalid)
        )

        if len(retry_valid) >= NUM_RECOMMENDATIONS:
            return retry_valid[:NUM_RECOMMENDATIONS]

        # Fall back to the top retrieved candidates.
        logger.error("Validation failed twice; falling back to top retrieved candidates")
        return fallback_recommendations(candidate_songs, k=NUM_RECOMMENDATIONS)
