"""Guardrail that verifies model recommendations are grounded in retrieved candidates."""

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def _key(title: str, artist: str) -> Tuple[str, str]:
    return (title.strip().lower(), artist.strip().lower())


def validate_recommendations(
    recommendations: List[Dict], candidates: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """Split recommendations into (valid, invalid) based on the retrieved candidate set.

    A recommendation is valid only if its title+artist matches a retrieved candidate
    exactly (case-insensitive). Invalid entries are logged as validation failures.
    """
    candidate_keys = {_key(c["title"], c["artist"]): c for c in candidates}

    valid: List[Dict] = []
    invalid: List[Dict] = []

    for rec in recommendations:
        title = rec.get("title", "")
        artist = rec.get("artist", "")
        match = candidate_keys.get(_key(title, artist))
        if match is not None:
            merged = {**match, "reason": rec.get("reason", "")}
            valid.append(merged)
        else:
            invalid.append(rec)
            logger.warning(
                "Validation failure: recommended song not in retrieved candidates: %r by %r",
                title,
                artist,
            )

    return valid, invalid


def fallback_recommendations(candidates: List[Dict], k: int = 3) -> List[Dict]:
    """Fall back to the top-k retrieved candidates with a generic note."""
    logger.warning("Falling back to top-%d retrieved candidates after repeated validation failure", k)
    fallback = []
    for song in candidates[:k]:
        fallback.append(
            {
                **song,
                "reason": "Selected as a top retrieval match (model recommendation could not be validated).",
            }
        )
    return fallback
