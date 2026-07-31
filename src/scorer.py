"""Confidence scoring for validated recommendations.

Combines two signals into a single 0-1 confidence score per recommendation:
- the retrieval similarity score (higher cosine similarity = more confident)
- how the recommendation made it into the final output (first-try validation
  is trusted fully, a corrected retry less so, and a fallback to raw
  retrieval results the least, since no explanation was actually validated).
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Multiplier applied to similarity depending on how the recommendation was produced.
STAGE_CONFIDENCE_WEIGHT = {
    "first_try": 1.0,
    "retry": 0.7,
    "fallback": 0.4,
}


def compute_confidence(similarity: float, stage: str) -> float:
    """Combine retrieval similarity with validation stage into one 0-1 confidence score."""
    weight = STAGE_CONFIDENCE_WEIGHT.get(stage, STAGE_CONFIDENCE_WEIGHT["fallback"])
    clamped_similarity = max(0.0, min(1.0, similarity))
    return round(clamped_similarity * weight, 4)


def attach_confidence(songs: List[Dict], stage: str) -> List[Dict]:
    """Return a copy of each song dict annotated with a 'confidence' score, logging each."""
    scored = []
    for song in songs:
        similarity = song.get("_similarity", 0.0)
        confidence = compute_confidence(similarity, stage)
        scored.append({**song, "confidence": confidence})
        logger.info(
            "Confidence for %r by %r: %.4f (stage=%s, similarity=%.4f)",
            song.get("title"),
            song.get("artist"),
            confidence,
            stage,
            similarity,
        )
    return scored
