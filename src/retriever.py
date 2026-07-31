"""Embedding-based retrieval over the local song dataset."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 8


def load_songs(path: str) -> List[Dict]:
    """Load the song dataset from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        songs = json.load(f)
    logger.info("Loaded %d songs from %s", len(songs), path)
    return songs


def _song_to_text(song: Dict) -> str:
    """Combine a song's metadata into one string for embedding."""
    mood_tags = ", ".join(song.get("mood_tags", []))
    instrumental = "instrumental" if song.get("instrumental") else "with vocals"
    return (
        f"{song['title']} by {song['artist']}. Genre: {song['genre']}. "
        f"Mood: {mood_tags}. Tempo: {song['tempo']}. {instrumental}. "
        f"{song['description']}"
    )


class SongRetriever:
    """Embeds songs in memory and retrieves the closest matches to a query."""

    def __init__(self, songs: List[Dict], model_name: str = MODEL_NAME):
        self.songs = songs
        logger.info("Loading sentence-transformer model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        texts = [_song_to_text(song) for song in songs]
        self.embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        logger.info("Built embedding index for %d songs", len(songs))

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> List[Tuple[Dict, float]]:
        """Return the top-k songs most similar to the query, with similarity scores."""
        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:k]
        results = [(self.songs[i], float(scores[i])) for i in top_indices]
        logger.debug(
            "Retrieved candidates for query %r: %s",
            query,
            [(s["title"], round(score, 4)) for s, score in results],
        )
        return results
