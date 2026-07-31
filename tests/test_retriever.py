import re

import numpy as np
import pytest

from src import retriever as retriever_module
from src.retriever import SongRetriever


class FakeSentenceTransformer:
    """A deterministic bag-of-words stand-in for a real sentence-transformer model.

    Avoids downloading a real model in tests while still producing embeddings
    where semantically similar text (shared words) scores higher.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.vocab = None  # fixed on the first (index-building) encode call

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        # Fix the vocabulary on the first call (building the song index) so
        # later query encodings share the same feature space/dimension.
        if self.vocab is None:
            all_words = []
            for t in texts:
                all_words.extend(re.findall(r"[a-z]+", t.lower()))
            self.vocab = sorted(set(all_words))

        matrix = np.zeros((len(texts), max(len(self.vocab), 1)))
        for i, t in enumerate(texts):
            words = re.findall(r"[a-z]+", t.lower())
            for w in words:
                if w in self.vocab:
                    matrix[i, self.vocab.index(w)] += 1.0

        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms

        return matrix


SONGS = [
    {
        "id": 1,
        "title": "Jazz Nightfall",
        "artist": "Blue Combo",
        "genre": "jazz",
        "mood_tags": ["cool", "relaxed"],
        "tempo": "medium",
        "instrumental": True,
        "description": "A smooth instrumental jazz piece with saxophone.",
    },
    {
        "id": 2,
        "title": "Metal Thunder",
        "artist": "Iron Storm",
        "genre": "metal",
        "mood_tags": ["aggressive", "intense"],
        "tempo": "fast",
        "instrumental": False,
        "description": "A heavy metal anthem with distorted guitars.",
    },
    {
        "id": 3,
        "title": "Classical Morning",
        "artist": "String Quartet",
        "genre": "classical",
        "mood_tags": ["peaceful", "calm"],
        "tempo": "slow",
        "instrumental": True,
        "description": "A peaceful classical piece for strings.",
    },
    {
        "id": 4,
        "title": "Jazz Blue Skies",
        "artist": "Blue Combo",
        "genre": "jazz",
        "mood_tags": ["cool", "sophisticated"],
        "tempo": "medium",
        "instrumental": True,
        "description": "Another smooth jazz instrumental with piano and saxophone.",
    },
]


@pytest.fixture(autouse=True)
def patch_sentence_transformer(monkeypatch):
    monkeypatch.setattr(retriever_module, "SentenceTransformer", FakeSentenceTransformer)


def test_retrieve_returns_k_results():
    retriever = SongRetriever(SONGS)
    results = retriever.retrieve("jazz saxophone", k=2)

    assert len(results) == 2
    for song, score in results:
        assert isinstance(song, dict)
        assert isinstance(score, float)


def test_relevant_genre_ranks_higher():
    retriever = SongRetriever(SONGS)
    results = retriever.retrieve("smooth jazz saxophone instrumental", k=4)

    top_song, top_score = results[0]
    assert top_song["genre"] == "jazz"

    # The jazz songs should score higher than the unrelated metal song.
    scores_by_id = {song["id"]: score for song, score in results}
    assert scores_by_id[1] > scores_by_id[2]
    assert scores_by_id[4] > scores_by_id[2]


def test_retrieve_respects_default_k():
    retriever = SongRetriever(SONGS)
    results = retriever.retrieve("music")

    assert len(results) == min(retriever_module.DEFAULT_TOP_K, len(SONGS))
