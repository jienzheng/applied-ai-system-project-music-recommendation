"""End-to-end reliability tests for the retriever -> recommender -> validator pipeline.

All Anthropic API calls are mocked; no test here requires a real API key or
network access.
"""

import json
import re

import httpx
import numpy as np
import pytest
import anthropic

from src import retriever as retriever_module
from src.retriever import SongRetriever
from src.recommender import Recommender, MusicRecommenderError, NUM_RECOMMENDATIONS


class FakeSentenceTransformer:
    """Deterministic bag-of-words stand-in for a real sentence-transformer model."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.vocab = None

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        if self.vocab is None:
            all_words = []
            for t in texts:
                all_words.extend(re.findall(r"[a-z]+", t.lower()))
            self.vocab = sorted(set(all_words))

        matrix = np.zeros((len(texts), max(len(self.vocab), 1)))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
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
        "description": "A smooth instrumental jazz piece with saxophone for studying.",
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
        "description": "A peaceful classical piece for strings, calm and reflective.",
    },
    {
        "id": 4,
        "title": "Jazz Blue Skies",
        "artist": "Blue Combo",
        "genre": "jazz",
        "mood_tags": ["cool", "sophisticated"],
        "tempo": "medium",
        "instrumental": True,
        "description": "Another smooth jazz instrumental with piano and saxophone for studying.",
    },
    {
        "id": 5,
        "title": "Sunrise Workout",
        "artist": "Pulse Collective",
        "genre": "pop",
        "mood_tags": ["upbeat", "happy"],
        "tempo": "fast",
        "instrumental": False,
        "description": "An upbeat, happy pop track built for a morning workout.",
    },
]


@pytest.fixture(autouse=True)
def patch_sentence_transformer(monkeypatch):
    monkeypatch.setattr(retriever_module, "SentenceTransformer", FakeSentenceTransformer)


@pytest.fixture
def retriever():
    return SongRetriever(SONGS)


def _fake_response(payload: dict):
    """Build a minimal object shaped like an anthropic.types.Message."""
    block = type("Block", (), {"type": "text", "text": json.dumps(payload)})()
    return type("Response", (), {"content": [block]})()


class FakeClient:
    """A stand-in Anthropic client whose .messages.create() replays a scripted queue."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    class _Messages:
        def __init__(self, outer):
            self._outer = outer

        def create(self, **kwargs):
            self._outer.calls += 1
            next_item = self._outer._responses.pop(0)
            if isinstance(next_item, Exception):
                raise next_item
            return _fake_response(next_item)

    @property
    def messages(self):
        return self._Messages(self)


# ---------------------------------------------------------------------------
# 1. Retrieval returns non-empty results for a normal query
# ---------------------------------------------------------------------------


def test_retrieval_returns_nonempty_results_for_normal_query(retriever):
    results = retriever.retrieve("chill instrumental music for studying")
    assert len(results) > 0
    for song, score in results:
        assert isinstance(song, dict)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# 2. Retrieval handles an empty query gracefully (no crash)
# ---------------------------------------------------------------------------


def test_retrieval_handles_empty_query_gracefully(retriever):
    results = retriever.retrieve("")
    assert isinstance(results, list)
    assert len(results) == min(retriever_module.DEFAULT_TOP_K, len(SONGS))


# ---------------------------------------------------------------------------
# 3. Retrieval handles a nonsense query gracefully (no crash)
# ---------------------------------------------------------------------------


def test_retrieval_handles_nonsense_query_gracefully(retriever):
    results = retriever.retrieve("asdkjfh qwoieur zzxccv blorgatron")
    assert isinstance(results, list)
    assert len(results) == min(retriever_module.DEFAULT_TOP_K, len(SONGS))


# ---------------------------------------------------------------------------
# 4. Full pipeline: validator PASSES when the model's first response is fully valid
# ---------------------------------------------------------------------------


def test_pipeline_passes_valid_first_try_response(retriever):
    top = retriever.retrieve("smooth jazz saxophone instrumental for studying", k=5)
    real_titles = [song["title"] for song, _ in top[:NUM_RECOMMENDATIONS]]
    real_artists = [song["artist"] for song, _ in top[:NUM_RECOMMENDATIONS]]

    payload = {
        "recommendations": [
            {"title": t, "artist": a, "reason": "matches the mood"}
            for t, a in zip(real_titles, real_artists)
        ]
    }
    client = FakeClient([payload])
    recommender = Recommender(retriever, client=client)

    result = recommender.recommend("smooth jazz saxophone instrumental for studying")

    assert len(result) == NUM_RECOMMENDATIONS
    assert client.calls == 1  # no re-prompt needed
    assert all(song["confidence"] > 0 for song in result)


# ---------------------------------------------------------------------------
# 5. Full pipeline: validator catches a hallucinated song and the retry recovers
# ---------------------------------------------------------------------------


def test_pipeline_catches_hallucinated_song_and_recovers_on_retry(retriever):
    top = retriever.retrieve("upbeat happy pop workout music", k=5)
    real_titles = [song["title"] for song, _ in top[:NUM_RECOMMENDATIONS]]
    real_artists = [song["artist"] for song, _ in top[:NUM_RECOMMENDATIONS]]

    hallucinated_payload = {
        "recommendations": [
            {"title": "Totally Made Up Song", "artist": "Nobody Real", "reason": "fake"},
            {"title": real_titles[0], "artist": real_artists[0], "reason": "real one"},
        ]
    }
    corrected_payload = {
        "recommendations": [
            {"title": t, "artist": a, "reason": "corrected pick"}
            for t, a in zip(real_titles, real_artists)
        ]
    }
    client = FakeClient([hallucinated_payload, corrected_payload])
    recommender = Recommender(retriever, client=client)

    result = recommender.recommend("upbeat happy pop workout music")

    assert client.calls == 2  # first response failed validation, retry was made
    assert len(result) == NUM_RECOMMENDATIONS
    returned_titles = {song["title"] for song in result}
    assert "Totally Made Up Song" not in returned_titles
    assert all(song["title"] in real_titles for song in result)
    # Every confidence is a valid 0-1 score, and the top retrieval match (the only
    # candidate here with nonzero bag-of-words overlap with the query) scores highest.
    assert all(0.0 <= song["confidence"] <= 1.0 for song in result)
    by_title = {song["title"]: song["confidence"] for song in result}
    assert by_title[real_titles[0]] > 0
    assert by_title[real_titles[0]] == max(by_title.values())


# ---------------------------------------------------------------------------
# 6. Full pipeline: fallback returns the top-3 retrieved songs when both tries fail
# ---------------------------------------------------------------------------


def test_pipeline_falls_back_to_top_retrieved_songs_after_two_failures(retriever):
    query = "epic orchestral classical piece"
    top = retriever.retrieve(query, k=5)
    expected_fallback_titles = [song["title"] for song, _ in top[:NUM_RECOMMENDATIONS]]

    hallucinated_payload = {
        "recommendations": [
            {"title": "Fake Song One", "artist": "Ghost Artist", "reason": "fake"},
        ]
    }
    client = FakeClient([hallucinated_payload, hallucinated_payload])
    recommender = Recommender(retriever, client=client)

    result = recommender.recommend(query)

    assert client.calls == 2
    assert len(result) == NUM_RECOMMENDATIONS
    assert [song["title"] for song in result] == expected_fallback_titles
    # Fallback confidence should be lower than a first-try confidence at the same similarity.
    for song in result:
        assert song["confidence"] <= song["_similarity"]


# ---------------------------------------------------------------------------
# 7. Missing ANTHROPIC_API_KEY produces a clear error, not a crash
# ---------------------------------------------------------------------------


def test_pipeline_missing_api_key_raises_clear_error(retriever, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(MusicRecommenderError, match="ANTHROPIC_API_KEY"):
        Recommender(retriever)


# ---------------------------------------------------------------------------
# 8. Network/timeout errors are handled gracefully, not as an unhandled crash
# ---------------------------------------------------------------------------


def test_pipeline_handles_network_error_gracefully(retriever):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    connection_error = anthropic.APIConnectionError(request=request)

    client = FakeClient([connection_error])
    recommender = Recommender(retriever, client=client)

    with pytest.raises(MusicRecommenderError, match="Network error"):
        recommender.recommend("smooth jazz saxophone instrumental")


# ---------------------------------------------------------------------------
# 9. Rate limit errors are also handled gracefully
# ---------------------------------------------------------------------------


def test_pipeline_handles_rate_limit_error_gracefully(retriever):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    rate_limit_error = anthropic.RateLimitError("rate limited", response=response, body=None)

    client = FakeClient([rate_limit_error])
    recommender = Recommender(retriever, client=client)

    with pytest.raises(MusicRecommenderError, match="Rate limit"):
        recommender.recommend("hip-hop with confident lyrics")


# ---------------------------------------------------------------------------
# 10. Confidence scoring stages are correctly ordered (first_try > retry > fallback)
# ---------------------------------------------------------------------------


def test_confidence_stage_ordering_matches_pipeline_trust():
    from src.scorer import compute_confidence

    similarity = 0.9
    first_try = compute_confidence(similarity, "first_try")
    retry = compute_confidence(similarity, "retry")
    fallback = compute_confidence(similarity, "fallback")

    assert first_try > retry > fallback
    assert first_try == similarity  # first-try applies full weight (1.0)
