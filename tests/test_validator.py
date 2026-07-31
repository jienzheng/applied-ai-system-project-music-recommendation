from src.validator import validate_recommendations, fallback_recommendations

CANDIDATES = [
    {"title": "Bohemian Rhapsody", "artist": "Queen", "genre": "rock"},
    {"title": "Take Five", "artist": "Dave Brubeck Quartet", "genre": "jazz"},
    {"title": "Clair de Lune", "artist": "Claude Debussy", "genre": "classical"},
]


def test_accepts_valid_songs():
    recs = [
        {"title": "Bohemian Rhapsody", "artist": "Queen", "reason": "epic"},
        {"title": "take five", "artist": "DAVE BRUBECK QUARTET", "reason": "cool jazz"},
    ]
    valid, invalid = validate_recommendations(recs, CANDIDATES)

    assert len(valid) == 2
    assert invalid == []
    assert valid[0]["genre"] == "rock"
    assert valid[0]["reason"] == "epic"


def test_rejects_hallucinated_songs():
    recs = [
        {"title": "Bohemian Rhapsody", "artist": "Queen", "reason": "epic"},
        {"title": "A Song That Does Not Exist", "artist": "Nobody", "reason": "made up"},
    ]
    valid, invalid = validate_recommendations(recs, CANDIDATES)

    assert len(valid) == 1
    assert len(invalid) == 1
    assert invalid[0]["title"] == "A Song That Does Not Exist"


def test_rejects_song_with_mismatched_artist():
    recs = [{"title": "Bohemian Rhapsody", "artist": "Someone Else", "reason": "epic"}]
    valid, invalid = validate_recommendations(recs, CANDIDATES)

    assert valid == []
    assert len(invalid) == 1


def test_fallback_returns_top_k_candidates_with_reason():
    fallback = fallback_recommendations(CANDIDATES, k=2)

    assert len(fallback) == 2
    assert fallback[0]["title"] == "Bohemian Rhapsody"
    assert "reason" in fallback[0]
    assert fallback[0]["reason"] != ""
