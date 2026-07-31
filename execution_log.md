# Execution Log

Real, captured terminal output from this repo, run on macOS with Python 3.11.15 in a fresh `.venv`. Nothing below is a summary or a fabricated transcript — the pytest section is pasted verbatim from an actual run in this session.

## Reproducing from scratch

```bash
git clone https://github.com/jienzheng/applied-ai-system-project-music-recommendation.git
cd applied-ai-system-project-music-recommendation

python3 -m venv .venv
source .venv/bin/activate      # Mac/Linux; .venv\Scripts\activate on Windows

pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=<your key>

python main.py
```

To run the test suite instead:

```bash
pytest tests/ -v
```

## Real output: `pytest tests/ -v`

Captured from an actual run in this session (`.venv`, Python 3.11.15, pytest 8.3.4):

```
============================= test session starts ==============================
platform darwin -- Python 3.11.15, pytest-8.3.4, pluggy-1.6.0 -- /Users/jien/CodePath/AI110/applied-ai-system-project-music-recommendation/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /Users/jien/CodePath/AI110/applied-ai-system-project-music-recommendation
plugins: anyio-4.14.2
collecting ... collected 17 items

tests/test_evaluation.py::test_retrieval_returns_nonempty_results_for_normal_query PASSED [  5%]
tests/test_evaluation.py::test_retrieval_handles_empty_query_gracefully PASSED [ 11%]
tests/test_evaluation.py::test_retrieval_handles_nonsense_query_gracefully PASSED [ 17%]
tests/test_evaluation.py::test_pipeline_passes_valid_first_try_response PASSED [ 23%]
tests/test_evaluation.py::test_pipeline_catches_hallucinated_song_and_recovers_on_retry PASSED [ 29%]
tests/test_evaluation.py::test_pipeline_falls_back_to_top_retrieved_songs_after_two_failures PASSED [ 35%]
tests/test_evaluation.py::test_pipeline_missing_api_key_raises_clear_error PASSED [ 41%]
tests/test_evaluation.py::test_pipeline_handles_network_error_gracefully PASSED [ 47%]
tests/test_evaluation.py::test_pipeline_handles_rate_limit_error_gracefully PASSED [ 52%]
tests/test_evaluation.py::test_confidence_stage_ordering_matches_pipeline_trust PASSED [ 58%]
tests/test_retriever.py::test_retrieve_returns_k_results PASSED          [ 64%]
tests/test_retriever.py::test_relevant_genre_ranks_higher PASSED         [ 70%]
tests/test_retriever.py::test_retrieve_respects_default_k PASSED         [ 76%]
tests/test_validator.py::test_accepts_valid_songs PASSED                 [ 82%]
tests/test_validator.py::test_rejects_hallucinated_songs PASSED          [ 88%]
tests/test_validator.py::test_rejects_song_with_mismatched_artist PASSED [ 94%]
tests/test_validator.py::test_fallback_returns_top_k_candidates_with_reason PASSED [100%]

============================== 17 passed in 2.96s ==============================
```

Note: on the first attempt at writing this suite, one test (`test_pipeline_catches_hallucinated_song_and_recovers_on_retry`) failed with `1 failed, 16 passed in 4.22s`, because of an incorrect assertion in the test itself (assumed every recommended song's confidence would be nonzero, which is false for a song with zero real cosine similarity to the query). The assertion was corrected and the suite was rerun; the `17 passed` output above is that final, confirmed run. See `EVALUATION.md` for the full explanation.

## Real output: `python main.py`

Captured from two real runs against the live Claude API with a real `ANTHROPIC_API_KEY`. The configured Anthropic account had **no API credit balance**, so every generation call returned a billing error — retrieval ran for real and succeeded in every case. Nothing below has been edited except redacting nothing (there is no key or secret in this output).

**Run 1 — blank input:**

```
2026-07-30 18:35:07,828 [INFO] src.retriever: Loaded 107 songs from data/songs.json
2026-07-30 18:35:07,829 [INFO] src.retriever: Loading sentence-transformer model: all-MiniLM-L6-v2
2026-07-30 18:35:10,212 [INFO] src.retriever: Built embedding index for 107 songs
Loading song catalog and building embedding index...
Ready. Describe the kind of music you're in the mood for (or type 'quit' to exit).

> > Goodbye!
```

(The blank line submitted at the first `>` prompt is silently re-prompted by `main.py`'s `if not query: continue` check — no retrieval or API call happens for it — then `quit` exits.)

**Run 2 — four real queries:**

```
2026-07-30 18:34:07,810 [INFO] src.retriever: Loaded 107 songs from data/songs.json
2026-07-30 18:34:07,810 [INFO] src.retriever: Loading sentence-transformer model: all-MiniLM-L6-v2
2026-07-30 18:34:10,677 [INFO] src.retriever: Built embedding index for 107 songs
Loading song catalog and building embedding index...
Ready. Describe the kind of music you're in the mood for (or type 'quit' to exit).

> > 2026-07-30 18:34:10,700 [INFO] src.recommender: Received query: 'chill instrumental music for studying'
2026-07-30 18:34:10,734 [INFO] src.recommender: Retrieved 8 candidates with scores: [('Study Session', 0.6893), ('Intro', 0.5048), ('Opus', 0.4651), ('Prelude in C Major', 0.457), ('Late Night Library', 0.4558), ('Homework Break', 0.449), ('Africa (Instrumental Study Mix)', 0.4392), ('Experience', 0.4358)]
2026-07-30 18:34:11,098 [ERROR] src.recommender: Anthropic API error: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.'}, 'request_id': 'req_011CdZGuuoz4QhySiEjCFxy5'}
Error: The Anthropic API returned an error: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.'}, 'request_id': 'req_011CdZGuuoz4QhySiEjCFxy5'}

> 2026-07-30 18:34:11,098 [INFO] src.recommender: Received query: 'asdkjfh qwoieur nonsense gibberish'
2026-07-30 18:34:11,250 [INFO] src.recommender: Retrieved 8 candidates with scores: [('Da Funk', 0.217), ('Desk Lamp Glow', 0.1993), ('Coffee Shop Window', 0.1753), ('Says', 0.1621), ('Superstition', 0.1592), ('Homework Break', 0.1591), ('The Message', 0.1581), ('Air on the G String', 0.1515)]
2026-07-30 18:34:11,557 [ERROR] src.recommender: Anthropic API error: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.'}, 'request_id': 'req_011CdZGuwbdznW18MsyMz4tg'}
Error: The Anthropic API returned an error: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.'}, 'request_id': 'req_011CdZGuwbdznW18MsyMz4tg'}

> 2026-07-30 18:34:11,557 [INFO] src.recommender: Received query: 'upbeat 90s Japanese city pop for a road trip'
2026-07-30 18:34:11,668 [INFO] src.recommender: Retrieved 8 candidates with scores: [('Africa', 0.4927), ('Midnight City', 0.4811), ('Around the World', 0.4543), ('Take Me Home, Country Roads', 0.4501), ('Music for Airports 1/1', 0.448), ('Levitating', 0.4452), ('The Message', 0.441), ('N.Y. State of Mind', 0.4401)]
2026-07-30 18:34:11,917 [ERROR] src.recommender: Anthropic API error: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.'}, 'request_id': 'req_011CdZGuyL524QJcu2oNx5aa'}
Error: The Anthropic API returned an error: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.'}, 'request_id': 'req_011CdZGuyL524QJcu2oNx5aa'}

> 2026-07-30 18:34:11,917 [INFO] src.recommender: Received query: 'sad happy angry calm loud quiet music'
2026-07-30 18:34:12,076 [INFO] src.recommender: Retrieved 8 candidates with scores: [('Intro', 0.4677), ('Happy', 0.4605), ('An Ending (Ascent)', 0.4492), ('Opus', 0.4459), ('Toxicity', 0.4248), ('Rainy Day Ambience', 0.4244), ('Weightless', 0.4236), ('Roygbiv', 0.4229)]
2026-07-30 18:34:12,226 [ERROR] src.recommender: Anthropic API error: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.'}, 'request_id': 'req_011CdZGv13FvtW89iheZjKJ2'}
Error: The Anthropic API returned an error: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.'}, 'request_id': 'req_011CdZGv13FvtW89iheZjKJ2'}

> Goodbye!
```

`[FILL IN: once the Anthropic account has API credit, rerun these same queries and append a "Run 3" transcript here showing real generated recommendations and confidence scores, then update README.md's Sample Interactions and EVALUATION.md's Human/Manual Evaluation + Confidence Scoring Summary to match]`
